#!/usr/bin/env python
"""CPU-only, multi-core reducer for the spooled real-data eval.

Phase 2 of the design in ``real_eval_ddp.py``. The training ranks have already
done the GPU work (DDP-parallel inference) and dumped one ``subj_*.npz``
(cropped uint8 ``pred`` + ``gt``) per subject into a spool directory. This
script is launched detached by rank 0 with ``CUDA_VISIBLE_DEVICES=""`` and runs
entirely off the critical path, while training gets on with the next epoch.

What makes it worth its own process:
  * Surface metrics are Euclidean distance transforms -- pure CPU, embarrassingly
    parallel across subjects, and by far the expensive half of this evaluation.
    A ``multiprocessing.Pool`` over subjects uses the cores that the GPU-bound
    training loop is not using.
  * Each worker loads its own npz from disk, so no multi-megabyte array is ever
    pickled through a pipe, and peak RSS is (pool size) x (one subject), not
    (all subjects).
  * fork-safety: this process is started fresh by ``subprocess``, so its pool
    never inherits a CUDA context or a live ``pymongo`` client (both of which
    are famously not fork-safe -- see the note at the top of
    ``curriculum_training_fast.py``).

Output matches ``eval_real_metrics.py``: ``per_class.csv`` + ``summary.json``,
plus a one-line append to ``<real_eval>/history.csv`` so the whole run's
trajectory is greppable without wandb.

Usage (normally invoked by RealEvalCallback, but standalone-friendly):
    python real_eval_reduce.py --spool DIR --out DIR --n-classes 18 \
        --expect-ranks 4 --expect-subjects 24 --jobs 8
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

# Belt and braces: the launcher already sets this, but if someone runs the
# script by hand we still refuse to touch a GPU that training is using.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Module-level config for the pool workers. Set once in main() BEFORE the pool
# is created, so forked workers inherit it and the per-task payload stays a
# single filename string.
_CFG = {"n_classes": 18, "spacing": (1.0, 1.0, 1.0), "tau": 1.0}


def _worker(path):
    """One subject: load the spooled label pair, return per-class metrics."""
    from surface_metrics import all_class_metrics
    try:
        with np.load(path) as z:
            gt = z["gt"]
            pred = z["pred"]
            sid = int(z["sid"]) if "sid" in z else -1
            aux_missing = z["aux_missing"] if "aux_missing" in z else None
            ref_missing = z["ref_missing"] if "ref_missing" in z else None
        res = all_class_metrics(gt.astype(np.int16), pred.astype(np.int16),
                                _CFG["n_classes"], spacing=_CFG["spacing"],
                                tau=_CFG["tau"])
        voxel_ml = float(np.prod(_CFG["spacing"])) / 1000.0
        volume = None
        if aux_missing is not None:
            phantom_ml = float((aux_missing > 0).sum()) * voxel_ml
            volume = {"phantom_ml": phantom_ml}
            if ref_missing is not None:
                ref = ref_missing > 0
                pred_missing = aux_missing > 0
                ref_ml = float(ref.sum()) * voxel_ml
                pred_ml = float(pred_missing.sum()) * voxel_ml
                volume.update({
                    "hallucinated_ml": float(((pred > 0) & ref).sum()) * voxel_ml,
                    "reference_missing_ml": ref_ml,
                    "predicted_missing_ml": pred_ml,
                    "volume_error": (abs(pred_ml - ref_ml) / ref_ml
                                     if ref_ml > 0 else float("nan")),
                })
        return {"path": path, "sid": sid, "per_class": res,
                "volume": volume, "error": None}
    except Exception as exc:                       # never kill the whole round
        return {"path": path, "sid": -1, "per_class": {}, "error": repr(exc)}


def _nanmean(xs):
    xs = [v for v in xs if v is not None and np.isfinite(v)]
    return float(np.mean(xs)) if xs else float("nan")


def wait_for_spool(spool, expect_ranks, expect_subjects, timeout):
    """Block until every rank has signalled done, or we run out of patience.

    This is the ONLY synchronization between the ranks and the reducer, and it
    is a filesystem poll, not a collective -- a rank that died simply makes us
    time out and reduce whatever landed, instead of hanging training.
    """
    t0 = time.time()
    while True:
        done = glob.glob(os.path.join(spool, "rank_*.done"))
        files = sorted(glob.glob(os.path.join(spool, "subj_*.npz")))
        if len(done) >= expect_ranks and len(files) >= expect_subjects:
            return files, True
        if time.time() - t0 > timeout:
            print(f"[reduce] TIMEOUT after {timeout}s: {len(done)}/{expect_ranks} ranks "
                  f"reported, {len(files)}/{expect_subjects} subjects present; "
                  f"reducing what is here", flush=True)
            return files, False
        time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--spacing", default="1,1,1")
    ap.add_argument("--fail-dice", type=float, default=0.5)
    ap.add_argument("--exclude-classes", default="")
    ap.add_argument("--expect-ranks", type=int, default=1)
    ap.add_argument("--expect-subjects", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep-spool", action="store_true")
    ap.add_argument("--marker", default="")
    args = ap.parse_args()

    # Yield to the training process's dataloader workers whenever the box is
    # oversubscribed: this job has no deadline, the training loop does.
    try:
        os.nice(max(0, args.nice))
    except Exception:
        pass

    n_classes = int(args.n_classes)
    spacing = tuple(float(x) for x in args.spacing.split(","))
    exclude = set(int(x) for x in args.exclude_classes.split(",") if x.strip())
    _CFG.update({"n_classes": n_classes, "spacing": spacing, "tau": float(args.tau)})
    os.makedirs(args.out, exist_ok=True)

    try:
        _run(args, n_classes, spacing, exclude)
    finally:
        # Always release the round, even on a crash -- otherwise the callback
        # sees a stale REDUCING marker and silently skips evals forever.
        if args.marker and os.path.exists(args.marker):
            try:
                os.remove(args.marker)
            except OSError:
                pass


def _run(args, n_classes, spacing, exclude):
    t0 = time.time()
    files, complete = wait_for_spool(args.spool, args.expect_ranks,
                                     args.expect_subjects, args.timeout)
    t_wait = time.time() - t0
    if not files:
        print("[reduce] nothing to reduce", flush=True)
        return
    print(f"[reduce] {len(files)} subjects, jobs={args.jobs}, waited {t_wait:.1f}s "
          f"for ranks (complete={complete})", flush=True)

    acc = {c: {"dice": [], "nsd": [], "hd95": [], "assd": [], "miss": 0,
               "present": 0, "gtvox": 0.0} for c in range(1, n_classes)}
    subj_fg_dice, n_fail, n_err, volume_rows = [], 0, 0, []

    t1 = time.time()
    import multiprocessing as mp
    jobs = max(1, min(int(args.jobs), len(files)))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs) as pool:
        for k, out in enumerate(pool.imap_unordered(_worker, files, chunksize=1)):
            if out["error"]:
                n_err += 1
                print(f"  [{k+1}/{len(files)}] FAILED {os.path.basename(out['path'])}: "
                      f"{out['error']}", flush=True)
                continue
            fg = []
            for c, r in out["per_class"].items():
                if r["status"] in ("ok", "empty_pred"):        # GT present
                    a = acc[c]
                    a["present"] += 1
                    a["dice"].append(r["dice"])
                    a["nsd"].append(r["nsd"])
                    a["assd"].append(r["assd"])
                    if r["status"] == "empty_pred":
                        a["miss"] += 1
                    else:
                        a["hd95"].append(r["hd95"])
                    a["gtvox"] += r["gt_vox"]
                    if c not in exclude:
                        fg.append(r["dice"])
            mean_fg = float(np.mean(fg)) if fg else float("nan")
            subj_fg_dice.append(mean_fg)
            if out.get("volume") is not None:
                volume_rows.append({"sid": out["sid"], **out["volume"]})
            if np.isfinite(mean_fg) and mean_fg < args.fail_dice:
                n_fail += 1
            print(f"  [{k+1}/{len(files)}] sid={out['sid']} mean_fg_dice={mean_fg:.4f}",
                  flush=True)
            if not args.keep_spool:
                try:
                    os.remove(out["path"])          # free the disk as we go
                except OSError:
                    pass
    t_metrics = time.time() - t1

    rows = []
    for c in range(1, n_classes):
        a = acc[c]
        rows.append({
            "class": c,
            "n_present": a["present"],
            "n_miss": a["miss"],
            "dice_mean": _nanmean(a["dice"]),
            "dice_min": (float(np.min(a["dice"])) if a["dice"] else float("nan")),
            "nsd_mean": _nanmean(a["nsd"]),
            "nsd_min": (float(np.min(a["nsd"])) if a["nsd"] else float("nan")),
            "hd95_mean": _nanmean(a["hd95"]),
            "hd95_max": (float(np.max(a["hd95"])) if a["hd95"] else float("nan")),
            "assd_mean": _nanmean(a["assd"]),
            "gtvox": a["gtvox"],
        })

    agg = [r for r in rows if r["class"] not in exclude]
    macro_dice = _nanmean([r["dice_mean"] for r in agg])
    macro_nsd = _nanmean([r["nsd_mean"] for r in agg])
    macro_hd95 = _nanmean([r["hd95_mean"] for r in agg])
    _w = np.array([r["gtvox"] for r in agg])
    _d = np.array([r["dice_mean"] for r in agg])
    _ok = np.isfinite(_d) & (_w > 0)
    volw_dice = float((_w[_ok] * _d[_ok]).sum() / _w[_ok].sum()) if _ok.any() else float("nan")
    worst = float(np.nanmin(subj_fg_dice)) if subj_fg_dice else float("nan")
    fail_rate = n_fail / max(1, len(subj_fg_dice))

    from surface_metrics import label_name
    print("\n" + "=" * 78)
    print(f"  REAL-DATA EVAL (spool reduce)  n={len(subj_fg_dice)}  tau={args.tau}mm")
    print("=" * 78)
    print(f"  {'cls':>3} {'structure':>12} {'n':>4} {'miss':>4} {'Dice':>7} {'Dmin':>7} "
          f"{'NSD':>7} {'NSDmin':>7} {'HD95':>7} {'HD95mx':>7} {'ASSD':>7}")
    for r in rows:
        print(f"  {r['class']:>3} {label_name(r['class']):>12} {r['n_present']:>4} "
              f"{r['n_miss']:>4} {r['dice_mean']:>7.3f} {r['dice_min']:>7.3f} "
              f"{r['nsd_mean']:>7.3f} {r['nsd_min']:>7.3f} "
              f"{r['hd95_mean']:>7.2f} {r['hd95_max']:>7.2f} {r['assd_mean']:>7.2f}")
    print("-" * 78)
    print(f"  MACRO  Dice={macro_dice:.4f}  VolWDice={volw_dice:.4f}  "
          f"NSD={macro_nsd:.4f}  HD95={macro_hd95:.2f}mm")
    print(f"  TAIL   worst-subject mean-fg-Dice={worst:.4f}  "
          f"failure-rate(<{args.fail_dice})={fail_rate:.3f} "
          f"({n_fail}/{len(subj_fg_dice)})")
    if volume_rows:
        ph = _nanmean([r["phantom_ml"] for r in volume_rows])
        hall = _nanmean([r.get("hallucinated_ml") for r in volume_rows])
        verr = _nanmean([r.get("volume_error") for r in volume_rows])
        print(f"  VOLUME phantom_mL={ph:.3f}  hallucinated_mL={hall:.3f}  "
              f"relative-volume-error={verr:.3f}")
    print(f"  TIME   rank-wait {t_wait:.1f}s + metrics {t_metrics:.1f}s "
          f"on {min(args.jobs, len(files))} cores")
    print("=" * 78, flush=True)

    with open(os.path.join(args.out, "per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "macro_dice": macro_dice, "volw_dice": volw_dice, "macro_nsd": macro_nsd,
        "macro_hd95": macro_hd95, "worst_subject_fg_dice": worst,
        "failure_rate": fail_rate, "n_subjects": len(subj_fg_dice),
        "n_errors": n_err, "complete": bool(complete), "tau_mm": args.tau,
        "wait_seconds": t_wait, "metrics_seconds": t_metrics,
    }
    if volume_rows:
        summary.update({
            # On a healthy cohort this is the false-positive cost of the aux
            # head. On a cavity cohort use reference_missing_id as well; then
            # the other two requested headline metrics are available.
            "phantom_ml_mean": _nanmean([r["phantom_ml"] for r in volume_rows]),
            "phantom_ml_max": float(np.nanmax([r["phantom_ml"] for r in volume_rows])),
            "hallucinated_ml_mean": _nanmean([
                r.get("hallucinated_ml") for r in volume_rows]),
            "volume_error_mean": _nanmean([
                r.get("volume_error") for r in volume_rows]),
        })
        with open(os.path.join(args.out, "per_subject_volume.csv"), "w", newline="") as f:
            fields = sorted({k for row in volume_rows for k in row})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(volume_rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Durable, wandb-independent trajectory of the whole run.
    hist = os.path.join(os.path.dirname(os.path.abspath(args.out)), "history.csv")
    try:
        new = not os.path.exists(hist)
        with open(hist, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch_dir"] + list(summary.keys()))
            w.writerow([os.path.basename(os.path.abspath(args.out))]
                       + list(summary.values()))
    except Exception as exc:
        print(f"[reduce] could not append history.csv: {exc}", flush=True)

    if not args.keep_spool:
        try:
            for p in glob.glob(os.path.join(args.spool, "*")):
                os.remove(p)
            os.rmdir(args.spool)
        except OSError:
            pass
    print(f"[reduce] wrote {args.out}/per_class.csv and summary.json", flush=True)


if __name__ == "__main__":
    main()
