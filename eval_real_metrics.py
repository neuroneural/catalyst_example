"""Real-data boundary-metric evaluation for the siam18 A/B.

Runs a checkpoint over N real subjects from MindfulTensors / MRN (T1 + a fused
label field already in the model's class space) and reports, PER STRUCTURE:
  Dice, Surface Dice (NSD@tau, mm), HD95 (mm), ASSD (mm),
aggregated as MEAN and WORST-CASE across subjects, plus a per-subject tail
(min foreground Dice, failure rate). Volumetric Dice alone hides the boundary
"floppiness" and leakage this project cares about; NSD/HD95 expose it, and the
tail exposes fragility (the real goal) that any mean metric hides.

Mirrors validate_checkpoint.py's model loading and Mongo plumbing. Metrics live
in surface_metrics.py (torch-free, unit-tested).

Usage (per checkpoint; run once for boundary, once for ctrl):
    python eval_real_metrics.py --config-name gn_hdc_deep_fast_turbo_siam18_boundary --best
    python eval_real_metrics.py --config-name gn_hdc_deep_fast_turbo_siam18_ctrl     --best

Flags (stripped before Hydra):
    --best                 use <logdir>/model.best.pth instead of paths.model
    --max-subjects N       cap subjects (default 300)
    --collection NAME      Mongo collection (default MRN)
    --labelfield NAME      label field (default labelfused; clamped to 0..C-1)
    --datafield NAME       image field (default T1)
    --tau MM               NSD tolerance in mm (default 1.0)
    --spacing SZ,SY,SX     voxel spacing mm (default 1,1,1 -- conformed iso)
    --fail-dice X          per-subject failure threshold on mean fg Dice (0.5)
    --out PATH             results dir (default <logdir>/real_eval)
"""
import os
import sys
import csv
import json

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def _pop_flag(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


def _pop_opt(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return cast(val)
    return default


USE_BEST = _pop_flag("--best")
CKPT = _pop_opt("--ckpt", None, str)          # explicit checkpoint path (overrides all)
MAX_SUBJECTS = _pop_opt("--max-subjects", 300, int)
COLLECTION = _pop_opt("--collection", "MRN", str)
LABELFIELD = _pop_opt("--labelfield", "labelfused", str)
DATAFIELD = _pop_opt("--datafield", "T1", str)
TAU = _pop_opt("--tau", 1.0, float)
SPACING = tuple(float(x) for x in _pop_opt("--spacing", "1,1,1", str).split(","))
FAIL_DICE = _pop_opt("--fail-dice", 0.5, float)
OUT_DIR = _pop_opt("--out", None, str)

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mindfultensors.utils import unit_interval_normalize
from mindfultensors.mongoloader import MongoDataset, MongoClient
from mindfultensors.utils import DBBatchSampler

import curriculum_training as base
import validate_checkpoint as vc          # reuse load_model + checkpoint plumbing
vc._USE_BEST = USE_BEST                    # module-level flag load_model reads
from surface_metrics import all_class_metrics


def build_loader(cfg, db_host, n_ids):
    dl_cfg = cfg.get("dataloader", {}) or {}
    num_workers = int(dl_cfg.get("num_workers", 4))
    prefetch = int(dl_cfg.get("prefetch_factor", 4))
    db_name = "MindfulTensors"
    index_id = cfg.mongo.index_id
    db_fields = (DATAFIELD, LABELFIELD)

    mc = MongoClient("mongodb://" + db_host + ":27017")
    try:
        ids = sorted(int(x) for x in mc[db_name][f"{COLLECTION}.meta"].distinct(index_id))
    except Exception:
        ids = []
    if not ids:
        ids = sorted(int(x) for x in mc[db_name][f"{COLLECTION}.bin"].distinct(index_id))
    mc.close()
    if not ids:
        raise SystemExit(f"[eval] no ids in {db_name}/{COLLECTION}")
    ids = ids[:n_ids]
    print(f"[eval] {db_name}/{COLLECTION}: using {len(ids)} subjects "
          f"(ids {ids[0]}..{ids[-1]})  fields={db_fields}", flush=True)

    cc = base.ClientCreator(db_host)
    cc.set_database(db_name)
    cc.set_collection(COLLECTION)
    cc.set_shape([256, 256, 256])
    cc.set_num_subcubes(1)

    dataset = MongoDataset(ids, cc.mytransform, None, db_fields,
                           normalize=unit_interval_normalize, id=index_id)
    sampler = DBBatchSampler(dataset, batch_size=1, seed=42)
    loader = DataLoader(dataset, sampler=sampler, collate_fn=cc.mycollate_full,
                        pin_memory=False, worker_init_fn=cc.create_client,
                        persistent_workers=True, prefetch_factor=prefetch,
                        num_workers=num_workers)
    return loader, len(ids)


def _nanmean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


@hydra.main(config_path="conf", config_name="gn_hdc_deep_fast_turbo_siam18_boundary",
            version_base=None)
def main(cfg: DictConfig):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("[eval] WARNING: CUDA unavailable — CPU (slow)!", flush=True)

    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host
    n_classes = int(cfg.model.n_classes)
    logdir = cfg.paths.logdir
    out_dir = OUT_DIR or os.path.join(logdir, "real_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the checkpoint UNAMBIGUOUSLY. --ckpt PATH wins; else --best uses
    # <logdir>/model.best.pth; else cfg.paths.model. We overwrite cfg.paths.model
    # so load_model loads exactly this, and we report the resolved path (the old
    # header printed cfg.paths.model regardless of --best -> misleading).
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    if CKPT:
        resolved_ckpt = CKPT
        vc._USE_BEST = False
    elif USE_BEST:
        resolved_ckpt = os.path.join(logdir, "model.best.pth")
        vc._USE_BEST = False
    else:
        resolved_ckpt = cfg.paths.model
    cfg.paths.model = resolved_ckpt
    cfg.paths.loadcheckpoint = True
    if not os.path.isfile(resolved_ckpt):
        raise SystemExit(f"[eval] checkpoint not found: {resolved_ckpt}")
    print(f"[eval] evaluating checkpoint: {resolved_ckpt}", flush=True)

    model = vc.load_model(cfg, device)

    channels_last = (cfg.get("perf", {}) or {}).get("channels_last", True)
    loader, n_subj = build_loader(cfg, db_host, MAX_SUBJECTS)

    tag = f"{cfg.model.model_label}_{COLLECTION}_{os.path.basename(resolved_ckpt)}"
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.team,
                     name=f"realeval {tag}", job_type="real_eval",
                     config={"checkpoint": cfg.paths.model, "collection": COLLECTION,
                             "labelfield": LABELFIELD, "tau_mm": TAU,
                             "spacing_mm": SPACING, "n_subjects": n_subj})

    # per-class accumulators over subjects (GT-present subjects only)
    acc = {c: {"dice": [], "nsd": [], "hd95": [], "assd": [], "miss": 0, "present": 0}
           for c in range(1, n_classes)}
    subj_min_fg_dice = []
    n_fail = 0

    for i, batch in enumerate(loader):
        with torch.no_grad():
            sample, label = batch
            sample = sample.to(device)
            label = label.to(device)
            if channels_last:
                sample = sample.contiguous(memory_format=torch.channels_last_3d)
            y_hat = model(sample)
            pred = torch.squeeze(torch.argmax(y_hat, dim=1)).to(torch.int16).cpu().numpy()
            gt = torch.squeeze(label).to(torch.int64)
            gt = torch.where(gt < n_classes, gt, torch.zeros_like(gt))  # first-18 clamp
            gt = gt.to(torch.int16).cpu().numpy()
            del sample, label, y_hat

        per_class = all_class_metrics(gt, pred, n_classes, spacing=SPACING, tau=TAU)
        fg_dice_this = []
        for c, r in per_class.items():
            if r["status"] in ("ok", "empty_pred"):      # GT present
                acc[c]["present"] += 1
                acc[c]["dice"].append(r["dice"])
                acc[c]["nsd"].append(r["nsd"])
                acc[c]["assd"].append(r["assd"])
                if r["status"] == "empty_pred":
                    acc[c]["miss"] += 1
                else:
                    acc[c]["hd95"].append(r["hd95"])
                fg_dice_this.append(r["dice"])
        mean_fg = float(np.mean(fg_dice_this)) if fg_dice_this else float("nan")
        subj_min_fg_dice.append(mean_fg)
        if np.isfinite(mean_fg) and mean_fg < FAIL_DICE:
            n_fail += 1
        print(f"  [{i+1:3d}/{n_subj}]  mean_fg_dice={mean_fg:.4f}", flush=True)

    # ---- aggregate ----
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
        })

    macro_dice = _nanmean([r["dice_mean"] for r in rows])
    macro_nsd = _nanmean([r["nsd_mean"] for r in rows])
    macro_hd95 = _nanmean([r["hd95_mean"] for r in rows])
    worst_subj = float(np.nanmin(subj_min_fg_dice)) if subj_min_fg_dice else float("nan")
    fail_rate = n_fail / max(1, len(subj_min_fg_dice))

    # ---- print table ----
    print("\n" + "=" * 78)
    print(f"  REAL-DATA EVAL  {COLLECTION}  n={n_subj}  tau={TAU}mm  ckpt={cfg.paths.model}")
    print("=" * 78)
    print(f"  {'cls':>3} {'n':>4} {'miss':>4} {'Dice':>7} {'Dmin':>7} "
          f"{'NSD':>7} {'NSDmin':>7} {'HD95':>7} {'HD95mx':>7} {'ASSD':>7}")
    for r in rows:
        print(f"  {r['class']:>3} {r['n_present']:>4} {r['n_miss']:>4} "
              f"{r['dice_mean']:>7.3f} {r['dice_min']:>7.3f} "
              f"{r['nsd_mean']:>7.3f} {r['nsd_min']:>7.3f} "
              f"{r['hd95_mean']:>7.2f} {r['hd95_max']:>7.2f} {r['assd_mean']:>7.2f}")
    print("-" * 78)
    print(f"  MACRO  Dice={macro_dice:.4f}  NSD={macro_nsd:.4f}  HD95={macro_hd95:.2f}mm")
    print(f"  TAIL   worst-subject mean-fg-Dice={worst_subj:.4f}  "
          f"failure-rate(<{FAIL_DICE})={fail_rate:.3f} ({n_fail}/{len(subj_min_fg_dice)})")
    print("=" * 78)

    # ---- persist ----
    with open(os.path.join(out_dir, "per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"macro_dice": macro_dice, "macro_nsd": macro_nsd,
               "macro_hd95": macro_hd95, "worst_subject_fg_dice": worst_subj,
               "failure_rate": fail_rate, "n_subjects": n_subj,
               "tau_mm": TAU, "collection": COLLECTION, "checkpoint": cfg.paths.model}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {out_dir}/per_class.csv and summary.json", flush=True)

    # ---- wandb ----
    table = wandb.Table(columns=list(rows[0].keys()))
    for r in rows:
        table.add_data(*[r[k] for k in rows[0].keys()])
    wandb.log({"real_eval/per_class": table,
               "real_eval/macro_dice": macro_dice,
               "real_eval/macro_nsd": macro_nsd,
               "real_eval/macro_hd95": macro_hd95,
               "real_eval/worst_subject_fg_dice": worst_subj,
               "real_eval/failure_rate": fail_rate})
    run.summary.update(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
