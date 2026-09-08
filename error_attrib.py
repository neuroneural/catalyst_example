"""Split the 104-class real-data Dice deficit into RIBBON error vs LATERAL
(parcel-assignment) error, exactly, from ground truth.

Why: macro Dice over 104 classes is dominated by 68 cortical parcels, and a
cortical parcel is a ~2mm-thick sheet, so its Dice is really a boundary metric.
Two completely different failures land on the same number:

  RIBBON  - the cortex/WM and cortex/CSF surfaces are in the wrong place.
            Image evidence exists. Fixable with shape/topology losses on the
            MARGINALIZED tissue classes (clDice, precision-favouring Tversky).
  LATERAL - the cortex mask is right but a voxel is given a neighbouring
            parcel. No image evidence at all; this is "where am I in the
            brain". A loss cannot fix it; position/context can.

This script reports, per subject and pooled:
  * as-measured per-parcel Dice (what real_eval prints)
  * collapsed group Dice (cortex as ONE class)          -> ribbon quality
  * counterfactual "lateral fixed, ribbon as-is"        -> ribbon-only Dice
  * counterfactual "ribbon fixed, lateral as-is"        -> lateral-only Dice
  * exact voxel accounting of the GT ribbon
  * the confusion structure (ipsilateral neighbour vs contralateral homolog)
  * whether over-called cortex sits inside the brain or outside it (dura)
  * the same per-label vs collapsed-group split for every other group

Two ways to feed it:

  1. FROM SPOOLS (no GPU, no Mongo, numpy+scipy only). real_eval_ddp already
     writes {pred, gt, sid} npz files; set `real_eval.keep_spool: true` in the
     config so a round's spool survives the reducer, then:

         python error_attrib.py --spool <logdir>/real_eval/epoch_00150/spool

  2. FROM A CHECKPOINT (needs torch + Mongo, i.e. the training server). Reuses
     validate_checkpoint.load_model and the same MRN loader real_eval_ddp uses:

         source ~/venv/torch/bin/activate
         python error_attrib.py \
             --config-name gn_hdc_deep_fast_turbo_siam \
             --ckpt ./logs/tmp/infant104_gn_hdc_deep_turbo24/model.best.pth \
             --n 24 --spool /tmp/attrib_spool

     Add --keep-spool to leave the npz files behind so mode 1 can re-analyse
     them without re-running inference.

Writes attrib_summary.json and attrib_groups.csv next to the spool.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


# --------------------------------------------------------------------------
# label scheme: 104 dense atlas labels -> 18 tissue/structure groups.
# Mirrors predict_samples_siam.lut104_to_18() exactly. Kept local so this
# script has no import-time dependency on torch or hydra in spool mode.
# --------------------------------------------------------------------------
def lut104_to_18():
    m = np.zeros(104, dtype=np.uint8)
    m[1:69] = 2                                  # cortex L+R (68 parcels)
    m[69:71] = 7                                 # thalamus
    m[71:73] = 8                                 # caudate
    m[73:75] = 9                                 # putamen
    m[75:77] = 10                                # pallidum
    m[77:79] = 14                                # hippocampus
    m[79:81] = 15                                # amygdala
    m[81:83] = 16                                # accumbens
    m[83:85] = 17                                # ventralDC
    m[85:87] = 1                                 # cerebral white matter
    m[87] = 3; m[88] = 4; m[89] = 3; m[90] = 4   # lateral / inf-lateral ventricle
    m[91] = 11                                   # 3rd ventricle
    m[92] = 12                                    # 4th ventricle
    m[93] = 0                                     # CSF -> background in 18-space
    m[94] = 13                                    # brain stem
    m[95:97] = 5                                  # cerebellum white matter
    m[97:99] = 6                                  # cerebellum cortex
    m[99:104] = 1                                 # corpus callosum -> white matter
    return m


GROUPS = {
    "cortex 1-68":      list(range(1, 69)),
    "cerebral WM 85-86": [85, 86],
    "corpus callosum 99-103": list(range(99, 104)),
    "subcortex 69-84":  list(range(69, 85)),
    "ventricles 87-92": list(range(87, 93)),
    "brainstem 94":     [94],
    "cerebellum 95-98": list(range(95, 99)),
    "CSF 93":           [93],
}
CORTEX = list(range(1, 69))


def dice_mask(a, b):
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return np.nan
    return 2.0 * int(np.count_nonzero(a & b)) / s


# --------------------------------------------------------------------------
# per-subject attribution
# --------------------------------------------------------------------------
def attribute_subject(pred, gt, cortex=CORTEX, brain_dist_mm=1.5):
    from scipy import ndimage as ndi

    P = pred.astype(np.int64, copy=False)
    G = gt.astype(np.int64, copy=False)
    lo, hi = cortex[0], cortex[-1]
    pc = (P >= lo) & (P <= hi)
    gc = (G >= lo) & (G <= hi)
    both = pc & gc

    per_raw = np.array([dice_mask(P == k, G == k) for k in cortex])
    d_raw = np.nanmean(per_raw)
    d_collapsed = dice_mask(pc, gc)

    # counterfactual 1: every lateral decision made correct, ribbon untouched.
    # Whatever Dice is left is ribbon error and nothing else.
    Pl = P.copy()
    Pl[both] = G[both]
    per_rib = np.array([dice_mask(Pl == k, G == k) for k in cortex])
    d_ribbon_only = np.nanmean(per_rib)

    # counterfactual 2: ribbon made correct, the model's own lateral choices
    # kept. GT ribbon voxels the model did not call cortex are filled from the
    # nearest voxel where it DID choose a parcel, so the model still owns every
    # parcel decision; only the mask is repaired.
    Pr = np.zeros_like(P)
    Pr[both] = P[both]
    need = gc & ~both
    if need.any():
        src = Pr > 0
        if src.any():
            _, (iz, iy, ix) = ndi.distance_transform_edt(~src, return_indices=True)
            Pr[need] = Pr[iz[need], iy[need], ix[need]]
    per_lat = np.array([dice_mask(Pr == k, G == k) for k in cortex])
    d_lateral_only = np.nanmean(per_lat)

    # exact voxel accounting, as a fraction of the GT ribbon
    n_gc = int(gc.sum())
    acc = {
        "right_parcel":  int((both & (P == G)).sum()) / n_gc,
        "wrong_parcel":  int((both & (P != G)).sum()) / n_gc,
        "missed_ribbon": int((gc & ~pc).sum()) / n_gc,
        "extra_ribbon":  int((pc & ~gc).sum()) / n_gc,
    }

    # is the over-call inside the brain (sulcal CSF / WM) or outside it (dura)?
    over = pc & ~gc
    n_over = int(over.sum())
    if n_over:
        dbrain = ndi.distance_transform_edt(G == 0)
        outside = int((over & (dbrain > brain_dist_mm)).sum()) / n_over
    else:
        outside = float("nan")

    return dict(parcel_dice=float(d_raw), collapsed_cortex_dice=float(d_collapsed),
                ribbon_only_dice=float(d_ribbon_only),
                lateral_only_dice=float(d_lateral_only),
                over_call_outside_brain=float(outside), n_gt_ribbon=n_gc,
                _per_raw=per_raw, _per_rib=per_rib, _per_lat=per_lat, **acc)


# --------------------------------------------------------------------------
# hemisphere assignment, derived from the data rather than assumed
# --------------------------------------------------------------------------
def hemisphere_sides(centroids, counts, brain_com, cortex=CORTEX):
    """Assign each cortical parcel a hemisphere from the LABEL ORDER, and verify
    that assumption against the data.

    The 104 scheme is blocked left-then-right: 1-34 are ctx-lh-*, 35-68 are
    ctx-rh-* in the same within-hemisphere order, so parcel k and k+34 are
    homologs. That is exact, so it is what gets used.

    The verification: if the assumption holds, the two blocks' parcel centroids
    must be separable along one axis with NO overlap. Returns that axis, the
    separation in mm, and whether it is overlap-free. If it is not, the label
    order is wrong for this label set and every contralateral number below is
    meaningless, so say so loudly rather than quietly reporting a wrong share.
    """
    n_half = len(cortex) // 2
    side = np.zeros(max(cortex) + 1, np.int8)
    for k in cortex:
        side[k] = -1 if (k - cortex[0]) < n_half else 1
    L = [k for k in cortex if side[k] < 0 and counts[k] > 0]
    R = [k for k in cortex if side[k] > 0 and counts[k] > 0]
    axis, gap, clean = 0, 0.0, False
    for a in range(3):
        lv = centroids[L, a]; rv = centroids[R, a]
        d = abs(float(lv.mean()) - float(rv.mean()))
        if d > gap:
            axis, gap = a, d
            clean = bool(lv.max() < rv.min() or rv.max() < lv.min())
    homolog_mm = float(np.mean([abs(centroids[k, axis] - centroids[k + n_half, axis])
                                for k in L if counts[k + n_half] > 0]))
    return side, dict(axis=axis, mean_gap_mm=gap, overlap_free=clean,
                      homolog_gap_mm=homolog_mm, n_left=len(L), n_right=len(R))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spool", required=True,
                    help="directory of {pred,gt,sid} npz files (read, or written "
                         "by --ckpt mode)")
    ap.add_argument("--out", default=None, help="results dir (default: --spool)")
    ap.add_argument("--top-pairs", type=int, default=15)
    # checkpoint mode
    ap.add_argument("--ckpt", default=None,
                    help="run this checkpoint over Mongo subjects first "
                         "(needs torch + Mongo)")
    ap.add_argument("--config-name", default="gn_hdc_deep_fast_turbo_siam")
    ap.add_argument("--n", type=int, default=24, help="subjects, --ckpt mode")
    ap.add_argument("--keep-spool", action="store_true")
    args = ap.parse_args()

    if args.ckpt:
        spool_from_checkpoint(args)

    files = sorted(glob.glob(os.path.join(args.spool, "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        raise SystemExit(f"no npz spool files in {args.spool}")
    out_dir = args.out or args.spool
    os.makedirs(out_dir, exist_ok=True)

    C = np.zeros((104, 104), np.int64)              # C[gt, pred]
    cent = np.zeros((104, 3)); cnt = np.zeros(104)
    brain_com = np.zeros(3); brain_n = 0.0
    rows = []
    print(f"[attrib] {len(files)} subjects from {args.spool}\n", flush=True)
    hdr = ("  sid   parcel  collapsed | ribbonOnly lateralOnly |  right  wrongP  "
           "missRib extraRib  outside")
    print(hdr)
    for f in files:
        z = np.load(f)
        P, G = z["pred"], z["gt"]
        sid = int(z["sid"]) if "sid" in z.files else -1
        r = attribute_subject(P, G)
        r["sid"] = sid
        rows.append(r)
        print("  %4d  %6.3f  %9.3f | %10.3f %11.3f | %6.3f %6.3f %8.3f %8.3f %8.1f%%"
              % (sid, r["parcel_dice"], r["collapsed_cortex_dice"],
                 r["ribbon_only_dice"], r["lateral_only_dice"], r["right_parcel"],
                 r["wrong_parcel"], r["missed_ribbon"], r["extra_ribbon"],
                 100 * r["over_call_outside_brain"]), flush=True)
        Pi = P.astype(np.int64); Gi = G.astype(np.int64)
        m = (Gi > 0) | (Pi > 0)
        np.add.at(C, (Gi[m], Pi[m]), 1)
        fg = np.array(np.nonzero(Gi > 0))
        if fg.shape[1]:
            brain_com += fg.mean(1) * fg.shape[1]; brain_n += fg.shape[1]
        for k in CORTEX:
            idx = np.array(np.nonzero(Gi == k))
            if idx.shape[1]:
                cent[k] += idx.mean(1) * idx.shape[1]; cnt[k] += idx.shape[1]
    cent[cnt > 0] /= cnt[cnt > 0, None]
    brain_com /= max(brain_n, 1.0)

    PR = np.nanmean(np.vstack([r.pop("_per_raw") for r in rows]), axis=0)
    PRIB = np.nanmean(np.vstack([r.pop("_per_rib") for r in rows]), axis=0)
    PLAT = np.nanmean(np.vstack([r.pop("_per_lat") for r in rows]), axis=0)
    mean = {k: float(np.nanmean([r[k] for r in rows]))
            for k in rows[0] if k not in ("sid", "n_gt_ribbon")}
    print("\n=== POOLED over %d subjects ===" % len(rows))
    print("  as-measured cortical parcel Dice          %.4f" % mean["parcel_dice"])
    print("  collapsed cortex-as-one-class Dice        %.4f" % mean["collapsed_cortex_dice"])
    print("  counterfactual, lateral fixed (ribbon only) %.4f  -> ribbon costs %.4f"
          % (mean["ribbon_only_dice"], 1.0 - mean["ribbon_only_dice"]))
    print("  counterfactual, ribbon fixed (lateral only) %.4f  -> lateral costs %.4f"
          % (mean["lateral_only_dice"], 1.0 - mean["lateral_only_dice"]))
    tot = (1 - mean["ribbon_only_dice"]) + (1 - mean["lateral_only_dice"])
    if tot > 0:
        print("  share of the deficit: ribbon %.0f%%   lateral %.0f%%"
              % (100 * (1 - mean["ribbon_only_dice"]) / tot,
                 100 * (1 - mean["lateral_only_dice"]) / tot))
    print("\n  GT cortical ribbon voxels:")
    print("    right parcel                %.1f%%" % (100 * mean["right_parcel"]))
    print("    cortex but WRONG parcel     %.1f%%   <- lateral" % (100 * mean["wrong_parcel"]))
    print("    not called cortex at all    %.1f%%   <- ribbon (miss)" % (100 * mean["missed_ribbon"]))
    print("    over-called elsewhere       %.1f%%   <- ribbon (fat), %.0f%% of it outside the brain"
          % (100 * mean["extra_ribbon"], 100 * mean["over_call_outside_brain"]))

    # ---------------- per-parcel: which parcels lose to which mode ----------
    # cost_ribbon  = 1 - Dice with every lateral decision corrected
    # cost_lateral = 1 - Dice with the ribbon mask corrected
    cr = 1.0 - PRIB
    cl = 1.0 - PLAT
    print("\n=== per-parcel cost, worst 12 by each mode "
          "(parcel: dice, ribbonCost, lateralCost) ===")
    for name, key in (("LATERAL-dominated", cl - cr), ("RIBBON-dominated", cr - cl)):
        order = np.argsort(-key)[:12]
        print("  %s:" % name)
        print("    " + "  ".join("%d:%.2f/%.2f/%.2f"
              % (CORTEX[i], PR[i], cr[i], cl[i]) for i in order))

    # ---------------- confusion structure ----------------
    ctx = np.array(CORTEX)
    sub = C[np.ix_(ctx, ctx)]
    wrong = int(sub.sum() - np.trace(sub))
    side, hemi = hemisphere_sides(cent, cnt, brain_com)
    n_half = len(CORTEX) // 2
    def homolog(k):
        return k + n_half if (k - CORTEX[0]) < n_half else k - n_half
    contra = 0
    homo = 0
    pairs = []
    for i, gi in enumerate(ctx):
        for j, pj in enumerate(ctx):
            if gi == pj:
                continue
            n = int(sub[i, j])
            if side[gi] != side[pj]:
                contra += n
            if pj == homolog(int(gi)):
                homo += n
            pairs.append((n, int(gi), int(pj), side[gi] != side[pj]))
    print("\n=== LATERAL confusion structure ===")
    print("  hemispheres from label order: %d left / %d right; homologs are k <-> k+%d"
          % (hemi["n_left"], hemi["n_right"], n_half))
    print("  check: the two blocks separate along axis %d by %.1f mm "
          "(homolog pairs %.1f mm apart), overlap-free = %s"
          % (hemi["axis"], hemi["mean_gap_mm"], hemi["homolog_gap_mm"],
             hemi["overlap_free"]))
    if not hemi["overlap_free"]:
        print("  !! the assumed label order does NOT split into two hemispheres for "
              "this label set. Ignore every contralateral number below.")
    if wrong:
        cs = contra / wrong
        print("  contralateral (other hemisphere)   %.1f%%" % (100 * cs))
        print("  ipsilateral neighbour              %.1f%%" % (100 * (1 - cs)))
        print("  of which CONTRALATERAL HOMOLOG (k <-> k+%d) %.1f%% of all wrong-parcel"
              % (n_half, 100 * homo / wrong))
        pairs.sort(reverse=True)
        cum = np.cumsum([p[0] for p in pairs]) / wrong
        print("  top 20 pairs = %.0f%% of all wrong-parcel voxels; top 100 = %.0f%%"
              % (100 * cum[min(19, len(cum) - 1)], 100 * cum[min(99, len(cum) - 1)]))
        print("  top %d confusions (gt -> pred):" % args.top_pairs)
        for n, gi, pj, isc in pairs[:args.top_pairs]:
            print("    %3d -> %3d  %8d  %4.1f%%  %s"
                  % (gi, pj, n, 100 * n / wrong,
                     ("HOMOLOG" if pj == homolog(gi) else "CONTRA") if isc else "ipsi"))

    # ---------------- where the ribbon leaks ----------------
    def gsum(rows_, cols_):
        return int(C[np.ix_(np.array(rows_), np.array(cols_))].sum())
    gc_tot = int(C[ctx].sum())
    print("\n=== RIBBON leakage: GT cortical voxels by PREDICTED group ===")
    for name, ids in GROUPS.items():
        if name == "cortex 1-68":
            continue
        print("  %-24s %.1f%%" % (name, 100 * gsum(ctx, ids) / gc_tot))
    print("  %-24s %.1f%%" % ("background 0", 100 * gsum(ctx, [0]) / gc_tot))
    pc_tot = int(C[:, ctx].sum())
    print("\n=== OVER-CALL: predicted-cortex voxels by TRUE group ===")
    for name, ids in GROUPS.items():
        if name == "cortex 1-68":
            continue
        print("  %-24s %.1f%%" % (name, 100 * gsum(ids, ctx) / pc_tot))
    print("  %-24s %.1f%%" % ("background 0", 100 * gsum([0], ctx) / pc_tot))

    # ---------------- per-group per-label vs collapsed ----------------
    print("\n=== every group: per-label Dice vs collapsed-group Dice ===")
    print("  %-24s %10s %10s %14s" % ("group", "per-label", "collapsed", "in-group swaps"))
    grows = []
    for name, ids in GROUPS.items():
        r = np.array(ids)
        per = []
        for k in ids:
            i = int(C[k].sum()); j = int(C[:, k].sum()); tp = int(C[k, k])
            per.append(2.0 * tp / (i + j) if (i + j) else np.nan)
        I = int(C[np.ix_(r, r)].sum()); gi = int(C[r].sum()); pj = int(C[:, r].sum())
        col = 2.0 * I / (gi + pj) if (gi + pj) else np.nan
        swap = (I - int(np.trace(C[np.ix_(r, r)]))) / gi if gi else np.nan
        grows.append(dict(group=name, n_labels=len(ids), per_label_dice=float(np.nanmean(per)),
                          collapsed_dice=float(col), in_group_swap_frac=float(swap)))
        print("  %-24s %10.3f %10.3f %13.1f%%"
              % (name, np.nanmean(per), col, 100 * swap))

    import csv
    with open(os.path.join(out_dir, "attrib_groups.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grows[0]))
        w.writeheader()
        w.writerows(grows)
    with open(os.path.join(out_dir, "attrib_summary.json"), "w") as fh:
        json.dump({"n_subjects": len(rows), "pooled": mean, "per_subject": rows,
                   "hemisphere_check": hemi,
                   "wrong_parcel_voxels": wrong,
                   "contralateral_frac": (contra / wrong) if wrong else None,
                   "contralateral_homolog_frac": (homo / wrong) if wrong else None,
                   "groups": grows}, fh, indent=2)
    print("\n[attrib] wrote %s/attrib_summary.json and attrib_groups.csv" % out_dir)


# --------------------------------------------------------------------------
# --ckpt mode: reuse the repo's own model loading + MRN loader
# --------------------------------------------------------------------------
def spool_from_checkpoint(args):
    import contextlib
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from mindfultensors.mongoloader import MongoDataset, MongoClient
    from mindfultensors.utils import unit_interval_normalize, DBBatchSampler

    import curriculum_training as base
    import validate_checkpoint as vc

    cfg = OmegaConf.load(os.path.join("conf", args.config_name + ".yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg.paths.model = args.ckpt
    cfg.paths.loadcheckpoint = True
    vc._USE_BEST = False
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    device = (torch.device("cuda:0") if torch.cuda.is_available()
              else torch.device("cpu"))
    if device.type == "cpu":
        print("[attrib] WARNING: no CUDA, inference on CPU will be very slow",
              flush=True)
    n_classes = int(cfg.model.n_classes)
    if n_classes > 256:
        raise SystemExit("spool dtype is uint8; widen it for >256 classes")
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    v = cfg.get("validation", {}) or {}
    db = str(v.get("db", "MindfulTensors"))
    col = str(v.get("collection", "MRN"))
    datafield = str(v.get("datafield", "T1"))
    labelfield = str(v.get("labelfield", "labelfused104"))
    print(f"[attrib] {db}/{col} data={datafield} label={labelfield}", flush=True)

    mc = MongoClient("mongodb://" + db_host + ":27017")
    try:
        try:
            ids = sorted(int(x) for x in mc[db][f"{col}.meta"].distinct(cfg.mongo.index_id))
        except Exception:
            ids = []
        if not ids:
            ids = sorted(int(x) for x in mc[db][f"{col}.bin"].distinct(cfg.mongo.index_id))
    finally:
        mc.close()
    ids = ids[: args.n]
    print(f"[attrib] subjects: {ids}", flush=True)

    cc = base.ClientCreator(db_host)
    cc.set_database(db)
    cc.set_collection(col)
    cc.set_shape([256, 256, 256])
    cc.set_num_subcubes(1)
    ds = MongoDataset(list(ids), cc.mytransform, None, (datafield, labelfield),
                      normalize=unit_interval_normalize, id=cfg.mongo.index_id)
    loader = DataLoader(ds, sampler=DBBatchSampler(ds, batch_size=1, seed=42),
                        collate_fn=cc.mycollate_full, pin_memory=False,
                        worker_init_fn=cc.create_client, persistent_workers=False,
                        num_workers=2, prefetch_factor=2)

    model = vc.load_model(cfg, device)
    model.eval()
    channels_last = bool((cfg.get("perf", {}) or {}).get("channels_last", True))
    amp_name = str((cfg.get("perf", {}) or {}).get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_name in ("bf16", "bfloat16") else torch.float16

    os.makedirs(args.spool, exist_ok=True)
    for i, (sample, label) in enumerate(loader):
        sid = int(ids[i]) if i < len(ids) else -i
        with torch.no_grad():
            sample = sample.to(device)
            label = label.to(device)
            if channels_last:
                sample = sample.contiguous(memory_format=torch.channels_last_3d)
            ctxm = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if device.type == "cuda" else contextlib.nullcontext())
            with ctxm:
                logits = model(sample)
            pred = torch.argmax(logits.float(), dim=1).squeeze()
            del logits, sample
            gt = label.squeeze().to(torch.int64)
            gt = torch.where(gt < n_classes, gt, torch.zeros_like(gt))
            del label
            fg = (gt > 0) | (pred > 0)
            nz = torch.nonzero(fg)
            if nz.numel():
                lo = (nz.min(0).values - 4).clamp_min(0)
                hi = (nz.max(0).values + 5)
                sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                pred = pred[sl]; gt = gt[sl]
            pn = pred.to(torch.uint8).cpu().numpy()
            gn = gt.to(torch.uint8).cpu().numpy()
        path = os.path.join(args.spool, f"subj_{i:03d}_{sid}.npz")
        tmp = path + ".tmp.npz"
        with open(tmp, "wb") as fh:
            np.savez(fh, pred=pn, gt=gn, sid=np.int64(sid))
        os.replace(tmp, path)
        print(f"  [{i+1}/{len(ids)}] sid={sid} spooled {pn.shape}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
