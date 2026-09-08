"""Run the SIAM + FreeSurfer-fusion pipeline on N random Mongo volumes.

Sibling of ``predict_samples.py``.  Instead of running a MeshNet checkpoint on
each extracted volume, this script, for every sampled volume:

  1. pulls the T1 (``datafield``) and the dense atlas label (``labelfield``,
     e.g. ``label104`` / ``label108``) straight out of Mongo — the same loader
     ``predict_samples.py`` uses already returns both;
  2. writes the T1 to ``<outdir>/vol<ID>/T1.nii.gz``;
  3. converts the dense label to the reduced 0..17 FreeSurfer-style scheme that
     ``fuse_fs_siam.py`` expects and writes it to ``.../label18.nii.gz``;
  4. runs ``siam-pred -i T1.nii.gz -o siam`` (writes
     ``siamV03_siamT1.nii.gz`` beside the input; located by globbing
     ``siamV03_*`` in the per-volume folder);
  5. runs ``fuse_fs_siam.py`` on the SIAM prediction + the 18-label volume and
     writes ``.../fused.nii.gz``.

All four NIfTIs (T1, siam, label18, fused) are kept per volume for inspection.

Defaults sample 10 random volumes so you can eyeball a handful before committing
to a full-collection conversion.

Usage
-----
    python predict_samples_siam.py                       # 10 random MRN vols
    python predict_samples_siam.py --n 10 --seed 42
    python predict_samples_siam.py --ids 1616,1682,2583
    python predict_samples_siam.py --label-map label104_to_18.json
    python predict_samples_siam.py --skip-siam --skip-fuse   # extract + convert only

The 108/104 -> 18 mapping lives in ``LABEL_TO_18`` below (or pass --label-map
pointing at a JSON dict / two-column TSV/CSV).
"""

import argparse
import glob
import gzip
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from catalyst.data import BatchPrefetchLoaderWrapper

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Reuse the NIfTI writer, affine parsing and label-intent constants from the
# sibling script so both write byte-identical headers.
import predict_samples as ps
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler
from mindfultensors.mongoloader import MongoDataset, MongoClient
import curriculum_training as base   # applies MongoDataset retry monkey-patch


# ---------------------------------------------------------------------------
# dense-atlas (label104) -> 18-class (0..17 FreeSurfer-style) mapping
# ---------------------------------------------------------------------------
# SynthSeg-style 18-class target with L/R homologs merged; matches the
# preprocessing `merge_homologs`.  This is the built-in default; pass
# --label-map to override with a JSON dict or two-column TSV/CSV instead.

def lut104_to_18() -> np.ndarray:
    """104-entry LUT, lut[src label] -> 0..17 class. Unlisted sources stay 0."""
    m = np.zeros(104, dtype=np.uint8)
    m[1:69] = 2                    # cortex L+R          -> Cerebral-Cortex
    m[69:71] = 7                   # thalamus
    m[71:73] = 8                   # caudate
    m[73:75] = 9                   # putamen
    m[75:77] = 10                  # pallidum
    m[77:79] = 14                  # hippocampus
    m[79:81] = 15                  # amygdala
    m[81:83] = 16                  # accumbens
    m[83:85] = 17                  # ventralDC
    m[85:87] = 1                   # cerebral white matter
    m[87] = 3; m[88] = 4; m[89] = 3; m[90] = 4   # lat / inf-lat ventricle (L,R)
    m[91] = 11                     # 3rd ventricle
    m[92] = 12                     # 4th ventricle
    m[93] = 0                      # CSF                 -> background
    m[94] = 13                     # brain-stem
    m[95:97] = 5                   # cerebellum white matter
    m[97:99] = 6                   # cerebellum cortex
    m[99:104] = 1                  # corpus callosum     -> white matter
    return m


def _read_map_file(path: str) -> dict[int, int]:
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}
    mapping: dict[int, int] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            mapping[int(float(parts[0]))] = int(float(parts[1]))
    if not mapping:
        raise SystemExit(f"--label-map {path} parsed to an empty mapping")
    return mapping


def build_lut(mapping: dict[int, int], max_label: int) -> np.ndarray:
    """Vectorized lookup table: lut[src] = dst, unlisted -> 0."""
    size = max(max(mapping) + 1, int(max_label) + 1)
    lut = np.zeros(size, dtype=np.uint8)
    for s, d in mapping.items():
        lut[int(s)] = int(d)
    return lut


def convert_to_18(label: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply the LUT; source indices beyond the LUT collapse to 0."""
    label = label.astype(np.int64, copy=False)
    safe = np.where(label < len(lut), label, 0)
    return lut[safe].astype(np.uint8)


def apply_transpose(vol: np.ndarray, perm) -> np.ndarray:
    """Reorder the mongo voxel axes so the on-disk NIfTI matches REFERENCE_AFFINE
    (RSP).  The browser applies ``tensor.transpose()`` (TF.js default perm for a
    3-D volume is [2,1,0], reversing all axes) to recover the raw mongo array for
    the model, so we pre-apply that same permutation here.  ascontiguousarray
    materializes the reorder so the writer's C-order tobytes() is correct.
    """
    if perm is None:
        return vol
    return np.ascontiguousarray(np.transpose(vol, perm))


def parse_perm(arg: str):
    """Parse --transpose: 'none'/'identity' -> None, else comma/space ints."""
    if arg is None:
        return (2, 1, 0)
    a = arg.strip().lower()
    if a in ("none", "identity", "off", ""):
        return None
    perm = tuple(int(v) for v in a.replace(",", " ").split())
    if sorted(perm) != [0, 1, 2]:
        raise SystemExit(f"--transpose must be a permutation of 0,1,2; got {perm}")
    return perm


# ---------------------------------------------------------------------------
# external tools
# ---------------------------------------------------------------------------

def run_siam(t1_path: str, sampledir: str, cmd: str) -> str:
    """Run ``siam-pred -i T1.nii.gz -o siam`` and return the produced volume.

    siam-pred writes ``siamV03_<-o value><input basename>`` into the *input's*
    directory (e.g. -i .../T1.nii.gz -o siam -> .../siamV03_siamT1.nii.gz).
    Each volume has its own folder, so we locate the result by globbing
    ``siamV03_*`` there rather than reconstructing the exact concatenation.
    """
    pattern = os.path.join(sampledir, "siamV03_*.nii*")
    before = set(glob.glob(pattern))
    subprocess.run([cmd, "-i", t1_path, "-o", "siam"], check=True)
    after = set(glob.glob(pattern))
    new = sorted(after - before) or sorted(after)
    if not new:
        raise FileNotFoundError(
            f"siam-pred produced no siamV03_* file in {sampledir} "
            f"(ran: {cmd} -i {t1_path} -o siam)"
        )
    if len(new) > 1:
        print(f"[siam] WARNING: multiple siamV03_* matches, using newest: {new[-1]}",
              flush=True)
    return new[-1]


def set_label_intent(path: str) -> None:
    """Stamp a NIfTI-1 file in place as a label volume (intent_code =
    NIFTI_INTENT_LABEL). siam-pred writes its segmentation without setting the
    intent, which some downstream tools rely on. Header-only patch: the image
    data and any header-extension bytes are preserved byte-for-byte.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    if len(raw) < 348:
        print(f"[siam] WARNING: {path} too small for a NIfTI-1 header; "
              "skipping intent patch", flush=True)
        return
    hdr = np.frombuffer(raw[:348], dtype=ps._NII_DTYPE).copy()
    if int(hdr["sizeof_hdr"][0]) != 348:
        print(f"[siam] WARNING: {path} sizeof_hdr={int(hdr['sizeof_hdr'][0])} "
              "(byte-swapped or not NIfTI-1?); skipping intent patch", flush=True)
        return
    if int(hdr["intent_code"][0]) == ps.NIFTI_INTENT_LABEL:
        return
    hdr["intent_code"] = ps.NIFTI_INTENT_LABEL
    payload = hdr.tobytes() + raw[348:]
    if path.endswith(".gz"):
        with gzip.open(path, "wb", compresslevel=9) as f:
            f.write(payload)
    else:
        with open(path, "wb") as f:
            f.write(payload)


def run_fuse(siam_path: str, fs_path: str, out_path: str,
             fuse_script: str, cortical_interface: str) -> None:
    subprocess.run(
        [sys.executable, fuse_script,
         "--siam", siam_path,
         "--freesurfer", fs_path,
         "--output", out_path,
         "--cortical-interface", cortical_interface],
        check=True,
    )


# ---------------------------------------------------------------------------
# loader (mirrors predict_samples.py)
# ---------------------------------------------------------------------------

def build_loader(cfg, db, collection, datafield, labelfield, db_host,
                 sampled_ids, n):
    index_id = cfg.mongo.index_id
    db_fields = (datafield, labelfield)
    print(f"[siam] fields: data={datafield} label={labelfield}", flush=True)

    client_creator = base.ClientCreator(db_host)
    client_creator.set_database(db)
    client_creator.set_collection(collection)
    client_creator.set_shape([256, 256, 256])
    client_creator.set_num_subcubes(1)

    dataset = MongoDataset(
        sampled_ids,
        client_creator.mytransform,
        None,
        db_fields,
        normalize=unit_interval_normalize,
        id=index_id,
    )
    dl_cfg = cfg.get("dataloader", {}) or {}
    num_workers = min(n, int(dl_cfg.get("num_workers", 12)))
    prefetch_factor = int(dl_cfg.get("prefetch_factor", 4))

    sampler = DBBatchSampler(dataset, batch_size=1, seed=42)
    loader = BatchPrefetchLoaderWrapper(
        DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=client_creator.mycollate_full,
            pin_memory=False,
            worker_init_fn=client_creator.create_client,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
        ),
        num_prefetches=n,
    )
    return loader


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract N random Mongo volumes and run the SIAM+FS fusion pipeline."
    )
    parser.add_argument("--config",     default="gn_hdc_deep_fast",
                        help="Config stem inside conf/ (no .yaml); supplies "
                             "mongo host / index_id. Default: gn_hdc_deep_fast")
    parser.add_argument("--n",          type=int, default=10,
                        help="Number of random volumes (default: 10)")
    parser.add_argument("--ids",        default=None,
                        help="Comma/space-separated volume IDs. Overrides --n/--seed.")
    parser.add_argument("--db",         default="MindfulTensors")
    parser.add_argument("--collection", default="MRN")
    parser.add_argument("--labelfield", default="label104",
                        help="Dense atlas label field to convert to 18 classes "
                             "(default: label104)")
    parser.add_argument("--datafield",  default="T1")
    parser.add_argument("--outdir",     default="siam_fuse_samples")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--affine",     default=None,
                        help="Output affine: .nii[.gz] path or 12 floats. "
                             "Default: predict_samples.REFERENCE_AFFINE.")
    parser.add_argument("--label-map",  default=None,
                        help="JSON dict or two-column TSV/CSV of src->18 mapping.")
    parser.add_argument("--transpose",  default="2,1,0",
                        help="Voxel-axis permutation applied before saving so the "
                             "NIfTI matches the RSP affine (browser tensor."
                             "transpose() undoes it). Default '2,1,0'; pass 'none' "
                             "to save the raw mongo axis order.")
    parser.add_argument("--siam-cmd",   default="siam-pred",
                        help="siam-pred executable (default: siam-pred on PATH)")
    parser.add_argument("--fuse-script", default=None,
                        help="Path to fuse_fs_siam.py (default: beside this script)")
    parser.add_argument("--cortical-interface", default="siam-envelope",
                        choices=("auto", "grow-csf", "siam-envelope", "freesurfer"),
                        help="Passed to fuse_fs_siam.py (default: siam-envelope)")
    parser.add_argument("--skip-siam",  action="store_true",
                        help="Skip siam-pred (and fuse). Only extract T1 + label18.")
    parser.add_argument("--skip-fuse",  action="store_true",
                        help="Run siam-pred but skip the fusion step.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue to the next volume if siam/fuse fails.")
    args = parser.parse_args()

    affine = ps.parse_affine(args.affine) if args.affine else None
    perm = parse_perm(args.transpose)
    print(f"[siam] voxel transpose: {perm if perm else 'none (raw mongo order)'}",
          flush=True)

    fuse_script = args.fuse_script or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fuse_fs_siam.py")

    # --- label mapping (fail fast before touching Mongo) ---
    map_dict = _read_map_file(args.label_map) if args.label_map else None
    if map_dict is not None:
        print(f"[siam] label map from {args.label_map}: {len(map_dict)} source "
              f"labels -> {sorted(set(map_dict.values()))}", flush=True)
    else:
        print("[siam] label map: built-in lut104_to_18 "
              "(SynthSeg-style, homologs merged)", flush=True)

    # --- config ---
    cfg_path = os.path.join("conf", args.config + ".yaml")
    if not os.path.isfile(cfg_path):
        for parent in ["..", "../.."]:
            alt = os.path.join(parent, cfg_path)
            if os.path.isfile(alt):
                cfg_path = alt
                break
    cfg = OmegaConf.load(cfg_path)

    db_host = (cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID")
               else cfg.mongo.host)

    # --- discover collection size ---
    _mc = MongoClient("mongodb://" + db_host + ":27017")
    index_id = cfg.mongo.index_id
    num_examples = int(
        _mc[args.db][args.collection + ".bin"]
        .find_one(sort=[(index_id, -1)])[index_id] + 1
    )
    _mc.close()
    print(f"[siam] {args.db}/{args.collection}: {num_examples} volumes", flush=True)

    # --- sample ---
    if args.ids:
        sampled_ids = [int(v) for v in args.ids.replace(",", " ").split()]
        bad = [i for i in sampled_ids if not (0 <= i < num_examples)]
        if bad:
            raise ValueError(f"--ids out of range [0,{num_examples}): {bad}")
        n = len(sampled_ids)
        print(f"[siam] explicit IDs ({n}): {sampled_ids}", flush=True)
    else:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.n, num_examples)
        sampled_ids = sorted(random.sample(range(num_examples), n))
        print(f"[siam] sampled IDs: {sampled_ids}", flush=True)

    loader = build_loader(cfg, args.db, args.collection, args.datafield,
                          args.labelfield, db_host, sampled_ids, n)

    os.makedirs(args.outdir, exist_ok=True)

    lut = None  # built lazily once we know the label range
    ok, failed = 0, []
    t_prev = time.time()
    for i, batch in enumerate(loader):
        t_load = time.time() - t_prev
        vol_id = sampled_ids[i]
        sampledir = os.path.join(args.outdir, f"vol{vol_id:05d}")
        os.makedirs(sampledir, exist_ok=True)

        sample, label = batch
        # T1: unit-interval normalized -> 0..255 uint8 (matches predict_samples)
        t1 = sample.squeeze().float().cpu().numpy()
        t1 = (t1 * 255.0).round().clip(0, 255).astype(np.uint8)
        # dense atlas label volume straight from Mongo
        lab = label.squeeze().cpu().numpy()
        # reorder mongo voxel axes to the RSP on-disk convention (browser
        # tensor.transpose() reverses this to recover the model's input array)
        t1 = apply_transpose(t1, perm)
        lab = apply_transpose(lab, perm)
        if lut is None:
            lut = (build_lut(map_dict, int(lab.max())) if map_dict is not None
                   else lut104_to_18())
        lab18 = convert_to_18(lab, lut)

        t1_path    = os.path.join(sampledir, "T1.nii.gz")
        label_path = os.path.join(sampledir, "label18.nii.gz")
        ps.save_nifti(t1, t1_path, affine)
        ps.save_nifti(lab18, label_path, affine, ps.NIFTI_INTENT_LABEL)

        siam_path = fused_path = None
        try:
            if not args.skip_siam:
                siam_path = run_siam(t1_path, sampledir, args.siam_cmd)
                set_label_intent(siam_path)   # siam-pred omits the label intent
                if not args.skip_fuse:
                    fused_path = os.path.join(sampledir, "fused.nii.gz")
                    run_fuse(siam_path, label_path, fused_path,
                             fuse_script, args.cortical_interface)
            ok += 1
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            failed.append((vol_id, str(e)))
            print(f"[siam] vol {vol_id}: FAILED — {e}", flush=True)
            if not args.keep_going:
                raise

        outs = "  ".join(p for p in (t1_path, label_path, siam_path, fused_path) if p)
        print(f"  [{i+1}/{n}] vol {vol_id:5d}  load={t_load:6.2f}s  "
              f"labels18={sorted(np.unique(lab18).tolist())} -> {outs}", flush=True)

        del sample, label
        t_prev = time.time()

    print(f"\n[siam] done — {ok}/{n} volumes fully processed, "
          f"outputs in {os.path.abspath(args.outdir)}/")
    if failed:
        print(f"[siam] {len(failed)} failed: {[v for v, _ in failed]}")


if __name__ == "__main__":
    main()
