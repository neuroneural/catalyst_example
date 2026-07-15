"""Validation-only inference script.

Loads a model checkpoint from the path specified in a config YAML,
runs a single validation pass over db=MindfulTensors collection=HCP,
logs macro_dice to the same wandb project/team as the training run, and
prints a summary.

Usage:
    python validate_checkpoint.py --config-name gn_hdc_deep_fast_100

The script honours all model/perf/dataloader knobs from the config but
skips every training step.  Only the validation loader (range(32)) runs.
"""

import os
import sys

# Same compile-thread safety fix as curriculum_training_fast.py
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Strip --best before Hydra parses sys.argv (Hydra doesn't know this flag).
_USE_BEST = "--best" in sys.argv
if _USE_BEST:
    sys.argv.remove("--best")

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dice import faster_dice, CEDiceLoss, DiceLoss
from meshnet_gn import enMesh_checkpoint as enMesh_checkpoint_gn
from meshnet import enMesh_checkpoint
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler
from mindfultensors.mongoloader import (
    create_client,
    mcollate,
    MongoDataset,
    MongoClient,
    mtransform,
)

import curriculum_training as base  # reuse ClientCreator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_compile_prefix(sd):
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in sd):
        return sd
    return {k.removeprefix(prefix): v for k, v in sd.items()}


def load_model(cfg, device):
    """Build model from config, load checkpoint, return in eval mode."""
    mc = cfg.model
    groupnorm = mc.use_groupnorm
    use_checkpoint = mc.get("use_checkpoint", True)
    affine = mc.get("use_affine", False)
    channels = mc.model_channels
    n_classes = mc.n_classes
    config_file = mc.config_file

    if groupnorm:
        model = enMesh_checkpoint_gn(
            in_channels=1,
            n_classes=n_classes,
            channels=channels,
            config_file=config_file,
            affine=affine,
        )
    else:
        model = enMesh_checkpoint(
            in_channels=1,
            n_classes=n_classes,
            channels=channels,
            config_file=config_file,
        )

    # Disable gradient checkpointing — no backward pass, so recomputing
    # activations is pure waste.  Force False regardless of config.
    model.use_checkpoint = False
    print(f"[validate] model={model.__class__.__name__}  "
          f"channels={channels}  n_classes={n_classes}  "
          f"use_checkpoint=False (forced for inference)",
          flush=True)

    ckpt_path = cfg.paths.model
    if _USE_BEST:
        ckpt_path = os.path.join(cfg.paths.logdir, "model.best.pth")
        print(f"[validate] --best: using {ckpt_path}", flush=True)
    if not cfg.paths.loadcheckpoint or not ckpt_path:
        print("[validate] WARNING: loadcheckpoint=False or empty paths.model — "
              "running with random weights!", flush=True)
    elif not os.path.isfile(ckpt_path):
        print(f"[validate] WARNING: checkpoint not found at {ckpt_path!r} — "
              "running with random weights!", flush=True)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = _strip_compile_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[validate] missing keys ({len(missing)}): {missing[:5]}...",
                  flush=True)
        if unexpected:
            print(f"[validate] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}...", flush=True)
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"[validate] loaded checkpoint {ckpt_path!r}  ({size_mb:.1f} MB)",
              flush=True)

    model = model.to(device)

    # channels_last_3d if requested (matches training layout)
    perf = cfg.get("perf", {}) or {}
    if perf.get("channels_last", True):
        try:
            model = model.to(memory_format=torch.channels_last_3d)
            print("[validate] model -> channels_last_3d", flush=True)
        except Exception as exc:
            print(f"[validate] channels_last_3d failed: {exc}", flush=True)

    model.eval()
    return model


def build_valid_loader(cfg, db_host):
    """Build a DataLoader for the full MindfulTensors / HCP collection."""
    dl_cfg = cfg.get("dataloader", {}) or {}
    # No backward pass → GPU is idle between batches → data loading is the
    # bottleneck.  Use more workers and a high prefetch_factor so the next
    # volume is already decoded and pinned in host RAM before the GPU is free.
    num_workers = int(dl_cfg.get("num_workers", 4))
    prefetch_factor = int(dl_cfg.get("prefetch_factor", 8))

    db_name    = "MindfulTensors"
    collection = "HCP"
    index_id   = cfg.mongo.index_id
    db_fields  = (cfg.mongo.datafield, cfg.mongo.labelfield)

    print(f"[validate] connecting to mongo {db_host}  "
          f"db={db_name}  collection={collection}  "
          f"fields={db_fields}", flush=True)

    # Enumerate the ids that ACTUALLY exist, rather than assuming a gapless
    # 0..max range. HCP (and other MindfulTensors collections) have sparse /
    # non-zero-based ids; iterating range(max+1) hits absent ids whose records
    # come back empty -> "mytransform failed! 0 bytes". Prefer <collection>.meta
    # (robust to gaps), fall back to distinct ids on the .bin collection. This
    # mirrors generator.py:source_ids.
    _mc = MongoClient("mongodb://" + db_host + ":27017")
    try:
        valid_ids = sorted(int(x) for x in _mc[db_name][f"{collection}.meta"].distinct(index_id))
    except Exception:
        valid_ids = []
    if not valid_ids:
        valid_ids = sorted(int(x) for x in _mc[db_name][f"{collection}.bin"].distinct(index_id))
    _mc.close()
    num_examples = len(valid_ids)
    if num_examples == 0:
        raise SystemExit(f"[validate] no ids found in {db_name}/{collection}")
    print(f"[validate] {db_name}/{collection}: {num_examples} volumes "
          f"(ids {valid_ids[0]}..{valid_ids[-1]})", flush=True)

    client_creator = base.ClientCreator(db_host)
    client_creator.set_database(db_name)
    client_creator.set_collection(collection)
    client_creator.set_shape([256, 256, 256])
    client_creator.set_num_subcubes(1)

    # Full-volume collate (shape==256)
    collate_fn = client_creator.mycollate_full

    vdataset = MongoDataset(
        valid_ids,
        client_creator.mytransform,
        None,
        db_fields,
        normalize=unit_interval_normalize,
        id=index_id,
    )

    vsampler = DBBatchSampler(vdataset, batch_size=1, seed=42)

    # pin_memory=False: 256³ tensors need a lot of /dev/shm shared-memory
    # slots; exhausting them kills the worker and triggers the
    # rebuild_storage_fd / ConnectionRefused crash in the pin-memory thread.
    # The host→GPU transfer for a 67 MB volume is negligible vs the ~3 s
    # forward pass, so pin_memory buys nothing here.
    vloader = DataLoader(
        vdataset,
        sampler=vsampler,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=client_creator.create_client,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )
    return vloader


def build_criterion(cfg, device):
    mc = cfg.model
    n_classes = mc.n_classes
    off_brain_weight = 1.0  # no attenuation for pure validation
    class_weight = torch.FloatTensor(
        [off_brain_weight] + [1.0] * (n_classes - 1)
    ).to(device)
    label_smoothing = mc.get("label_smoothing", 0.01)
    generalized = mc.get("dice_generalized", False)
    if mc.get("loss_fused", False):
        loss_weight = list(mc.loss_weight)
        total = sum(loss_weight)
        loss_weight = [w / total for w in loss_weight]
        return CEDiceLoss(
            loss_weight=tuple(loss_weight),
            class_weight=class_weight,
            label_smoothing=label_smoothing,
            generalized=generalized,
        ).to(device)
    # fallback: CE only
    return torch.nn.CrossEntropyLoss(
        weight=class_weight, label_smoothing=label_smoothing
    ).to(device)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="gn_hdc_deep_fast_100", version_base=None)
def main(cfg: DictConfig):
    # Explicit GPU setup — mirrors what Catalyst's GPUEngine does.  Without
    # this, torch.cuda.is_available() can return True while the current device
    # is still unset, causing model.to("cuda") to silently fall back to CPU.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("[validate] WARNING: CUDA not available — running on CPU!", flush=True)
    print(f"[validate] device={device}  "
          f"cuda_device_count={torch.cuda.device_count()}  "
          f"current_device={torch.cuda.current_device() if device.type=='cuda' else 'n/a'}",
          flush=True)

    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    # --- wandb ---
    wandb_experiment = f"val HCP {cfg.model.model_label}"
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.team,
        name=wandb_experiment,
        job_type="validation",
        config={
            "checkpoint": cfg.paths.model,
            "db": "MindfulTensors",
            "collection": "HCP",
            "n_classes": cfg.model.n_classes,
            "model_channels": cfg.model.model_channels,
        },
    )
    print(f"[validate] wandb run: {run.url}", flush=True)

    model = load_model(cfg, device)
    criterion = build_criterion(cfg, device)
    vloader = build_valid_loader(cfg, db_host)

    n_classes = cfg.model.n_classes
    dice_scores = []
    losses = []

    perf = cfg.get("perf", {}) or {}
    channels_last = perf.get("channels_last", True)

    print(f"[validate] running validation over {len(vloader)} batches ...", flush=True)

    # Do NOT wrap the DataLoader loop in inference_mode: eval_forward() inside
    # CheckpointMixin already enters inference_mode for the model forward.
    # Nesting a second inference_mode around a non_blocking copy causes the
    # async-copied tensor to be treated as still on CPU in some PyTorch builds.
    # Use no_grad here so criterion / dice don't build a graph.
    for i, batch in enumerate(vloader):
        with torch.no_grad():
            sample, label = batch
            # Blocking transfers — ensure data is on GPU before forward.
            sample = sample.to(device)
            label = label.to(device)

            if channels_last:
                sample = sample.contiguous(memory_format=torch.channels_last_3d)

            if i == 0:
                p_dev = next(model.parameters()).device
                print(f"[validate] first batch: sample.device={sample.device}  "
                      f"model.device={p_dev}", flush=True)

            # eval_forward() internally wraps in torch.inference_mode()
            y_hat = model(sample)
            loss = criterion(y_hat, label)

            result = torch.squeeze(torch.argmax(y_hat, dim=1)).long()
            labels = torch.squeeze(label)
            dice = torch.mean(faster_dice(result, labels, range(n_classes)))

            dice_val = dice.item()
            loss_val = loss.item()
            dice_scores.append(dice_val)
            losses.append(loss_val)

            print(f"  [{i+1:3d}/{len(vloader)}]  dice={dice_val:.4f}  loss={loss_val:.4f}",
                  flush=True)

            wandb.log({
                "validation/dice": dice_val,
                "validation/loss": loss_val,
                "validation/step": i,
            })

            del sample, label, y_hat, result, labels, loss, dice

    if dice_scores:
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_loss = sum(losses) / len(losses)
        min_dice  = min(dice_scores)
        max_dice  = max(dice_scores)
    else:
        mean_dice = mean_loss = min_dice = max_dice = float("nan")

    summary = {
        "validation/macro_dice":      mean_dice,
        "validation/loss_mean":       mean_loss,
        "validation/dice_min":        min_dice,
        "validation/dice_max":        max_dice,
        "validation/n_volumes":       len(dice_scores),
    }
    wandb.log(summary)
    run.summary.update(summary)
    wandb.finish()

    print("\n" + "="*60)
    print(f"  Validation summary  (MindfulTensors / HCP full, n={len(dice_scores)})")
    print(f"  macro_dice : {mean_dice:.4f}  (min={min_dice:.4f}  max={max_dice:.4f})")
    print(f"  loss       : {mean_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
