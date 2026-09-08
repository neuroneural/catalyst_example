import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import os
import math
import random
import shutil
from packaging import version
import yaml
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.utils import load_checkpoint

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dice import faster_dice, DiceLoss, CEDiceLoss, GROUP_LUTS
from surface_metrics import all_class_metrics, label_name
from meshnet import enMesh_checkpoint, enMesh, enMesh_checkpoint_SE, enMesh_SE
from meshnet_gn import enMesh_checkpoint as enMesh_checkpoint_gn, SpatialAEMeshNet
from meshnetme import MeshNetME_checkpoint
from refiner import enDynamicMesh_checkpoint, enDynamicMesh
from distill import Distiller
from jdx import jdx_penalty
from mindfultensors.gencoords import CoordsGenerator
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler

from mindfultensors.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MongoDataset,
    MongoClient,
    MongoheadDataset,
    mtransform,
)

REFINER_CLASS_NAMES = {"KernelRefiner", "OneShotKernelRefiner", "BasisKernelAdapter"}


class EpochShuffleBatchSampler(DBBatchSampler):
    """DBBatchSampler that reshuffles the order every epoch.

    The stock DBBatchSampler re-seeds NumPy with the SAME fixed ``seed`` on every
    ``__iter__`` call, so it yields the identical permutation each epoch -> the
    batch order is frozen for the whole run (visible as periodic, epoch-aligned
    artifacts in the training loss). Here we offset the seed by an epoch counter
    so each epoch gets a different but fully reproducible permutation.

    The sampler lives in the main process (workers receive indices, not the
    sampler), so the counter advances correctly even with persistent_workers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = 0

    def __iter__(self):
        base = 0 if self.seed is None else int(self.seed)
        np.random.seed((base + self._epoch) % (2**32))
        self._epoch += 1
        return self.__chunks__(
            np.random.permutation(self.data_size), self.batch_size
        )


def _strip_compile_prefix(state_dict):
    """Strip the ``_orig_mod.`` prefix that ``torch.compile`` adds to state-dict
    keys, so checkpoints are portable between compiled and non-compiled models.

    If no keys carry the prefix the dict is returned unchanged (zero-copy).
    """
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


class CompileSafeCheckpointCallback(dl.CheckpointCallback):
    """Drop-in replacement for ``dl.CheckpointCallback`` that normalises
    ``_orig_mod.`` key prefixes on every save **and** every load so that
    checkpoints work regardless of whether ``torch.compile`` was active at
    save time, load time, both, or neither."""

    # -- load path ----------------------------------------------------------
    def _load(self, runner, resume_logpath=None, resume_model=None,
              resume_runner=None):
        # Sanitise the on-disk checkpoint *before* the parent class calls
        # load_state_dict, so the keys always match the raw model.
        for path in (resume_logpath, resume_model):
            if path is not None and os.path.isfile(path):
                sd = load_checkpoint(path)
                clean = _strip_compile_prefix(sd)
                if clean is not sd:  # keys were rewritten
                    torch.save(clean, path)
                    print(f"[ckpt] stripped _orig_mod. prefix from {path}",
                          file=sys.stderr, flush=True)

        # Two-head resume: the live model is a TwoHeadMeshNet (base.model.* +
        # head_aux.*) but the on-disk checkpoint is single-head (model.*). The
        # parent's strict load_state_dict would fail on the key mismatch. Load
        # the base ourselves, non-strict (leaving the fresh 18-class aux head
        # untouched), and take the model out of the parent's hands. A genuine
        # two-head checkpoint (already base.*/head_aux.*) is loaded verbatim.
        from two_head import TwoHeadMeshNet
        unwrapped = runner.engine.unwrap_model(runner.model)
        while hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        if isinstance(unwrapped, TwoHeadMeshNet) and resume_model and os.path.isfile(resume_model):
            sd = _strip_compile_prefix(load_checkpoint(resume_model))
            remapped = {}
            for k, v in sd.items():
                if k.startswith("base.") or k.startswith("head_aux."):
                    remapped[k] = v                 # already a two-head ckpt
                else:
                    remapped["base." + k] = v        # single-head -> base.*
            missing, unexpected = unwrapped.load_state_dict(remapped, strict=False)
            aux_missing = [m for m in missing if m.startswith("head_aux.")]
            base_missing = [m for m in missing if not m.startswith("head_aux.")]
            print(f"[two_head/_load] resumed base from {resume_model}: "
                  f"{len(base_missing)} base-missing (want 0), "
                  f"{len(aux_missing)} aux-fresh (want 2), "
                  f"{len(unexpected)} unexpected", file=sys.stderr, flush=True)
            resume_model = None  # prevent the parent from strict-loading again

        super()._load(
            runner,
            resume_logpath=resume_logpath,
            resume_model=resume_model,
            resume_runner=resume_runner,
        )

    # -- save path ----------------------------------------------------------
    def _save(self, runner, obj, logprefix):
        # For "model" mode, unwrap any OptimizedModule so state_dict() never
        # contains the _orig_mod. prefix in the first place.
        if self.mode == "model" and isinstance(obj, torch.nn.Module):
            unwrapped = runner.engine.unwrap_model(obj)
            # torch.compile wraps in torch._dynamo.OptimizedModule which
            # stores the real model in ._orig_mod.
            while hasattr(unwrapped, "_orig_mod"):
                unwrapped = unwrapped._orig_mod
            # Temporarily replace the runner's model so the parent's _save
            # sees the fully-unwrapped module.
            return super()._save(runner, unwrapped, logprefix)
        return super()._save(runner, obj, logprefix)


import sys
import gc
import time
from pymongo.errors import OperationFailure

# NOTE: no __getitem__ monkey-patching. Both train and validation use
# MongoheadDataset, whose native __getitem__ is wrapped in the library's
# @retry_on_eof_error (10 retries, 1s sleep, on EOFError/OperationFailure/
# RuntimeError) -- which is exactly the wirehead-swap protection we need:
# it retries the query on the SAME open client until the dropped-and-recreated
# collection reappears. The previous patch reimplemented this AND closed the
# client on error, which turned recoverable swaps into "Cannot use MongoClient
# after close" crashes.


SEED = random.randint(0, 9999)
utils.set_global_seed(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
# os.environ["NCCL_P2P_LEVEL"] = "NVL"

torch_version = torch.__version__
if version.parse(torch_version) >= version.parse("2.3"):
    scaler = torch.amp.GradScaler()
else:
    scaler = torch.cuda.amp.GradScaler()


def qnormalize(img, qmin=0.02, qmax=0.98):
    """Unit interval preprocessing with clipping"""
    qlow = torch.quantile(img, qmin)
    qhigh = torch.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = torch.clamp(img, 0, 1)  # Clip the values to be between 0 and 1
    return img


def crop_tensor(tensor, label, percentile=10):

    # Use torch.quantile instead of kthvalue for potentially faster operation
    threshold = torch.quantile(tensor.flatten(), percentile / 100)

    # Create a mask on the original device
    mask = tensor > threshold

    # If the mask is all False, return the original tensors
    if not torch.any(mask):
        return tensor, label

    # Find the bounding box (this part is already efficient)
    nonzero = torch.nonzero(mask)
    min_coords, _ = torch.min(nonzero, dim=0)
    max_coords, _ = torch.max(nonzero, dim=0)

    # Crop the original tensor and label using the bounding box
    slices = tuple(
        slice(min_coord.item(), max_coord.item() + 1)
        for min_coord, max_coord in zip(min_coords[2:], max_coords[2:])
    )
    cropped_tensor = tensor[(slice(None), slice(None)) + slices]
    cropped_label = label[(slice(None),) + slices]

    return cropped_tensor, cropped_label


class ProductScheduler:
    def __init__(self, scheduler1, scheduler2):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.initial_lr = scheduler1.optimizer.param_groups[0]["lr"]

    def step(self):
        lr1 = self.scheduler1.get_last_lr()[0]
        lr2 = self.scheduler2.get_last_lr()[0]
        combined_lr = lr1 * lr2
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler1.optimizer.param_groups[0]["lr"] = combined_lr
        return combined_lr


# CustomRunner – PyTorch for-loop decomposition
# https://github.com/catalyst-team/catalyst#minimal-examples
class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir: str,
        wandb_project: str,
        wandb_experiment: str,
        model_path: str,
        n_channels: int,
        n_classes: int,
        n_epochs: int,
        optimize_inline: bool,
        validation_percent: float,
        onecycle_lr: float,
        rmsprop_lr: float,
        num_subcubes: int,
        num_volumes: int,
        client_creator,
        off_brain_weight: float,
        indexid: str,
        modelconfig: str,
        db_host: str,
        db_name: str,
        db_collection: str,
        wandb_team: str,
        db_fields: tuple,
        groupnorm=False,
        affine=False,
        prefetches=8,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        valid_prefetch_factor=2,
        dice_every_n_steps=1,
        ddp_batch_sync=True,
        dice_subsample_stride=4,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        lowprecision=False,
        meshnetme=False,
        lossweight=[1, 0],
        label_smoothing=0.01,
        dice_generalized=False,
        loss_fused=False,
        boundary_weight=0.0,
        boundary_radius=8,
        boundary_include_bg=False,
        boundary_downsample=2,
        cldice_weight=0.0,
        cldice_iters=5,
        cldice_downsample=1,
        cldice_include_bg=False,
        cldice_classes=None,
        tversky_weight=0.0,
        tversky_alpha=0.7,
        tversky_beta=0.3,
        tversky_classes=None,
        group_lut=None,
        group_n_classes=None,
        group_cldice_weight=0.0,
        group_cldice_iters=5,
        group_cldice_downsample=1,
        group_cldice_include_bg=False,
        group_cldice_classes=None,
        group_tversky_weight=0.0,
        group_tversky_alpha=0.6,
        group_tversky_beta=0.4,
        group_tversky_classes=None,
        loss_log_terms=False,
        ce_class_weight_overrides=None,
        valid_cfg=None,
        maxshape=300,
        hparams=None,
        use_refiner=False,
        refiner_kwargs=None,
        refiner_delta_lambda=1e-3,
        refiner_delta_target=0.15,
        refiner_freeze_epochs=0,
        refiner_blend=1.0,
        refiner_bypass_prob=0.0,
        refiner_base_loss_lambda=0.0,
        me_kwargs=None,
        me_weight_diversity_lambda=0.0,
        use_se=False,
        se_kwargs=None,
        use_checkpoint=True,
        weight_decay=0.0,
        grad_clip=0.0,
        accum_steps=1,
        amp_dtype="float16",
        use_ema=False,
        ema_decay=0.999,
        use_spatial_ae=False,
        spatial_ae_mult=2,
        spatial_ae_down="avgpool",
        spatial_ae_up="transposed",
        sched_pct_start=0.1,
        sched_div_factor=100.0,
        sched_final_div=1e4,
        jdx_kwargs=None,
    ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.config_file = modelconfig
        self.optimize_inline = optimize_inline
        self.onecycle_lr = onecycle_lr
        self.rmsprop_lr = rmsprop_lr
        self.prefetches = prefetches
        self.num_workers = num_workers
        # persistent_workers=True avoids per-epoch worker respawn but makes workers
        # long-lived, so glibc malloc retention/fragmentation from the ~67MB buffer
        # churn (b"".join + lz4 + torch.load in the mindfultensors path) accumulates
        # as host-RAM creep -> OOM (seen ~epoch 27). Safe to set True ONLY with the
        # allocator tamed at launch: MALLOC_TRIM_THRESHOLD_=0 MALLOC_ARENA_MAX=2, or
        # LD_PRELOAD jemalloc/tcmalloc. Default False = respawn each epoch (bounded).
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self.valid_prefetch_factor = valid_prefetch_factor
        self.dice_every_n_steps = dice_every_n_steps
        self.ddp_batch_sync = ddp_batch_sync
        self.dice_subsample_stride = dice_subsample_stride
        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.db_fields = db_fields
        self.shape = subvolume_shape[0]
        self.num_subcubes = num_subcubes
        self.num_volumes = num_volumes
        self.n_epochs = n_epochs
        self.off_brain_weight = off_brain_weight
        self.client_creator = client_creator
        self.funcs = None
        self.collate = None
        self.bit16 = lowprecision
        self.index_id = indexid
        self.groupnorm = groupnorm
        self.affine = affine
        self.loss_weight = lossweight
        self.label_smoothing = label_smoothing
        self.dice_generalized = dice_generalized
        self.loss_fused = loss_fused
        self.boundary_weight = float(boundary_weight)
        self.boundary_radius = int(boundary_radius)
        self.boundary_include_bg = bool(boundary_include_bg)
        self.boundary_downsample = int(boundary_downsample)
        self.cldice_weight = float(cldice_weight)
        self.cldice_iters = int(cldice_iters)
        self.cldice_downsample = int(cldice_downsample)
        self.cldice_include_bg = bool(cldice_include_bg)
        self.cldice_classes = list(cldice_classes) if cldice_classes else None
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.tversky_classes = list(tversky_classes) if tversky_classes else None
        # Marginalized (group-space) aux terms. Defaults are inert: with
        # group_*_weight == 0 the criterion is built exactly as before.
        self.group_lut = list(group_lut) if group_lut else None
        self.group_n_classes = group_n_classes
        self.group_cldice_weight = float(group_cldice_weight)
        self.group_cldice_iters = int(group_cldice_iters)
        self.group_cldice_downsample = int(group_cldice_downsample)
        self.group_cldice_include_bg = bool(group_cldice_include_bg)
        self.group_cldice_classes = (list(group_cldice_classes)
                                     if group_cldice_classes else None)
        self.group_tversky_weight = float(group_tversky_weight)
        self.group_tversky_alpha = float(group_tversky_alpha)
        self.group_tversky_beta = float(group_tversky_beta)
        self.group_tversky_classes = (list(group_tversky_classes)
                                      if group_tversky_classes else None)
        self.loss_log_terms = bool(loss_log_terms)
        self.ce_class_weight_overrides = dict(ce_class_weight_overrides or {})
        # Optional separate validation source + surface metrics (e.g. real MRN).
        # Empty dict => legacy behavior (validate on synth range(32), dice only).
        self.valid_cfg = dict(valid_cfg or {})
        self.meshnetme = meshnetme
        self.wandb_team = wandb_team
        self.maxshape = maxshape
        self._hparams = hparams
        self.use_refiner = use_refiner
        self.refiner_kwargs = refiner_kwargs or {}
        self.refiner_delta_lambda = refiner_delta_lambda
        self.refiner_delta_target = refiner_delta_target
        self.refiner_freeze_epochs = refiner_freeze_epochs
        self.refiner_blend = refiner_blend
        self.refiner_bypass_prob = refiner_bypass_prob
        self.refiner_base_loss_lambda = refiner_base_loss_lambda
        self.me_kwargs = me_kwargs or {}
        self.me_weight_diversity_lambda = me_weight_diversity_lambda
        self.use_se = use_se
        self.se_kwargs = se_kwargs or {}
        self.use_checkpoint = use_checkpoint
        # Decoupled L2 (AdamW) on conv weights only. 0.0 => old behavior (plain
        # Adam, no decay). Norm/bias params are always excluded; see
        # get_optimizer. Keeps conv-weight/activation magnitudes down so the
        # exported model stays fp16-safe.
        self.weight_decay = float(weight_decay)
        # Clip total grad norm before the optimizer step (caps outlier-batch
        # gradients -> stops the transient train-dice collapses). 0.0 = off.
        self.grad_clip = float(grad_clip)
        # --- convergence-speed + EMA + spatial-AE knobs (default = old behavior) ---
        self.accum_steps = max(1, int(accum_steps))
        self.amp_dtype = amp_dtype
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.use_spatial_ae = bool(use_spatial_ae)
        self.spatial_ae_mult = int(spatial_ae_mult)
        self.spatial_ae_down = spatial_ae_down
        self.spatial_ae_up = spatial_ae_up
        self.sched_pct_start = float(sched_pct_start)
        self.sched_div_factor = float(sched_div_factor)
        self.sched_final_div = float(sched_final_div)
        self._ema_shadow = None
        self._ema_backup = None
        self._accum_count = 0
        self._local_epoch_index = 0
        self._refiner_frozen = False

        # --- JDX / JDRX channel-decorrelation pressure (default: OFF) ----------
        # Auxiliary per-layer loss that pushes feature channels to encode
        # DIFFERENT sources (minimize off-diagonal channel covariance on
        # structured sub-batches; see jdx.py + JDX.pdf eq. 3). Training-only:
        # never touches inference / the WebGPU export, and adds no peak memory
        # (activations already live in the graph with use_checkpoint=false).
        _jdx = jdx_kwargs or {}
        self.jdx_enabled = bool(_jdx.get("enabled", False))
        self.jdx_lambda = float(_jdx.get("lambda", 0.0))
        self.jdx_num_batches = int(_jdx.get("num_batches", 4))
        # subcube: int (uniform) OR list (per hooked layer; e.g. larger middle).
        _sub = _jdx.get("subcube", 9)
        self.jdx_subcube = list(_sub) if isinstance(_sub, (list, tuple, ListConfig)) \
            else int(_sub)
        # Restrict subcube sampling to the brain bounding box (from the label),
        # so covariances are estimated on tissue, not uninformative background.
        self.jdx_foreground = bool(_jdx.get("foreground", True))
        # "corr" (recommended): variance-normalized off-diagonal correlation
        # (Barlow-style; cannot be gamed by inflating channel variance).
        # "ratio": raw JDX off/diag covariance energy ratio (eq. 3, legacy).
        self.jdx_mode = str(_jdx.get("mode", "ratio"))
        # randomize_jdx: False => JDX (structured contiguous subcubes, the
        # promising spatial regime). True => JDRX (global i.i.d. voxel sampling).
        self.jdx_randomize = bool(_jdx.get("randomize_jdx", False))
        # Exclude the last N HIDDEN layers from the penalty. The penultimate
        # feature layers compress toward the task's class count (~3/18), so a
        # low channel rank there is appropriate, not waste -- pressuring them
        # fights the task loss (the diagnostic shows layer 12 resists it). The
        # reclaimable redundancy is in the early/middle layers. 0 = all layers.
        self.jdx_skip_last = int(_jdx.get("skip_last_layers", 0))
        self.jdx_warmup_steps = int(_jdx.get("warmup_steps", 0))
        self._jdx_acts = []
        self._jdx_handles = []
        self._jdx_hooked = False
        self._jdx_step = 0
        self._jdx_region = None
        self._last_jdx = None

    def set_refiner_blend(self, blend):
        if not getattr(self, "use_refiner", False) or not hasattr(self, "model"):
            return
        for module in self.model.modules():
            if hasattr(module, "refiner_blend"):
                module.refiner_blend = float(blend)

    def set_refiner_trainable(self, trainable):
        if not getattr(self, "use_refiner", False) or not hasattr(self, "model"):
            return
        blend = self.refiner_blend if trainable else 0.0
        for module in self.model.modules():
            if hasattr(module, "refiner_blend"):
                module.refiner_blend = blend
            if module.__class__.__name__ in REFINER_CLASS_NAMES:
                module.train(trainable)
                for p in module.parameters():
                    if not trainable:
                        p.grad = None
        self._refiner_frozen = not trainable

    def zero_refiner_grads(self):
        if not getattr(self, "_refiner_frozen", False):
            return
        for module in self.model.modules():
            if module.__class__.__name__ in REFINER_CLASS_NAMES:
                for p in module.parameters():
                    p.grad = None

    def snapshot_refiner_stats(self):
        snapshot = []
        for module in self.model.modules():
            if module.__class__.__name__ in REFINER_CLASS_NAMES and hasattr(module, "step"):
                snapshot.append((
                    module.step,
                    getattr(module.step, "last_fwd_iters", None),
                    getattr(module.step, "last_fwd_rel", None),
                    getattr(module.step, "last_fwd_delta", None),
                    getattr(module.step, "last_fwd_alpha", None),
                ))
        return snapshot

    def restore_refiner_stats(self, snapshot):
        for step, iters, rel, delta, alpha in snapshot:
            if iters is not None:
                step.last_fwd_iters = iters
            if rel is not None:
                step.last_fwd_rel = rel
            if delta is not None:
                step.last_fwd_delta = delta
            if alpha is not None:
                step.last_fwd_alpha = alpha

    def ddp_shared_random(self, device):
        value = torch.rand((), device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(value, src=0)
        return value.item()

    def is_rbp_model(self):
        if not hasattr(self, "model"):
            return False
        model = self.model.module if hasattr(self.model, "module") else self.model
        return hasattr(model, "step") and hasattr(model, "rbp_max_iter")

    def on_epoch_start(self, runner):
        super().on_epoch_start(runner)
        freeze_refiner = self._local_epoch_index < self.refiner_freeze_epochs
        self.set_refiner_trainable(not freeze_refiner)
        if getattr(self, "use_refiner", False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            "refiner/freeze/enabled": float(freeze_refiner),
                            "refiner/freeze/blend": 0.0 if freeze_refiner else self.refiner_blend,
                        },
                        commit=False,
                    )
            except Exception:
                pass
        self._local_epoch_index += 1

    def get_refiner_delta_penalty(self, device):
        penalties = []
        if not getattr(self, "use_refiner", False):
            return torch.zeros((), device=device)
        for module in self.model.modules():
            delta_ratio = getattr(module, "refiner_delta_ratio", None)
            if delta_ratio is not None:
                excess = torch.relu(delta_ratio - self.refiner_delta_target)
                penalties.append(excess.pow(2))
        if not penalties:
            return torch.zeros((), device=device)
        return torch.stack(penalties).mean()

    def get_me_weight_diversity_penalty(self, device):
        if not self.meshnetme or self.me_weight_diversity_lambda <= 0:
            return torch.zeros((), device=device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model, "weight_diversity_loss"):
            return torch.zeros((), device=device)
        return model.weight_diversity_loss()

    def _register_jdx_hooks(self, model):
        """Attach forward hooks that capture each hidden trunk activation for the
        JDX penalty. No-op unless jdx.enabled. Hooks fire only in training and
        only when JDX is active; captured tensors are read (never mutated)."""
        if not self.jdx_enabled or self._jdx_hooked:
            return
        import torch.nn as nn
        # Unwrap the two-head wrapper (if any) to reach the plain trunk Sequential
        # that ends in the deployment head Conv3d.
        base = getattr(model, "base", model)
        trunk = getattr(base, "model", None)
        if trunk is None or len(trunk) == 0:
            print("[jdx] WARNING: could not find trunk Sequential; JDX disabled",
                  file=sys.stderr, flush=True)
            self._jdx_hooked = True
            return
        n_hooked = 0
        layers = list(trunk)
        # Hook hidden layers [0, cutoff); cutoff excludes the head (last module)
        # plus the last jdx_skip_last hidden layers.
        cutoff = len(layers) - 1 - max(0, self.jdx_skip_last)
        for li, layer in enumerate(layers):
            if li >= cutoff:
                continue  # skip the head and the last jdx_skip_last hidden layers
            act_mod = None
            for m in layer.modules():
                if isinstance(m, (nn.GELU, nn.ReLU, nn.ELU)):
                    act_mod = m  # last activation in the block == its output act
            if act_mod is not None:
                self._jdx_handles.append(
                    act_mod.register_forward_hook(self._jdx_hook))
                n_hooked += 1
        self._jdx_hooked = True
        print(f"[jdx] enabled: hooked {n_hooked} activation layers "
              f"(sampling={'JDRX' if self.jdx_randomize else 'JDX'}, "
              f"objective={self.jdx_mode}, lambda={self.jdx_lambda}, "
              f"L={self.jdx_num_batches}, subcube={self.jdx_subcube}, "
              f"foreground={self.jdx_foreground}, skip_last={self.jdx_skip_last}, "
              f"warmup={self.jdx_warmup_steps})",
              file=sys.stderr, flush=True)

    def _jdx_hook(self, module, inputs, output):
        # Capture only during a training forward while JDX is active. The list is
        # cleared each step in get_jdx_penalty after it is consumed.
        if self.jdx_enabled and getattr(self, "model", None) is not None \
                and self.model.training:
            self._jdx_acts.append(output)

    def _jdx_foreground_region(self, label):
        """Brain bounding box (z0,z1,y0,y1,x0,x1) from the label (foreground =
        non-background class), unioned over the batch. Returns None if the label
        shape is unexpected or empty. MeshNet keeps every layer at input
        resolution, so this one box applies to all hooked activations."""
        try:
            with torch.no_grad():
                fg = label > 0
                while fg.dim() > 4:      # drop channel dim if present: [N,1,D,H,W]
                    fg = fg.any(1)
                if fg.dim() == 4:
                    fg = fg.any(0)       # union over batch -> [D,H,W]
                if fg.dim() != 3 or not bool(fg.any()):
                    return None
                zz = torch.where(fg.any(2).any(1))[0]
                yy = torch.where(fg.any(2).any(0))[0]
                xx = torch.where(fg.any(1).any(0))[0]
                return (int(zz[0]), int(zz[-1]) + 1,
                        int(yy[0]), int(yy[-1]) + 1,
                        int(xx[0]), int(xx[-1]) + 1)
        except Exception:
            return None

    def _jdx_weight_now(self):
        if self.jdx_warmup_steps <= 0:
            return self.jdx_lambda
        frac = min(1.0, self._jdx_step / float(self.jdx_warmup_steps))
        return self.jdx_lambda * frac

    def get_jdx_penalty(self, device):
        """Consume the activations captured this step and return the (unweighted)
        mean JDX energy ratio. Clears the capture buffer. Returns 0 if JDX is off
        or nothing was captured."""
        acts = self._jdx_acts
        self._jdx_acts = []
        if not self.jdx_enabled or not acts:
            return torch.zeros((), device=device)
        # Compute in fp32 with autocast disabled: this runs inside the bf16/fp16
        # autocast region, and a bf16 covariance would be too coarse for a
        # meaningful off/diag energy ratio.
        dev_type = "cuda" if (hasattr(device, "type") and device.type == "cuda") \
            else ("cuda" if torch.cuda.is_available() else "cpu")
        with torch.autocast(device_type=dev_type, enabled=False):
            pen = jdx_penalty(
                acts,
                num_batches=self.jdx_num_batches,
                subcube=self.jdx_subcube,
                randomize=self.jdx_randomize,
                mode=self.jdx_mode,
                region=getattr(self, "_jdx_region", None),
            )
        self._last_jdx = pen.detach()
        return pen.to(device)

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            # Unique DDP rendezvous port per SLURM job so concurrent DDP runs on
            # ONE node don't both grab Catalyst's default 2112 (EADDRINUSE). All
            # ranks of a job share SLURM_JOB_ID -> same port; different jobs differ.
            jid = int(os.environ.get("SLURM_JOB_ID", "0") or "0")
            port = 20000 + (jid % 20000)
            os.environ["MASTER_PORT"] = str(port)   # for the env-reading code path
            try:
                return dl.DistributedDataParallelEngine(
                    port=port,
                    process_group_kwargs={"backend": "nccl"},
                )
            except TypeError:                        # older Catalyst without `port`
                return dl.DistributedDataParallelEngine(
                    process_group_kwargs={"backend": "nccl"},
                )
        else:
            return dl.GPUEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            # "tensorboard": dl.TensorboardLogger(logdir=self._logdir,
            #                                     log_batch_metrics=True),
            "wandb": dl.WandbLogger(
                project=self.wandb_project,
                name=self.wandb_experiment,
                entity=self.wandb_team,
                log_batch_metrics=True,
                # log_epoch_metrics=True,
            ),
        }

    @property
    def stages(self):
        return ["train"]

    @property
    def num_epochs(self) -> int:
        return self.n_epochs

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        random_data = os.urandom(4)
        SEED = int.from_bytes(random_data, byteorder="big")
        utils.set_global_seed(SEED)
        return SEED

    def get_stage_len(self) -> int:
        return self.n_epochs

    def get_loaders(self):
        self.funcs = {
            "createclient": self.client_creator.create_client,
            "createVclient": self.client_creator.create_v_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }

        self.collate = (
            self.funcs["mycollate_full"]
            if self.shape == 256
            else self.funcs["mycollate"]
        )

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts = db[self.db_collection + ".bin"]
        num_examples = int(
            posts.find_one(sort=[(self.index_id, -1)])[self.index_id] + 1
        )
        client.close()  # Close MongoClient to prevent inheriting open sockets in child processes

        tdataset = MongoheadDataset(
            range(32,num_examples),
            # [
            #     int(x)
            #     for x in np.random.permutation(
            #         list(np.random.randint(0, num_examples, 8)) * 100
            #     )
            # ],
            self.funcs["mytransform"],
            None,
            self.db_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )

        # Reshuffle the training order every epoch (stock DBBatchSampler freezes
        # it). seed=SEED under DDP keeps it reproducible; the per-epoch offset
        # lives inside EpochShuffleBatchSampler.
        tsampler = (
            EpochShuffleBatchSampler(tdataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else EpochShuffleBatchSampler(tdataset, batch_size=self.num_volumes)
        )

        # One-time per process: confirm each DDP rank shuffles differently (i.e.
        # ranks see diverse data, not redundant copies). Read-only: uses a LOCAL
        # RandomState so it does NOT touch global RNG or the sampler's state.
        if not getattr(self, "_logged_sampler_diversity", False):
            try:
                import torch.distributed as dist
                rank = (
                    dist.get_rank()
                    if (dist.is_available() and dist.is_initialized())
                    else 0
                )
                seed_val = SEED if self.engine.is_ddp else None
                preview = (
                    np.random.RandomState(int(seed_val))
                    .permutation(len(tdataset))[:6].tolist()
                    if seed_val is not None else "n/a (seed=None)"
                )
                print(
                    f"[data-diversity] rank={rank} sampler_seed={seed_val} "
                    f"epoch0_first_idx={preview}",
                    file=sys.stderr, flush=True,
                )
            except Exception as exc:
                print(f"[data-diversity] log failed: {exc}",
                      file=sys.stderr, flush=True)
            self._logged_sampler_diversity = True

        # persistent_workers=False: the worker pool (+ its pinned buffers, Mongo
        # clients and IPC semaphores) is torn down and recreated every epoch
        # instead of living for the whole run. With persistent_workers=True the
        # per-epoch worker state accumulated ~28GB/epoch (host RAM) and OOM'd the
        # node at epoch 27 -- the "leaked semaphore" signature is worker IPC that
        # never got released. Re-spawn cost is a few seconds vs a ~7.5 min epoch
        # (run is compute-bound), so this is nearly free. The EpochShuffleBatchSampler
        # counter lives in the main process, so reshuffling is unaffected.
        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                sampler=tsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                num_workers=self.num_workers,
            ),
            num_prefetches=self.prefetches,
        )

        # Validation source. Default: same DB as train, range(32) (legacy). If a
        # `validation.db` override is configured (e.g. real MindfulTensors/MRN
        # with T1 + labelfused), build a DEDICATED client/dataset for it so the
        # eval reflects real-data quality, not the synth train distribution.
        vc = self.valid_cfg
        if vc.get("db"):
            v_host = vc.get("host", self.db_host)
            v_db = vc["db"]
            v_col = vc["collection"]
            v_fields = (vc.get("datafield", "T1"), vc.get("labelfield", "labelfused"))
            v_n = int(vc.get("num_subjects", 32))
            self.v_client_creator = ClientCreator(
                v_host, volume_shape=self.client_creator.volume_shape
            )
            self.v_client_creator.set_database(v_db)
            self.v_client_creator.set_collection(v_col)
            self.v_client_creator.set_shape([256, 256, 256])
            self.v_client_creator.set_num_subcubes(1)
            # MRN/HCP ids are sparse/non-zero-based -> enumerate the ids that
            # actually exist (prefer <col>.meta, fall back to <col>.bin), else
            # empty records come back as "mytransform 0 bytes". Cap at v_n.
            _c = MongoClient("mongodb://" + v_host + ":27017")
            try:
                v_ids = sorted(int(x) for x in _c[v_db][f"{v_col}.meta"].distinct(self.index_id))
            except Exception:
                v_ids = []
            if not v_ids:
                v_ids = sorted(int(x) for x in _c[v_db][f"{v_col}.bin"].distinct(self.index_id))
            _c.close()
            if not v_ids:
                raise SystemExit(f"[valid] no ids in {v_db}/{v_col}")
            v_ids = v_ids[:v_n]
            print(f"[valid] real-data eval: {v_db}/{v_col} fields={v_fields} "
                  f"n={len(v_ids)} (ids {v_ids[0]}..{v_ids[-1]})",
                  file=sys.stderr, flush=True)
            vdataset = MongoheadDataset(
                v_ids, self.v_client_creator.mytransform, None, v_fields,
                normalize=unit_interval_normalize, id=self.index_id,
            )
            v_worker_init = self.v_client_creator.create_client
            v_collate = self.v_client_creator.mycollate_full
        else:
            vdataset = MongoheadDataset(
                range(32),
                self.funcs["mytransform"],
                None,
                self.db_fields,
                normalize=unit_interval_normalize,
                id=self.index_id,
            )
            v_worker_init = self.funcs["createclient"]
            v_collate = self.collate

        vsampler = DBBatchSampler(vdataset, batch_size=self.num_volumes, seed=SEED)

        # Validation is a tiny set (range(32), ~1 iter/epoch). With
        # persistent_workers=True it would keep a SECOND full worker pool
        # (num_workers x pinned buffers x Mongo clients) resident for the whole
        # run, on top of the train loader's -- the resident step that OOM'd host
        # RAM at the first epoch boundary. Use a couple of NON-persistent workers
        # that are released after each validation, and a shallow prefetch queue.
        valid_workers = min(2, self.num_workers)
        vdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                vdataset,
                sampler=vsampler,
                collate_fn=v_collate,
                pin_memory=True,
                worker_init_fn=v_worker_init,
                persistent_workers=False,
                prefetch_factor=self.valid_prefetch_factor,
                num_workers=valid_workers,
            ),
            num_prefetches=min(2, self.prefetches),
        )

        return {"train": tdataloader, "valid": vdataloader}

    def get_model(self):
        if getattr(self, "use_spatial_ae", False) and not self.use_refiner:
            # Peak-memory-neutral spatial-AE bottleneck around the dilated trunk
            # (no skips). Inherits CheckpointMixin, so use_checkpoint /
            # checkpoint_segments / channels_last behave as for the flat model.
            model = SpatialAEMeshNet(
                in_channels=1,
                n_classes=self.n_classes,
                channels=self.n_channels,
                config_file=self.config_file,
                affine=self.affine,
                bottleneck_mult=getattr(self, "spatial_ae_mult", 2),
                downsample=getattr(self, "spatial_ae_down", "avgpool"),
                upsample=getattr(self, "spatial_ae_up", "transposed"),
            )
            model.use_checkpoint = self.use_checkpoint
            print(
                f"[get_model] SpatialAEMeshNet mult={getattr(self, 'spatial_ae_mult', 2)} "
                f"down={getattr(self, 'spatial_ae_down', 'avgpool')} "
                f"up={getattr(self, 'spatial_ae_up', 'transposed')} "
                f"use_checkpoint={self.use_checkpoint}",
                file=sys.stderr, flush=True,
            )
            return model
        if self.use_refiner:
            if self.shape > self.maxshape:
                model = enDynamicMesh(
                    in_channels=1,
                    n_classes=self.n_classes,
                    channels=self.n_channels,
                    config_file=self.config_file,
                    optimize_inline=self.optimize_inline,
                    groupnorm=self.groupnorm,
                    **self.refiner_kwargs,
                )
            else:
                model = enDynamicMesh_checkpoint(
                    in_channels=1,
                    n_classes=self.n_classes,
                    channels=self.n_channels,
                    config_file=self.config_file,
                    groupnorm=self.groupnorm,
                    **self.refiner_kwargs,
                )
        else:
            if self.meshnetme:
                modelClass = MeshNetME_checkpoint
            elif self.use_se:
                modelClass = enMesh_checkpoint_SE
            else:
                modelClass = (
                    enMesh_checkpoint_gn if self.groupnorm else enMesh_checkpoint
                )
            if self.shape > self.maxshape:
                if self.use_se:
                    model = enMesh_SE(
                        in_channels=1,
                        n_classes=self.n_classes,
                        channels=self.n_channels,
                        config_file=self.config_file,
                        optimize_inline=self.optimize_inline,
                        **self.se_kwargs,
                    )
                else:
                    model = enMesh(
                        in_channels=1,
                        n_classes=self.n_classes,
                        channels=self.n_channels,
                        config_file=self.config_file,
                        optimize_inline=self.optimize_inline,
                    )
            else:
                extra_kwargs = (
                    self.se_kwargs if self.use_se
                    else self.me_kwargs if self.meshnetme
                    else {"affine": self.affine} if self.groupnorm
                    else {}
                )
                model = modelClass(
                    in_channels=1,
                    n_classes=self.n_classes,
                    channels=self.n_channels,
                    config_file=self.config_file,
                    **extra_kwargs,
                )
        # Checkpointed model variants honor this flag in train_forward;
        # the manual-backprop variants (enMesh/enMesh_SE/enDynamicMesh) ignore it.
        model.use_checkpoint = self.use_checkpoint
        print(
            f"[get_model] {model.__class__.__name__} "
            f"use_checkpoint={self.use_checkpoint} shape={self.shape} maxshape={self.maxshape}",
            file=sys.stderr, flush=True,
        )

        # Two-head student: keep the deployment head, add a parallel aux head
        # matching the teacher's class count. Load the resume weights into the
        # single-head base BEFORE wrapping (keys match exactly), then attach the
        # fresh aux head. Use paths.loadcheckpoint=false for this run so Catalyst
        # does not also try to resume into the wrapper; init_from drives it here.
        th = getattr(self, "two_head_cfg", None)
        if th and th.get("enabled", False):
            from two_head import TwoHeadMeshNet
            init_from = th.get("init_from") or (self.model_path or "")
            aux_sd = {}
            if init_from and os.path.isfile(init_from):
                sd = _strip_compile_prefix(load_checkpoint(init_from))
                # tolerate BOTH a single-head checkpoint (model.*) and a two-head
                # one (base.*/head_aux.*): strip base., drop head_aux, so the base
                # loads cleanly either way (aux head reinitializes fresh here; if
                # resuming a two-head run, _load then restores the trained aux).
                base_sd = {}
                for k, v in sd.items():
                    if k.startswith("head_aux."):
                        aux_sd[k.removeprefix("head_aux.")] = v
                        continue
                    base_sd[k[len("base."):] if k.startswith("base.") else k] = v
                missing, unexpected = model.load_state_dict(base_sd, strict=False)
                print(f"[two_head] base loaded from {init_from} "
                      f"({len(missing)} missing, {len(unexpected)} unexpected)",
                      file=sys.stderr, flush=True)
            else:
                print(f"[two_head] WARNING: init_from not found ({init_from!r}); "
                      f"base starts from random init", file=sys.stderr, flush=True)
            model = TwoHeadMeshNet(model, aux_classes=int(th.get("aux_classes", 18)))
            if aux_sd:
                missing, unexpected = model.head_aux.load_state_dict(
                    aux_sd, strict=False)
                print(f"[two_head] aux resumed from {init_from} "
                      f"({len(missing)} missing, {len(unexpected)} unexpected)",
                      file=sys.stderr, flush=True)
            model.use_checkpoint = self.use_checkpoint
            print(f"[two_head] wrapped: deploy={model.n_classes} aux={model.aux_classes}",
                  file=sys.stderr, flush=True)

        # JDX pressure: tap the per-layer trunk activations via forward hooks.
        # Registered here (before Catalyst's DDP wrap + the in-place
        # torch.compile on the first batch) so the hooks are part of the
        # initial compiled graph and never trigger a recompile.
        self._register_jdx_hooks(model)
        return model

    def get_criterion(self):
        cw = [self.off_brain_weight] + [1.0] * (self.n_classes - 1)
        # Per-class CE upweighting for hard/small structures (CE term only; the
        # generalized-Dice term keeps its own inverse-volume weighting). Maps
        # class_index -> multiplier, e.g. {4: 3.0, 5: 2.0, 12: 2.0}. Default {} =
        # uniform. Only the CE gradient is reweighted, so the deployed model and
        # its memory are unchanged.
        overrides = getattr(self, "ce_class_weight_overrides", None) or {}
        for k, v in overrides.items():
            ki = int(k)
            if 0 <= ki < self.n_classes:
                cw[ki] = float(v)
        if overrides:
            print(f"[loss] CE class-weight overrides: {overrides}",
                  file=sys.stderr, flush=True)
        class_weight = torch.FloatTensor(cw).to(self.engine.device)
        label_smoothing = getattr(self, "label_smoothing", 0.01)
        generalized = getattr(self, "dice_generalized", False)

        boundary_weight = float(getattr(self, "boundary_weight", 0.0))
        boundary_radius = int(getattr(self, "boundary_radius", 8))
        boundary_include_bg = bool(getattr(self, "boundary_include_bg", False))
        boundary_downsample = int(getattr(self, "boundary_downsample", 2))
        cldice_weight = float(getattr(self, "cldice_weight", 0.0))
        cldice_iters = int(getattr(self, "cldice_iters", 5))
        cldice_downsample = int(getattr(self, "cldice_downsample", 1))
        cldice_include_bg = bool(getattr(self, "cldice_include_bg", False))
        cldice_classes = getattr(self, "cldice_classes", None)
        tversky_weight = float(getattr(self, "tversky_weight", 0.0))
        tversky_alpha = float(getattr(self, "tversky_alpha", 0.7))
        tversky_beta = float(getattr(self, "tversky_beta", 0.3))
        tversky_classes = getattr(self, "tversky_classes", None)
        group_lut = getattr(self, "group_lut", None)
        group_n_classes = getattr(self, "group_n_classes", None)
        group_cldice_weight = float(getattr(self, "group_cldice_weight", 0.0))
        group_cldice_iters = int(getattr(self, "group_cldice_iters", 5))
        group_cldice_downsample = int(getattr(self, "group_cldice_downsample", 1))
        group_cldice_include_bg = bool(getattr(self, "group_cldice_include_bg", False))
        group_cldice_classes = getattr(self, "group_cldice_classes", None)
        group_tversky_weight = float(getattr(self, "group_tversky_weight", 0.0))
        group_tversky_alpha = float(getattr(self, "group_tversky_alpha", 0.6))
        group_tversky_beta = float(getattr(self, "group_tversky_beta", 0.4))
        group_tversky_classes = getattr(self, "group_tversky_classes", None)
        loss_log_terms = bool(getattr(self, "loss_log_terms", False))

        # Fused path: one log_softmax shared by CE and Dice (saves a full
        # softmax volume at 104 classes / 256^3). Opt-in via cfg.model.loss_fused.
        # The Kervadec boundary term (boundary_weight>0) also reuses that softmax.
        if getattr(self, "loss_fused", False):
            if boundary_weight > 0:
                print(f"[loss] Kervadec boundary term ON: weight={boundary_weight} "
                      f"radius={boundary_radius} include_bg={boundary_include_bg} "
                      f"downsample={boundary_downsample}",
                      file=sys.stderr, flush=True)
            if cldice_weight > 0:
                print(f"[loss] clDice topology term ON: weight={cldice_weight} "
                      f"iters={cldice_iters} downsample={cldice_downsample} "
                      f"include_bg={cldice_include_bg} classes={cldice_classes or 'all-fg'}",
                      file=sys.stderr, flush=True)
            if tversky_weight > 0:
                print(f"[loss] Tversky term ON: weight={tversky_weight} "
                      f"alpha={tversky_alpha} beta={tversky_beta} "
                      f"classes={tversky_classes or 'all-fg'}",
                      file=sys.stderr, flush=True)
            if group_tversky_weight > 0 or group_cldice_weight > 0:
                n_g = (int(group_n_classes) if group_n_classes
                       else (max(group_lut) + 1 if group_lut else 0))
                print(f"[loss] MARGINALIZED aux terms ON: {self.n_classes} classes "
                      f"-> {n_g} groups (lut len {len(group_lut or [])})",
                      file=sys.stderr, flush=True)
                if group_tversky_weight > 0:
                    print(f"[loss]   group Tversky: weight={group_tversky_weight} "
                          f"alpha={group_tversky_alpha} beta={group_tversky_beta} "
                          f"GROUP classes={group_tversky_classes or 'all-fg'}",
                          file=sys.stderr, flush=True)
                if group_cldice_weight > 0:
                    print(f"[loss]   group clDice: weight={group_cldice_weight} "
                          f"iters={group_cldice_iters} "
                          f"downsample={group_cldice_downsample} "
                          f"GROUP classes={group_cldice_classes or 'all-fg'}",
                          file=sys.stderr, flush=True)
            if loss_log_terms:
                print("[loss] per-term logging ON (wandb only). Set "
                      "FAST_COMPILE_LOSS=0: writing terms to self inside "
                      "forward breaks the compiled graph.",
                      file=sys.stderr, flush=True)
            return CEDiceLoss(
                loss_weight=tuple(self.loss_weight),
                class_weight=class_weight,
                label_smoothing=label_smoothing,
                generalized=generalized,
                boundary_weight=boundary_weight,
                boundary_radius=boundary_radius,
                boundary_include_bg=boundary_include_bg,
                boundary_downsample=boundary_downsample,
                cldice_weight=cldice_weight,
                cldice_iters=cldice_iters,
                cldice_downsample=cldice_downsample,
                cldice_include_bg=cldice_include_bg,
                cldice_classes=cldice_classes,
                tversky_weight=tversky_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_classes=tversky_classes,
                group_lut=group_lut,
                group_n_classes=group_n_classes,
                group_cldice_weight=group_cldice_weight,
                group_cldice_iters=group_cldice_iters,
                group_cldice_downsample=group_cldice_downsample,
                group_cldice_include_bg=group_cldice_include_bg,
                group_cldice_classes=group_cldice_classes,
                group_tversky_weight=group_tversky_weight,
                group_tversky_alpha=group_tversky_alpha,
                group_tversky_beta=group_tversky_beta,
                group_tversky_classes=group_tversky_classes,
                log_terms=loss_log_terms,
            ).to(self.engine.device)

        if (boundary_weight > 0 or cldice_weight > 0 or tversky_weight > 0
                or group_cldice_weight > 0 or group_tversky_weight > 0):
            raise ValueError(
                "boundary_weight/cldice_weight/tversky_weight/group_* > 0 require "
                "model.loss_fused=True (all live in the fused CEDiceLoss path)."
            )

        ce_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing
        )
        dice_criterion = DiceLoss(generalized=generalized)

        def combined_loss(output, target):
            if self.loss_weight[0] == 1:
                combined_loss = ce_criterion(output, target)
            elif self.loss_weight[1] == 1:
                combined_loss = dice_criterion(output, target)
            else:
                combined_loss = self.loss_weight[0] * ce_criterion(
                    output, target
                ) + self.loss_weight[1] * dice_criterion(output, target)
            return combined_loss

        return combined_loss

    @staticmethod
    def _decay_param_groups(model, weight_decay):
        """Split params so decay hits only conv weights. GroupNorm gamma/beta
        (1-D) and biases get weight_decay=0 -- decaying the affine scale would
        fight the deep GroupNorm stack."""
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):  # GroupNorm gamma/beta + biases
                no_decay.append(p)
            else:
                decay.append(p)                          # conv weights
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def get_optimizer(self, model):
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        wd = float(getattr(self, "weight_decay", 0.0))
        if wd > 0.0:
            # AdamW (decoupled decay), excluding norms/biases. Top-level
            # weight_decay=0.0 so the per-group values are authoritative.
            groups = self._decay_param_groups(model, wd)
            optimizer = torch.optim.AdamW(
                groups, lr=self.onecycle_lr, weight_decay=0.0
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.onecycle_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        # With gradient accumulation the optimizer (and thus the scheduler) steps
        # once per `accum_steps` micro-batches, so OneCycle's total step budget
        # must shrink accordingly. accum_steps=1 reproduces the old schedule.
        accum = max(1, int(getattr(self, "accum_steps", 1)))
        steps_per_epoch = max(1, len(self.loaders["train"]) // accum)
        # pct_start is the FRACTION of the whole cycle spent ramping up. With the
        # old 10-rep curriculum each cycle was short so 0.1 was fine; in a single
        # long run (maxreps=1, many epochs) 0.1 becomes a huge warmup (e.g. ~75k
        # steps) that pins LR at peak forever. Keep it small for single long runs.
        pct_start = float(getattr(self, "sched_pct_start", 0.1))
        div_factor = float(getattr(self, "sched_div_factor", 100.0))
        final_div_factor = float(getattr(self, "sched_final_div", 1e4))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            pct_start=pct_start,
            epochs=self.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return scheduler

    # ----------------------- EMA (eval/checkpoint weights) -----------------------
    # Maintains an exponential moving average of the weights. The averaged
    # weights are swapped in for the validation loader and held through the
    # end-of-epoch checkpoint save (so the saved "best" model is the EMA model),
    # then the raw training weights are restored at the next train loader start.
    # All swaps are in-place (copy_/load_state_dict) so the optimizer keeps
    # referencing the same parameter tensors. use_ema=False => fully inert.
    def _ema_module(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _ema_update(self):
        if not getattr(self, "use_ema", False):
            return
        m = self._ema_module()
        d = float(getattr(self, "ema_decay", 0.999))
        with torch.no_grad():
            msd = m.state_dict()
            if getattr(self, "_ema_shadow", None) is None:
                self._ema_shadow = {k: v.detach().clone() for k, v in msd.items()}
                return
            for k, v in self._ema_shadow.items():
                src = msd[k]
                if v.is_floating_point():
                    v.mul_(d).add_(src.detach().to(v.dtype), alpha=1.0 - d)
                else:
                    v.copy_(src)

    def _ema_swap_in(self):
        if getattr(self, "_ema_shadow", None) is None:
            return
        m = self._ema_module()
        self._ema_backup = {k: v.detach().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(self._ema_shadow, strict=False)

    def _ema_restore(self):
        if getattr(self, "_ema_backup", None) is None:
            return
        m = self._ema_module()
        m.load_state_dict(self._ema_backup, strict=False)
        self._ema_backup = None

    def get_callbacks(self):
        checkpoint_params = {
            # "sync": False,
            "save_best": True,
            "metric_key": "macro_dice",
            "loader_key": "valid",
            "minimize": False,
        }
        if self.model_path:
            checkpoint_params.update({"resume_model": self.model_path})
        return {
            "checkpoint": CompileSafeCheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    # ---- validation surface metrics (NSD@tau, HD95, per-class/worst Dice) ----
    def _surface_enabled(self):
        return bool(self.valid_cfg.get("surface_metrics", False))

    def _reset_surface_acc(self):
        self._surf = {
            c: {"dice": [], "nsd": [], "hd95": [], "assd": [], "miss": 0, "gtvox": []}
            for c in range(1, self.n_classes)
        }
        self._surf_subj = []          # per-subject mean foreground Dice

    def _accumulate_surface(self, pred_t, label_t):
        """pred_t: [.,D,H,W] argmax; label_t: [.,(1),D,H,W] int GT. Runs on CPU
        (scipy), guarded so an eval hiccup never kills training."""
        try:
            tau = float(self.valid_cfg.get("nsd_tau", 1.0))
            spacing = tuple(float(x) for x in self.valid_cfg.get("spacing", [1.0, 1.0, 1.0]))
            fail_dice = float(self.valid_cfg.get("fail_dice", 0.5))
            exclude = set(int(x) for x in self.valid_cfg.get("metric_exclude_classes", []))
            pred = pred_t.reshape((-1,) + tuple(pred_t.shape[-3:])).to(torch.int16).cpu().numpy()
            gt = label_t.reshape((-1,) + tuple(label_t.shape[-3:])).to(torch.int64)
            gt = torch.where(gt < self.n_classes, gt, torch.zeros_like(gt))  # first-18 clamp
            gt = gt.to(torch.int16).cpu().numpy()
            for b in range(pred.shape[0]):
                per = all_class_metrics(gt[b], pred[b], self.n_classes,
                                        spacing=spacing, tau=tau)
                fg = []
                for c, r in per.items():
                    if r["status"] in ("ok", "empty_pred"):     # GT present
                        self._surf[c]["dice"].append(r["dice"])
                        self._surf[c]["nsd"].append(r["nsd"])
                        self._surf[c]["assd"].append(r["assd"])
                        if r["status"] == "empty_pred":
                            self._surf[c]["miss"] += 1
                        else:
                            self._surf[c]["hd95"].append(r["hd95"])
                        self._surf[c]["gtvox"].append(r["gt_vox"])
                        if c not in exclude:            # context classes off the headline
                            fg.append(r["dice"])
                if fg:
                    self._surf_subj.append(float(np.mean(fg)))
        except Exception as exc:
            print(f"[valid-surface] skipped a batch: {exc}", file=sys.stderr, flush=True)

    def _finalize_surface_metrics(self, runner):
        """Aggregate per-class + tail, DDP all-reduce, log to wandb + stderr."""
        import numpy as _np
        C = self.n_classes
        dev = getattr(self.engine, "device", "cpu")
        # per-class sums/counts as tensors for a single all_reduce each
        sum_dice = torch.zeros(C, device=dev); cnt = torch.zeros(C, device=dev)
        sum_nsd = torch.zeros(C, device=dev)
        sum_hd = torch.zeros(C, device=dev); cnt_hd = torch.zeros(C, device=dev)
        miss = torch.zeros(C, device=dev)
        gtvox = torch.zeros(C, device=dev)      # total GT voxels/class (vol weights)
        min_dice = torch.full((C,), float("inf"), device=dev)
        max_hd = torch.full((C,), float("-inf"), device=dev)
        for c in range(1, C):
            d = self._surf[c]
            if d["dice"]:
                sum_dice[c] = float(_np.sum(d["dice"])); cnt[c] = len(d["dice"])
                sum_nsd[c] = float(_np.sum(d["nsd"]))
                min_dice[c] = float(_np.min(d["dice"]))
            if d["hd95"]:
                sum_hd[c] = float(_np.sum(d["hd95"])); cnt_hd[c] = len(d["hd95"])
                max_hd[c] = float(_np.max(d["hd95"]))
            gtvox[c] = float(_np.sum(d["gtvox"])) if d["gtvox"] else 0.0
            miss[c] = d["miss"]
        fail_dice = float(self.valid_cfg.get("fail_dice", 0.5))
        subj = self._surf_subj
        n_subj = torch.tensor(float(len(subj)), device=dev)
        n_fail = torch.tensor(float(sum(1 for s in subj if s < fail_dice)), device=dev)
        worst = torch.tensor(min(subj) if subj else float("inf"), device=dev)

        if self.engine.is_ddp:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    for t in (sum_dice, cnt, sum_nsd, sum_hd, cnt_hd, miss, gtvox, n_subj, n_fail):
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(min_dice, op=dist.ReduceOp.MIN)
                    dist.all_reduce(max_hd, op=dist.ReduceOp.MAX)
                    dist.all_reduce(worst, op=dist.ReduceOp.MIN)
            except Exception as exc:
                print(f"[valid-surface] all_reduce skipped: {exc}", file=sys.stderr, flush=True)

        rows = []
        for c in range(1, C):
            n = int(cnt[c].item()); nh = int(cnt_hd[c].item())
            rows.append((c, n, int(miss[c].item()),
                         (sum_dice[c].item() / n) if n else float("nan"),
                         (min_dice[c].item() if n else float("nan")),
                         (sum_nsd[c].item() / n) if n else float("nan"),
                         (sum_hd[c].item() / nh) if nh else float("nan"),
                         (max_hd[c].item() if nh else float("nan"))))
        # Headline aggregates exclude context/undeployed classes (e.g. CSF/skull)
        # so the number stays comparable to runs without them. Per-class table
        # below still shows every class.
        exclude = set(int(x) for x in self.valid_cfg.get("metric_exclude_classes", []))
        agg = [r for r in rows if r[0] not in exclude]
        macro_dice = _np.nanmean([r[3] for r in agg]) if agg else float("nan")
        macro_nsd = _np.nanmean([r[5] for r in agg]) if agg else float("nan")
        macro_hd = _np.nanmean([r[6] for r in agg]) if agg else float("nan")
        # Volume-weighted Dice: each class weighted by its GT volume, so the big
        # structures dominate and the tiny hard ones (class 4) barely count -- the
        # "overall voxel quality" lens (vs macro's structure-equal lens).
        _w = _np.array([gtvox[r[0]].item() for r in agg])
        _d = _np.array([r[3] for r in agg])
        _ok = _np.isfinite(_d) & (_w > 0)
        volw_dice = float((_w[_ok] * _d[_ok]).sum() / _w[_ok].sum()) if _ok.any() else float("nan")
        nsubj = int(n_subj.item()); nfail = int(n_fail.item())
        worst_v = worst.item()
        fail_rate = nfail / max(1, nsubj)

        print("\n[valid-surface] per-class (Dice / minDice / NSD@%.1fmm / HD95 / HD95max):"
              % float(self.valid_cfg.get("nsd_tau", 1.0)), file=sys.stderr, flush=True)
        for (c, n, m, dm, dmin, nsd, hd, hdmx) in rows:
            print(f"  cls {c:2d} {label_name(c):>12} n={n:3d} miss={m:2d}  "
                  f"dice={dm:.3f} min={dmin:.3f}  nsd={nsd:.3f}  "
                  f"hd95={hd:.2f} max={hdmx:.2f}", file=sys.stderr, flush=True)
        _excl = f" [excl {sorted(exclude)}]" if exclude else ""
        print(f"[valid-surface] MACRO dice={macro_dice:.4f} volw_dice={volw_dice:.4f} "
              f"nsd={macro_nsd:.4f} hd95={macro_hd:.2f}mm | worst-subj={worst_v:.4f} "
              f"fail(<{fail_dice})={fail_rate:.3f} ({nfail}/{nsubj}){_excl}",
              file=sys.stderr, flush=True)

        # loader_metrics show in the epoch summary + wandb, NOT the per-batch tqdm
        # bar (that's fed by self.meters), so these stay off the progress bar.
        self.loader_metrics["surf_macro_dice"] = macro_dice
        self.loader_metrics["surf_volw_dice"] = volw_dice
        self.loader_metrics["surf_macro_nsd"] = macro_nsd
        self.loader_metrics["surf_macro_hd95"] = macro_hd
        self.loader_metrics["surf_worst_subject_dice"] = worst_v
        try:
            import wandb
            if wandb.run is not None:
                payload = {"surface/valid/macro_dice": macro_dice,
                           "surface/valid/volw_dice": volw_dice,
                           "surface/valid/macro_nsd": macro_nsd,
                           "surface/valid/macro_hd95": macro_hd,
                           "surface/valid/worst_subject_dice": worst_v,
                           "surface/valid/failure_rate": fail_rate}
                for (c, n, m, dm, dmin, nsd, hd, hdmx) in rows:
                    payload[f"surface/valid/dice_c{c}"] = dm
                    payload[f"surface/valid/nsd_c{c}"] = nsd
                    payload[f"surface/valid/hd95_c{c}"] = hd
                wandb.log(payload, commit=False)
        except Exception:
            pass

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        # Surface metrics are CPU/scipy and cost minutes per pass -> only run them
        # every `surface_every_n_epochs` (Dice stays on GPU, logged every epoch).
        self._surf_this_epoch = False
        if getattr(runner, "loader_key", "") == "valid" and self._surface_enabled():
            # Count validation passes ourselves (robust to Catalyst version's
            # epoch-attribute naming). Fire on the FIRST pass, then every N.
            self._valid_pass = getattr(self, "_valid_pass", 0) + 1
            every = int(self.valid_cfg.get("surface_every_n_epochs", 1))
            self._surf_this_epoch = (
                self._valid_pass == 1 or every <= 1 or self._valid_pass % every == 0
            )
            if self._surf_this_epoch:
                self._reset_surface_acc()
        keys = ["loss", "macro_dice", "learning rate"]
        if getattr(self, "use_refiner", False):
            keys += ["refiner_iters_mean", "refiner_iters_min", "refiner_iters_max", "refiner_iters_relative_mean", "refiner_residual_mean", "refiner_residual_min", "refiner_residual_max", "refiner_avg_kernel_delta", "refiner_avg_alpha", "refiner_coeff_abs_mean", "refiner_coeff_abs_max", "refiner_delta_penalty", "refiner_scheduled_blend", "refiner_effective_blend", "refiner_bypassed", "refiner_bypass_prob", "refiner_base_loss", "refiner_base_loss_weighted"]
        if self.is_rbp_model():
            keys += ["rbp_iters", "rbp_iters_relative", "rbp_residual", "rbp_state_delta"]
        # (two-head kd/marg are wandb-only, logged directly in handle_batch;
        #  intentionally NOT meters, so they never reach the tqdm bar.)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in keys
        }
        # EMA: validate (and checkpoint) on averaged weights; restore raw
        # weights at the next train loader. Also reset gradient-accumulation
        # state and clear any stale grads at the start of each train loader.
        loader_key = getattr(runner, "loader_key", "")
        if getattr(self, "use_ema", False):
            if loader_key == "valid":
                self._ema_swap_in()
            elif loader_key == "train":
                self._ema_restore()
        if loader_key == "train":
            self._accum_count = 0
            try:
                self.optimizer.zero_grad()
            except Exception:
                pass

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        keys = ["loss", "macro_dice", "learning rate"]
        if getattr(self, "use_refiner", False):
            keys += ["refiner_iters_mean", "refiner_iters_min", "refiner_iters_max", "refiner_iters_relative_mean", "refiner_residual_mean", "refiner_residual_min", "refiner_residual_max", "refiner_avg_kernel_delta", "refiner_avg_alpha", "refiner_coeff_abs_mean", "refiner_coeff_abs_max", "refiner_delta_penalty", "refiner_scheduled_blend", "refiner_effective_blend", "refiner_bypassed", "refiner_bypass_prob", "refiner_base_loss", "refiner_base_loss_weighted"]
        if self.is_rbp_model():
            keys += ["rbp_iters", "rbp_iters_relative", "rbp_residual", "rbp_state_delta"]
        # (two-head kd/marg are wandb-only, not meters -- see handle_batch.)
        loader_key = getattr(runner, "loader_key", getattr(self, "loader_key", "loader"))
        refiner_epoch_metrics = {}
        for key in keys:
            value = self.meters[key].compute()[0]
            if key.startswith("refiner_"):
                refiner_epoch_metrics[key] = value
            elif key.startswith("rbp_"):
                refiner_epoch_metrics[key] = value
            else:
                self.loader_metrics[key] = value
        if refiner_epoch_metrics:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            f"{'rbp' if key.startswith('rbp_') else 'refiner'}/{loader_key}/epoch/{key.removeprefix('refiner_').removeprefix('rbp_')}": value
                            for key, value in refiner_epoch_metrics.items()
                        },
                        commit=False,
                    )
            except Exception:
                pass
        if loader_key == "valid" and getattr(self, "_surf_this_epoch", False) and hasattr(self, "_surf"):
            self._finalize_surface_metrics(runner)
        # Per-epoch cleanup insurance against host-RAM creep: force a GC sweep so
        # any dropped-but-uncollected loader/prefetch references (and their pinned
        # buffers) are freed at the loader boundary. NOTE: do NOT empty_cache() here
        # -- the leak is HOST RAM; empty_cache only releases GPU blocks and forces a
        # slow cudaMalloc re-grow on the next epoch (added latency, no benefit).
        gc.collect()
        super().on_loader_end(runner)

    def _get_distiller(self):
        """Lazily build the frozen teacher (device is only known once Catalyst
        set up the runner). Returns None if no teacher is configured. Shared by
        the single-head (marginalized) and two-head (raw-18) KD paths. The
        teacher runs train-time only, under no_grad, in eval -- peak memory and
        export are unaffected."""
        cfg = getattr(self, "distill_cfg", None)
        if not cfg or not cfg.get("teacher_checkpoint"):
            return None
        if getattr(self, "_distiller", None) is None:
            _bf16 = str(getattr(self, "amp_dtype", "float16")).lower() in ("bf16", "bfloat16")
            self._distiller = Distiller(
                checkpoint_path=cfg["teacher_checkpoint"],
                device=self.engine.device,
                teacher_channels=int(cfg.get("teacher_channels", 16)),
                teacher_classes=int(cfg.get("teacher_classes", 18)),
                student_classes=self.n_classes,
                config_file=cfg.get("teacher_config_file", self.config_file),
                affine=bool(cfg.get("teacher_affine", True)),
                temperature=float(cfg.get("temperature", 2.0)),
                amp_dtype=torch.bfloat16 if _bf16 else torch.float16,
                channels_last=bool(getattr(self, "channels_last", True)),
            )
        return self._distiller

    def _maybe_kd_loss(self, y_hat, sample):
        """Single-head (marginalized 18->3) KD term. Returns (kd, alpha) or
        (None, 0.0) when disabled. Superseded by the two-head path when
        model.two_head.enabled is set."""
        cfg = getattr(self, "distill_cfg", None)
        if not cfg or not cfg.get("enabled", False) or not self.model.training:
            return None, 0.0
        d = self._get_distiller()
        if d is None:
            return None, 0.0
        alpha = float(cfg.get("alpha", 0.5))
        kd = d.kd_loss(y_hat, sample)
        self._last_kd = kd.detach()
        return kd, alpha

    def _two_head_on(self):
        th = getattr(self, "two_head_cfg", None)
        return bool(th and th.get("enabled", False))

    def _supervised_missing_on(self):
        th = getattr(self, "two_head_cfg", None)
        return bool(th and th.get("enabled", False)
                    and th.get("supervision") == "missing_tissue")

    def _aux_weight_now(self):
        """Warm up the supervised missing-tissue head before it reshapes trunk."""
        th = self.two_head_cfg
        w = float(th.get("aux_loss_weight", 0.2))
        self._aux_step = getattr(self, "_aux_step", 0) + 1
        warm = int(th.get("aux_warmup_steps", 2000))
        if warm > 0:
            w *= min(1.0, self._aux_step / warm)
        return w

    def _apply_auxiliary_head_loss(self, loss, y_hat, y_aux, sample, aux_target):
        """Add either supervised missing-tissue GDL or legacy teacher KD."""
        if y_aux is None:
            return loss
        if self._supervised_missing_on():
            if y_aux.shape[1] != 2:
                raise ValueError("missing_tissue supervision requires aux_classes=2")
            if aux_target is None:
                raise RuntimeError("missing_tissue aux target was not constructed")
            aux_loss = DiceLoss(generalized=bool(
                self.two_head_cfg.get("aux_generalized_dice", True)))(
                    y_aux, aux_target)
            self._last_aux = aux_loss.detach()
            return loss + self._aux_weight_now() * aux_loss

        _d = self._get_distiller()
        if _d is not None:
            _kd18 = _d.kd18_loss(y_aux, sample)
            self._last_kd = _kd18.detach()
            loss = loss + self._kd_weight_now() * _kd18
            _lm = self._marg_weight_now()
            if _lm > 0.0:
                _mc = _d.marginal_consistency_loss(y_hat, y_aux)
                self._last_marg = _mc.detach()
                loss = loss + _lm * _mc
        return loss

    def _kd_weight_now(self):
        """Effective KD weight with a linear warmup. The raw 18-class KD is a
        T^2-scaled KL summed over 18 classes -- at init (random aux head) it is
        ~20-30x the 3-class dice+CE, so a large kd_weight lets it overwrite the
        resumed trunk (the deploy dice unlearns). Keep kd_weight small and ramp
        it in so the aux head first becomes sensible before it reshapes the
        trunk."""
        th = self.two_head_cfg
        w = float(th.get("kd_weight", 0.05))
        self._kd_step = getattr(self, "_kd_step", 0) + 1   # single per-step counter
        warm = int(th.get("kd_warmup_steps", 2000))
        if warm > 0:
            w *= min(1.0, self._kd_step / warm)
        return w

    def _marg_weight_now(self):
        """Effective weight for the marginal-consistency loss, with its own
        warmup. Must be called AFTER _kd_weight_now() each step (that advances
        the shared step counter). Ramping delays it while the aux head is still
        random -- chasing a garbage marginal early would hurt the deploy head."""
        th = self.two_head_cfg
        lm = float(th.get("lambda_marg", 0.0))
        if lm <= 0.0:
            return 0.0
        warm = int(th.get("marg_warmup_steps", th.get("kd_warmup_steps", 2000)))
        step = getattr(self, "_kd_step", 0)
        if warm > 0:
            lm *= min(1.0, step / warm)
        return lm

    # model train/valid step
    def handle_batch(self, batch):
        # Per-step full device sync. Default ON (historical behavior). It
        # serializes CPU<->GPU and can cost throughput; set perf.ddp_batch_sync
        # =False to drop it and A/B. Kept default-True so nothing changes unless
        # explicitly opted out.
        if self.engine.is_ddp and getattr(self, "ddp_batch_sync", True):
            torch.cuda.synchronize()

        sample, label = batch
        raw_label = label
        aux_target = None
        if self._supervised_missing_on():
            missing_id = int(self.two_head_cfg.get(
                "missing_target_id", self.n_classes))
            aux_target = (raw_label == missing_id).long()
        # Clamp any label index >= n_classes to 0 (background). CE gather /
        # class_weight[targets] / dice would otherwise index the class dim out of
        # bounds -> CUDA device-side assert. Two cases this covers: (1) real
        # validation labels (MRN `labelfused` carries extra classes), and (2)
        # training an N-class model on data with >N classes -- e.g. an 18-class
        # model on the 0-20 synth: CSF/skull fold back into background, which IS
        # the 18-class scheme. No-op when labels already fit [0, n_classes-1].
        label = torch.where(raw_label < self.n_classes, raw_label,
                            torch.zeros_like(raw_label))
        refiner_delta_penalty = torch.zeros((), device=sample.device)
        refiner_base_loss = torch.zeros((), device=sample.device)
        scheduled_blend = 0.0 if getattr(self, "_refiner_frozen", False) else self.refiner_blend
        effective_blend = scheduled_blend
        refiner_bypassed = 0.0
        # np.save("labels.npy", label.cpu().numpy())
        # np.save("input.npy", sample.cpu().numpy())
        # stop
        # run model forward/backward pass
        if self.model.training:
            # JDX: drop any activations captured by a previous (non-consuming)
            # forward and advance the warmup counter once per training step.
            if self.jdx_enabled:
                self._jdx_acts = []
                self._jdx_step += 1
                self._jdx_region = (
                    self._jdx_foreground_region(label)
                    if self.jdx_foreground else None
                )
            if scheduled_blend > 0 and self.refiner_bypass_prob > 0 and self.ddp_shared_random(sample.device) < self.refiner_bypass_prob:
                effective_blend = 0.0
                refiner_bypassed = 1.0
            self.set_refiner_blend(effective_blend)
            if self.shape > self.maxshape:
                if self.engine.is_ddp:
                    with self.model.no_sync():
                        loss, y_hat = self.model.forward(
                            x=sample,
                            y=label,
                            loss=self.criterion,
                            verbose=False,
                        )
                    torch.distributed.barrier()
                else:
                    loss, y_hat = self.model.forward(
                        x=sample, y=label, loss=self.criterion, verbose=False
                    )
            else:
                if self.bit16:
                    _bf16 = str(getattr(self, "amp_dtype", "float16")).lower() in ("bf16", "bfloat16")
                    _amp_dtype = torch.bfloat16 if _bf16 else torch.float16
                    with torch.amp.autocast(
                        device_type="cuda", dtype=_amp_dtype
                    ):
                        if self._two_head_on() and self.model.training:
                            y_hat, y_aux = self.model(sample, return_aux=True)
                        else:
                            y_hat, y_aux = self.model.forward(sample), None

                        loss = self.criterion(y_hat, label)
                        if y_aux is not None:
                            loss = self._apply_auxiliary_head_loss(
                                loss, y_hat, y_aux, sample, aux_target)
                        else:
                            _kd, _alpha = self._maybe_kd_loss(y_hat, sample)
                            if _kd is not None:
                                loss = (1.0 - _alpha) * loss + _alpha * _kd
                        loss = loss + self.me_weight_diversity_lambda * self.get_me_weight_diversity_penalty(loss.device)
                        if self.jdx_enabled:
                            loss = loss + self._jdx_weight_now() * self.get_jdx_penalty(loss.device)
                        refiner_delta_penalty = self.get_refiner_delta_penalty(loss.device)
                        if self.refiner_delta_lambda > 0:
                            loss = loss + self.refiner_delta_lambda * refiner_delta_penalty
                        if self.refiner_base_loss_lambda > 0 and effective_blend > 0:
                            refiner_stats_snapshot = self.snapshot_refiner_stats()
                            self.set_refiner_blend(0.0)
                            y_hat_base = self.model.forward(sample)
                            refiner_base_loss = self.criterion(y_hat_base, label)
                            loss = loss + self.refiner_base_loss_lambda * refiner_base_loss
                            del y_hat_base
                            self.set_refiner_blend(effective_blend)
                            self.restore_refiner_stats(refiner_stats_snapshot)
                    # Gradient accumulation: scale the loss so accumulated grads
                    # average (not sum). bf16 needs no GradScaler.
                    _accum = max(1, int(getattr(self, "accum_steps", 1)))
                    if _bf16:
                        (loss / _accum).backward()
                    else:
                        scaler.scale(loss / _accum).backward()
                else:
                    if self._two_head_on() and self.model.training:
                        y_hat, y_aux = self.model(sample, return_aux=True)
                    else:
                        y_hat, y_aux = self.model.forward(sample), None
                    loss = self.criterion(y_hat, label)
                    if y_aux is not None:
                        loss = self._apply_auxiliary_head_loss(
                            loss, y_hat, y_aux, sample, aux_target)
                    else:
                        _kd, _alpha = self._maybe_kd_loss(y_hat, sample)
                        if _kd is not None:
                            loss = (1.0 - _alpha) * loss + _alpha * _kd
                    loss = loss + self.me_weight_diversity_lambda * self.get_me_weight_diversity_penalty(loss.device)
                    if self.jdx_enabled:
                        loss = loss + self._jdx_weight_now() * self.get_jdx_penalty(loss.device)
                    refiner_delta_penalty = self.get_refiner_delta_penalty(loss.device)
                    if self.refiner_delta_lambda > 0:
                        loss = loss + self.refiner_delta_lambda * refiner_delta_penalty
                    if self.refiner_base_loss_lambda > 0 and effective_blend > 0:
                        refiner_stats_snapshot = self.snapshot_refiner_stats()
                        self.set_refiner_blend(0.0)
                        y_hat_base = self.model.forward(sample)
                        refiner_base_loss = self.criterion(y_hat_base, label)
                        loss = loss + self.refiner_base_loss_lambda * refiner_base_loss
                        del y_hat_base
                        self.set_refiner_blend(effective_blend)
                        self.restore_refiner_stats(refiner_stats_snapshot)
                    (loss / max(1, int(getattr(self, "accum_steps", 1)))).backward()
            if not self.optimize_inline:
                # Gradient accumulation: backward runs every micro-step (grads
                # accumulate because zero_grad only fires on a real step); the
                # optimizer/scheduler/EMA only advance every accum_steps.
                _accum = max(1, int(getattr(self, "accum_steps", 1)))
                self._accum_count = getattr(self, "_accum_count", 0) + 1
                if self._accum_count >= _accum:
                    _use_scaler = self.bit16 and str(getattr(self, "amp_dtype", "float16")).lower() not in ("bf16", "bfloat16")
                    self.zero_refiner_grads()
                    # Gradient clipping: bound the (accumulated) grad norm so an
                    # occasional outlier batch can't knock the weights off (the
                    # transient train-dice collapses). fp16 grads must be unscaled
                    # first to measure the true norm. If the norm is non-finite
                    # (NaN/Inf), SKIP the step entirely so a bad batch never lands.
                    # grad_clip <= 0 disables (old behavior).
                    _clip = float(getattr(self, "grad_clip", 0.0))
                    _skip_step = False
                    if _clip > 0:
                        if _use_scaler:
                            scaler.unscale_(self.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), _clip
                        )
                        _skip_step = not bool(torch.isfinite(total_norm))
                    if _skip_step:
                        # Drop this update; keep scaler/LR/EMA aligned with real steps.
                        if _use_scaler:
                            scaler.update()
                        self.optimizer.zero_grad()
                        self._accum_count = 0
                    else:
                        if _use_scaler:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self._accum_count = 0
                        self._ema_update()
        else:
            self.set_refiner_blend(self.refiner_blend)
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
                loss = loss + self.me_weight_diversity_lambda * self.get_me_weight_diversity_penalty(loss.device)
                refiner_delta_penalty = self.get_refiner_delta_penalty(loss.device)
        # The macro_dice metric (argmax + faster_dice over n_classes) is a
        # monitoring-only quantity -- it is NOT part of the loss when
        # loss_weight=[1,0]. At 256^3 x 104 classes it is expensive in both
        # compute and peak memory, so during training we compute it only every
        # dice_every_n_steps steps and carry the last value forward in between.
        # On the skipped steps we compute a cheap *approximate* dice on a
        # strided spatial subsample (e.g. every 4th voxel per axis ≈ 64×
        # fewer voxels) so the monitoring curve stays informative.
        # Validation always computes the full dice (it is the eval metric).
        is_train = self.model.training
        n_every = getattr(self, "dice_every_n_steps", 1) or 1
        if is_train:
            self._dice_step = getattr(self, "_dice_step", 0) + 1
        compute_dice = (
            (not is_train)
            or n_every <= 1
            or getattr(self, "_last_dice", None) is None
            or (self._dice_step % n_every == 0)
        )
        if compute_dice:
            with torch.inference_mode():
                result = torch.squeeze(torch.argmax(y_hat, 1)).long()
                labels = torch.squeeze(label)
                dice = torch.mean(
                    faster_dice(result, labels, range(self.n_classes))
                )
            self._last_dice = dice.detach()
            approx_dice = dice.detach()  # full dice IS the approx on compute steps
            if (not is_train) and self._surface_enabled() and getattr(self, "_surf_this_epoch", False):
                self._accumulate_surface(result, labels)
        else:
            dice = self._last_dice
            # Cheap approximate dice on a strided spatial subsample.
            stride = getattr(self, "dice_subsample_stride", 4) or 4
            with torch.inference_mode():
                sub_hat = y_hat[..., ::stride, ::stride, ::stride]
                sub_lbl = label[..., ::stride, ::stride, ::stride]
                sub_result = torch.squeeze(torch.argmax(sub_hat, 1)).long()
                sub_labels = torch.squeeze(sub_lbl)
                approx_dice = torch.mean(
                    faster_dice(sub_result, sub_labels, range(self.n_classes))
                ).detach()

        # Collect refiner convergence stats if available
        refiner_iters = []
        refiner_iters_rel = []
        refiner_residuals = []
        refiner_deltas = []
        refiner_alphas = []
        refiner_coeff_means = []
        refiner_coeff_maxes = []
        if getattr(self, "use_refiner", False) and hasattr(self, "model") and hasattr(self.model, "named_modules"):
            for name, module in self.model.named_modules():
                if module.__class__.__name__ in REFINER_CLASS_NAMES and hasattr(module, "step"):
                    if hasattr(module.step, "last_fwd_iters"):
                        refiner_iters.append(module.step.last_fwd_iters)
                        max_it = getattr(module, "max_iter", 30)
                        refiner_iters_rel.append(module.step.last_fwd_iters / max_it if max_it > 0 else 0.0)
                    if hasattr(module.step, "last_fwd_rel"):
                        refiner_residuals.append(module.step.last_fwd_rel)
                    if hasattr(module.step, "last_fwd_delta"):
                        refiner_deltas.append(module.step.last_fwd_delta)
                    if hasattr(module.step, "last_fwd_alpha"):
                        refiner_alphas.append(module.step.last_fwd_alpha)
                    if hasattr(module.step, "last_coeff_abs_mean"):
                        refiner_coeff_means.append(module.step.last_coeff_abs_mean)
                    if hasattr(module.step, "last_coeff_abs_max"):
                        refiner_coeff_maxes.append(module.step.last_coeff_abs_max)

        metrics_dict = {
            "loss": loss.detach(),
            "macro_dice": dice,
            "learning rate": torch.tensor(
                self.optimizer.param_groups[0]["lr"]
            ),
        }
        refiner_metrics_dict = {}
        rbp_metrics_dict = {}
        
        meter_keys = ["loss", "macro_dice", "learning rate"]
        # NOTE: two-head kd/marg are logged wandb-ONLY (below), like approx_dice,
        # so they stay OFF the tqdm bar.
        if refiner_iters:
            import math
            clean_iters = [it for it in refiner_iters if it is not None and not math.isnan(it) and not math.isinf(it)]
            clean_iters_rel = [it for it in refiner_iters_rel if it is not None and not math.isnan(it) and not math.isinf(it)]
            clean_res = [r for r in refiner_residuals if r is not None and not math.isnan(r) and not math.isinf(r)]
            clean_delta = [d for d in refiner_deltas if d is not None and not math.isnan(d) and not math.isinf(d)]
            clean_alpha = [a for a in refiner_alphas if a is not None and not math.isnan(a) and not math.isinf(a)]
            clean_coeff_mean = [c for c in refiner_coeff_means if c is not None and not math.isnan(c) and not math.isinf(c)]
            clean_coeff_max = [c for c in refiner_coeff_maxes if c is not None and not math.isnan(c) and not math.isinf(c)]
            avg_iters = sum(clean_iters) / len(clean_iters) if clean_iters else 0.0
            min_iters = min(clean_iters) if clean_iters else 0.0
            max_iters = max(clean_iters) if clean_iters else 0.0
            avg_iters_rel = sum(clean_iters_rel) / len(clean_iters_rel) if clean_iters_rel else 0.0
            avg_res = sum(clean_res) / len(clean_res) if clean_res else 0.0
            min_res = min(clean_res) if clean_res else 0.0
            max_res = max(clean_res) if clean_res else 0.0
            avg_delta = sum(clean_delta) / len(clean_delta) if clean_delta else 0.0
            avg_alpha = sum(clean_alpha) / len(clean_alpha) if clean_alpha else 0.0
            avg_coeff_mean = sum(clean_coeff_mean) / len(clean_coeff_mean) if clean_coeff_mean else 0.0
            avg_coeff_max = sum(clean_coeff_max) / len(clean_coeff_max) if clean_coeff_max else 0.0
            refiner_metrics_dict["refiner_iters_mean"] = torch.tensor(avg_iters, device=loss.device)
            refiner_metrics_dict["refiner_iters_min"] = torch.tensor(min_iters, device=loss.device)
            refiner_metrics_dict["refiner_iters_max"] = torch.tensor(max_iters, device=loss.device)
            refiner_metrics_dict["refiner_iters_relative_mean"] = torch.tensor(avg_iters_rel, device=loss.device)
            refiner_metrics_dict["refiner_residual_mean"] = torch.tensor(avg_res, device=loss.device)
            refiner_metrics_dict["refiner_residual_min"] = torch.tensor(min_res, device=loss.device)
            refiner_metrics_dict["refiner_residual_max"] = torch.tensor(max_res, device=loss.device)
            refiner_metrics_dict["refiner_avg_kernel_delta"] = torch.tensor(avg_delta, device=loss.device)
            refiner_metrics_dict["refiner_avg_alpha"] = torch.tensor(avg_alpha, device=loss.device)
            refiner_metrics_dict["refiner_coeff_abs_mean"] = torch.tensor(avg_coeff_mean, device=loss.device)
            refiner_metrics_dict["refiner_coeff_abs_max"] = torch.tensor(avg_coeff_max, device=loss.device)
            refiner_metrics_dict["refiner_delta_penalty"] = refiner_delta_penalty.detach()
            refiner_metrics_dict["refiner_scheduled_blend"] = torch.tensor(scheduled_blend, device=loss.device)
            refiner_metrics_dict["refiner_effective_blend"] = torch.tensor(effective_blend, device=loss.device)
            refiner_metrics_dict["refiner_bypassed"] = torch.tensor(refiner_bypassed, device=loss.device)
            refiner_metrics_dict["refiner_bypass_prob"] = torch.tensor(self.refiner_bypass_prob, device=loss.device)
            refiner_metrics_dict["refiner_base_loss"] = refiner_base_loss.detach()
            refiner_metrics_dict["refiner_base_loss_weighted"] = (self.refiner_base_loss_lambda * refiner_base_loss).detach()
            meter_keys += ["refiner_iters_mean", "refiner_iters_min", "refiner_iters_max", "refiner_iters_relative_mean", "refiner_residual_mean", "refiner_residual_min", "refiner_residual_max", "refiner_avg_kernel_delta", "refiner_avg_alpha", "refiner_coeff_abs_mean", "refiner_coeff_abs_max", "refiner_delta_penalty", "refiner_scheduled_blend", "refiner_effective_blend", "refiner_bypassed", "refiner_bypass_prob", "refiner_base_loss", "refiner_base_loss_weighted"]

        model_for_metrics = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_for_metrics, "step") and hasattr(model_for_metrics.step, "last_fwd_iters"):
            max_iter = getattr(model_for_metrics, "rbp_max_iter", 1)
            rbp_metrics_dict["rbp_iters"] = torch.tensor(model_for_metrics.step.last_fwd_iters, device=loss.device)
            rbp_metrics_dict["rbp_iters_relative"] = torch.tensor(model_for_metrics.step.last_fwd_iters / max(max_iter, 1), device=loss.device)
            rbp_metrics_dict["rbp_residual"] = torch.tensor(getattr(model_for_metrics.step, "last_fwd_rel", 0.0), device=loss.device)
            rbp_metrics_dict["rbp_state_delta"] = torch.tensor(getattr(model_for_metrics.step, "last_fwd_delta", 0.0), device=loss.device)
            meter_keys += ["rbp_iters", "rbp_iters_relative", "rbp_residual", "rbp_state_delta"]

        self.batch_metrics.update(metrics_dict)

        # approx_dice -> wandb only (NOT batch_metrics), so it never shows in tqdm.
        # Catalyst's WandbLogger logs all batch metrics with an explicit
        # step=runner.sample_step. A wandb.log() with NO step resolves to wandb's
        # internal counter, conflicts with those explicit steps, and gets dropped
        # -> it showed in neither tqdm nor wandb. Log at the SAME sample_step (and
        # with Catalyst's "{key}_batch/{loader}" naming) so it lands on the same
        # row next to macro_dice_batch/<loader>.
        try:
            import wandb
            if wandb.run is not None:
                loader_key = getattr(self, "loader_key", "train")
                step = getattr(self, "sample_step",
                               getattr(self, "global_sample_step", None))
                if step is None:
                    step = wandb.run.step
                wandb.log(
                    {f"approx_dice_batch/{loader_key}": float(approx_dice)},
                    step=step,
                    commit=False,
                )
        except Exception:
            pass

        # Per-term loss breakdown -> wandb ONLY (never tqdm), same naming/step
        # convention as approx_dice. CEDiceLoss stashes each term of its LAST
        # forward in `_terms` when model.loss_log_terms is on; nothing here
        # feeds the training math. This is what makes an aux-loss arm readable
        # as it unfolds: if lossterm_group_tversky falls while lossterm_ce
        # climbs, the aux term is winning against the data term and the weight
        # is too high. Values are detached tensors, so float() is the only sync.
        _terms = getattr(getattr(self, "criterion", None), "_terms", None)
        if _terms:
            try:
                import wandb
                if wandb.run is not None:
                    loader_key = getattr(self, "loader_key", "train")
                    step = getattr(self, "sample_step",
                                   getattr(self, "global_sample_step", None))
                    if step is None:
                        step = wandb.run.step
                    wandb.log(
                        {f"lossterm_{k}_batch/{loader_key}": float(v)
                         for k, v in _terms.items()},
                        step=step,
                        commit=False,
                    )
            except Exception:
                pass

        # two-head KD / marginal-consistency -> wandb only (never in tqdm), same
        # naming/step convention as approx_dice so they land as kd_batch/<loader>
        # and marg_batch/<loader>.
        if self._two_head_on() and self.model.training:
            try:
                import wandb
                if wandb.run is not None:
                    loader_key = getattr(self, "loader_key", "train")
                    step = getattr(self, "sample_step",
                                   getattr(self, "global_sample_step", None))
                    if step is None:
                        step = wandb.run.step
                    payload = {}
                    if getattr(self, "_last_kd", None) is not None:
                        payload[f"kd_batch/{loader_key}"] = float(self._last_kd)
                    if getattr(self, "_last_marg", None) is not None:
                        payload[f"marg_batch/{loader_key}"] = float(self._last_marg)
                    if getattr(self, "_last_aux", None) is not None:
                        payload[f"missing_gdl_batch/{loader_key}"] = float(
                            self._last_aux)
                    if getattr(self, "_last_jdx", None) is not None:
                        payload[f"jdx_batch/{loader_key}"] = float(self._last_jdx)
                    if payload:
                        wandb.log(payload, step=step, commit=False)
            except Exception:
                pass

        if refiner_metrics_dict:
            try:
                import wandb
                if wandb.run is not None:
                    loader_key = getattr(self, "loader_key", "batch")
                    wandb.log(
                        {
                            f"refiner/{loader_key}/batch/{key.removeprefix('refiner_')}": value.detach().item()
                            for key, value in refiner_metrics_dict.items()
                        },
                        commit=False,
                    )
            except Exception:
                pass
        if rbp_metrics_dict:
            try:
                import wandb
                if wandb.run is not None:
                    loader_key = getattr(self, "loader_key", "batch")
                    wandb.log(
                        {
                            f"rbp/{loader_key}/batch/{key.removeprefix('rbp_')}": value.detach().item()
                            for key, value in rbp_metrics_dict.items()
                        },
                        commit=False,
                    )
            except Exception:
                pass

        for key in meter_keys:
            metric_value = metrics_dict.get(key, refiner_metrics_dict.get(key))
            if metric_value is None:
                metric_value = rbp_metrics_dict.get(key)
            self.meters[key].update(
                metric_value.item(), self.num_volumes
            )

        del sample
        del label
        del y_hat
        if compute_dice:
            del result
            del labels
        del loss


class ClientCreator:
    def __init__(self, mongohost, volume_shape=[256] * 3, crop_tensor=False):
        self.mongohost = mongohost
        self.volume_shape = volume_shape
        self.subvolume_shape = None
        self.dbname = None
        self.collection = None
        self.num_subcubes = None
        self.crop_tensor = crop_tensor

    def set_shape(self, shape):
        self.subvolume_shape = shape
        self.coord_generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )

    def set_collection(self, collection):
        self.collection = collection

    def set_database(self, database):
        self.dbname = database

    def set_num_subcubes(self, num_subcubes):
        self.num_subcubes = num_subcubes

    def create_client(self, x):
        return create_client(
            x,
            dbname=self.dbname,
            colname=self.collection,
            mongohost=self.mongohost,
        )

    def create_v_client(self, x):
        return create_client(
            x,
            dbname="HPC1200z",
            colname="HCP",
            mongohost=self.mongohost,
        )

    def mycollate(self, x):
        return collate_subcubes(
            x,
            self.coord_generator,
            samples=self.num_subcubes,
        )

    def mycollate_full(self, x):
        return crop_tensor(*mcollate(x)) if self.crop_tensor else mcollate(x)

    def mytransform(self, x):
        try:
            return mtransform(x)
        except Exception as e:
            import sys
            print(f"\n[ERROR] mytransform failed!", file=sys.stderr, flush=True)
            print(f"[ERROR] type(x): {type(x)}", file=sys.stderr, flush=True)
            if isinstance(x, (bytes, bytearray)):
                print(f"[ERROR] len(x): {len(x)} bytes", file=sys.stderr, flush=True)
                if len(x) > 0:
                    print(f"[ERROR] First 50 bytes: {x[:50]}", file=sys.stderr, flush=True)
                    print(f"[ERROR] Last 50 bytes: {x[-50:]}", file=sys.stderr, flush=True)
            else:
                print(f"[ERROR] Value of x: {x}", file=sys.stderr, flush=True)
            raise e


def assert_equal_length(*args):
    assert all(
        len(arg) == len(args[0]) for arg in args
    ), "Not all parameter lists have the same length!"



@hydra.main(config_path="conf", config_name="vanilla_3class_gn_11chan32.16.1_exp01", version_base=None)
def main(cfg: DictConfig):
    # Loading common parameters
    # Model parameters
    volume_shape = cfg.model.volume_shape
    n_classes = cfg.model.n_classes
    config_file = cfg.model.config_file
    optimize_inline = cfg.model.optimize_inline
    model_channels = cfg.model.model_channels
    model_label = cfg.model.model_label
    use_groupnorm = cfg.model.use_groupnorm
    use_affine = cfg.model.get("use_affine", False)
    use_refiner = cfg.model.get("use_refiner", False)
    use_se = cfg.model.get("use_se", False)
    use_checkpoint = cfg.model.get("use_checkpoint", True)
    # convergence-speed + EMA + spatial-AE knobs (all default to old behavior)
    accum_steps = int(cfg.experiment.get("accum_steps", 1))
    amp_dtype = str((cfg.get("perf", {}) or {}).get("amp_dtype", "float16"))
    use_ema = bool(cfg.model.get("use_ema", False))
    ema_decay = float(cfg.model.get("ema_decay", 0.999))
    use_spatial_ae = bool(cfg.model.get("use_spatial_ae", False))
    spatial_ae_mult = int(cfg.model.get("spatial_ae_bottleneck_mult", 2))
    spatial_ae_down = str(cfg.model.get("spatial_ae_downsample", "avgpool"))
    spatial_ae_up = str(cfg.model.get("spatial_ae_upsample", "transposed"))
    weight_decay = float(cfg.experiment.get("weight_decay", 0.0))
    grad_clip = float(cfg.experiment.get("grad_clip", 0.0))
    sched_pct_start = float(cfg.experiment.get("pct_start", 0.1))
    sched_div_factor = float(cfg.experiment.get("div_factor", 100.0))
    sched_final_div = float(cfg.experiment.get("final_div_factor", 1e4))
    se_cfg = cfg.model.get("se", {})
    se_kwargs = OmegaConf.to_container(se_cfg, resolve=True) if se_cfg else {}
    me_cfg = cfg.model.get("me", {})
    me_kwargs = OmegaConf.to_container(me_cfg, resolve=True) if me_cfg else {}
    refiner_cfg = cfg.model.get("refiner", {})
    refiner_kwargs = OmegaConf.to_container(refiner_cfg, resolve=True) if refiner_cfg else {}
    refiner_delta_lambda = cfg.model.get("refiner_delta_lambda", 1e-3)
    refiner_delta_target = cfg.model.get("refiner_delta_target", 0.15)
    refiner_freeze_epochs_after_transition = cfg.model.get("refiner_freeze_epochs_after_transition", 1)
    refiner_bypass_prob = cfg.model.get("refiner_bypass_prob", 0.0)
    refiner_base_loss_lambda = cfg.model.get("refiner_base_loss_lambda", 0.0)
    me_weight_diversity_lambda = cfg.model.get("me_weight_diversity_lambda", 0.0)
    jdx_cfg = cfg.model.get("jdx", {})
    jdx_kwargs = OmegaConf.to_container(jdx_cfg, resolve=True) if jdx_cfg else {}
    label_smoothing = cfg.model.get("label_smoothing", 0.01)
    dice_generalized = cfg.model.get("dice_generalized", False)
    loss_fused = cfg.model.get("loss_fused", False)
    boundary_weight = float(cfg.model.get("boundary_weight", 0.0))
    boundary_radius = int(cfg.model.get("boundary_radius", 8))
    boundary_include_bg = bool(cfg.model.get("boundary_include_bg", False))
    boundary_downsample = int(cfg.model.get("boundary_downsample", 2))
    cldice_weight = float(cfg.model.get("cldice_weight", 0.0))
    cldice_iters = int(cfg.model.get("cldice_iters", 5))
    cldice_downsample = int(cfg.model.get("cldice_downsample", 1))
    cldice_include_bg = bool(cfg.model.get("cldice_include_bg", False))
    _clc = cfg.model.get("cldice_classes", None)
    cldice_classes = list(OmegaConf.to_container(_clc, resolve=True)) if _clc is not None else None
    tversky_weight = float(cfg.model.get("tversky_weight", 0.0))
    tversky_alpha = float(cfg.model.get("tversky_alpha", 0.7))
    tversky_beta = float(cfg.model.get("tversky_beta", 0.3))
    _tvc = cfg.model.get("tversky_classes", None)
    tversky_classes = list(OmegaConf.to_container(_tvc, resolve=True)) if _tvc is not None else None
    # --- marginalized (group-space) aux terms -------------------------------
    # model.group_lut takes either a NAME from dice.GROUP_LUTS ("lut104_to_18")
    # or an explicit list of length n_classes. Absent => terms stay off and the
    # criterion is built exactly as before.
    _glut = cfg.model.get("group_lut", None)
    if _glut is None:
        group_lut = None
    elif isinstance(_glut, str):
        if _glut not in GROUP_LUTS:
            raise SystemExit(f"model.group_lut '{_glut}' unknown; "
                             f"known: {sorted(GROUP_LUTS)}")
        group_lut = list(GROUP_LUTS[_glut])
    else:
        group_lut = list(OmegaConf.to_container(_glut, resolve=True))
    if group_lut is not None and len(group_lut) != int(cfg.model.n_classes):
        raise SystemExit(
            f"model.group_lut has {len(group_lut)} entries but "
            f"model.n_classes is {int(cfg.model.n_classes)}"
        )
    group_n_classes = cfg.model.get("group_n_classes", None)
    group_n_classes = int(group_n_classes) if group_n_classes is not None else None
    group_cldice_weight = float(cfg.model.get("group_cldice_weight", 0.0))
    group_cldice_iters = int(cfg.model.get("group_cldice_iters", 5))
    group_cldice_downsample = int(cfg.model.get("group_cldice_downsample", 1))
    group_cldice_include_bg = bool(cfg.model.get("group_cldice_include_bg", False))
    _gclc = cfg.model.get("group_cldice_classes", None)
    group_cldice_classes = (list(OmegaConf.to_container(_gclc, resolve=True))
                            if _gclc is not None else None)
    group_tversky_weight = float(cfg.model.get("group_tversky_weight", 0.0))
    group_tversky_alpha = float(cfg.model.get("group_tversky_alpha", 0.6))
    group_tversky_beta = float(cfg.model.get("group_tversky_beta", 0.4))
    _gtvc = cfg.model.get("group_tversky_classes", None)
    group_tversky_classes = (list(OmegaConf.to_container(_gtvc, resolve=True))
                             if _gtvc is not None else None)
    loss_log_terms = bool(cfg.model.get("loss_log_terms", False))
    _cw = cfg.model.get("ce_class_weight_overrides", None)
    ce_class_weight_overrides = OmegaConf.to_container(_cw, resolve=True) if _cw is not None else {}
    _vc = cfg.get("validation", None)
    valid_cfg = OmegaConf.to_container(_vc, resolve=True) if _vc is not None else {}
    model_path = cfg.paths.model if cfg.paths.loadcheckpoint else ""
    logdir = cfg.paths.logdir
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    # MongoDB parameters
    validation_percent = cfg.mongo.validation_percent

    wandb_project = cfg.wandb.project

    bit16 = cfg.bit16

    # DataLoader knobs (config-driven; defaults preserve previous hardcoded 4/4/2)
    dl_cfg = cfg.get("dataloader", {}) or {}
    num_workers = int(dl_cfg.get("num_workers", 4))
    persistent_workers = bool(dl_cfg.get("persistent_workers", False))
    prefetch_factor = int(dl_cfg.get("prefetch_factor", 4))
    valid_prefetch_factor = int(dl_cfg.get("valid_prefetch_factor", 2))

    # Training-only macro_dice metric cadence (validation always computes it).
    metrics_cfg = cfg.get("metrics", {}) or {}
    dice_every_n_steps = int(metrics_cfg.get("dice_every_n_steps", 1))

    # Per-step DDP device sync; default True = historical behavior.
    perf_cfg = cfg.get("perf", {}) or {}
    ddp_batch_sync = bool(perf_cfg.get("ddp_batch_sync", True))
    dice_subsample_stride = int(metrics_cfg.get("dice_subsample_stride", 4))

    client_creator = ClientCreator(
        db_host, crop_tensor=cfg.client_creator.crop_tensor
    )

    # Specify curriculum parameters
    # Set up the environment for eval
    context = {"maxreps": cfg.experiment.maxreps}

    # Evaluate the Python code from the YAML config
    cubesizes = eval(cfg.experiment.cubesizes_code, globals(), context)
    numcubes = eval(cfg.experiment.numcubes_code, globals(), context)
    numvolumes = eval(cfg.experiment.numvolumes_code, globals(), context)
    weights = eval(cfg.experiment.weights_code, globals(), context)
    databases = eval(cfg.experiment.databases_code, globals(), context)
    collections = eval(cfg.experiment.collections_code, globals(), context)
    dbfields = eval(cfg.experiment.dbfields_code, globals(), context)
    epochs = eval(cfg.experiment.epochs_code, globals(), context)
    prefetches = eval(cfg.experiment.prefetches_code, globals(), context)
    attenuates = eval(cfg.experiment.attenuates_code, globals(), context)
    refiner_blends = eval(cfg.experiment.get("refiner_blends_code", "[1.0] * maxreps"), globals(), context)

    assert_equal_length(
        cubesizes,
        numcubes,
        numvolumes,
        weights,
        databases,
        collections,
        epochs,
        prefetches,
        attenuates,
        refiner_blends,
    )

    start_experiment = 0
    for experiment in range(len(cubesizes)):
        subvolume_shape = [cubesizes[experiment]] * 3
        onecycle_lr = rmsprop_lr = (
            attenuates[experiment] ** experiment
            * 8
            * cfg.experiment.lr_scale
            * numcubes[experiment]
            * numvolumes[experiment]
            / 256
        )
        # Distinguishing tag so sibling A/B runs aren't all named identically in
        # wandb. Default = logdir basename (ctrl / boundary / cldice / ceweight),
        # override with wandb.run_tag in the yaml.
        _run_tag = (cfg.wandb.get("run_tag", None)
                    or os.path.basename(os.path.normpath(logdir)))
        wandb_experiment = (
            f"{_run_tag} | "
            + f"{start_experiment + experiment:02} cube "
            + str(subvolume_shape[0])
            + " "
            + collections[experiment]
            + model_label
        )

        # Set database parameters
        client_creator.set_database(databases[experiment])
        client_creator.set_collection(collections[experiment])
        client_creator.set_num_subcubes(numcubes[experiment])
        client_creator.set_shape(subvolume_shape)

        hparam_config_file = cfg.model.config_file[0] if isinstance(cfg.model.config_file, ListConfig) else cfg.model.config_file
        with open(hparam_config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            hparams = {"model_arch": config_dict, **OmegaConf.to_container(cfg)}

        runner = CustomRunner(
            logdir=logdir,
            wandb_project=wandb_project,
            wandb_experiment=wandb_experiment,
            model_path=model_path,
            n_channels=model_channels,
            n_classes=n_classes,
            modelconfig=config_file,
            n_epochs=epochs[experiment],
            optimize_inline=optimize_inline,
            validation_percent=validation_percent,
            onecycle_lr=onecycle_lr,
            rmsprop_lr=rmsprop_lr,
            num_subcubes=numcubes[experiment],
            num_volumes=numvolumes[experiment],
            groupnorm=use_groupnorm,
            affine=use_affine,
            client_creator=client_creator,
            off_brain_weight=weights[experiment],
            prefetches=prefetches[experiment],
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            valid_prefetch_factor=valid_prefetch_factor,
            dice_every_n_steps=dice_every_n_steps,
            dice_subsample_stride=dice_subsample_stride,
            ddp_batch_sync=ddp_batch_sync,
            indexid=cfg.mongo.index_id,
            db_collection=collections[experiment],
            db_name=databases[experiment],
            db_fields=dbfields[experiment],
            subvolume_shape=subvolume_shape,
            lowprecision=bit16,
            lossweight = [w / sum(cfg.model.loss_weight) for w in cfg.model.loss_weight] if sum(cfg.model.loss_weight) != 0 else ValueError("The sum of loss weights cannot be zero."),
            label_smoothing=label_smoothing,
            dice_generalized=dice_generalized,
            loss_fused=loss_fused,
            boundary_weight=boundary_weight,
            boundary_radius=boundary_radius,
            boundary_include_bg=boundary_include_bg,
            boundary_downsample=boundary_downsample,
            cldice_weight=cldice_weight,
            cldice_iters=cldice_iters,
            cldice_downsample=cldice_downsample,
            cldice_include_bg=cldice_include_bg,
            cldice_classes=cldice_classes,
            tversky_weight=tversky_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_classes=tversky_classes,
            group_lut=group_lut,
            group_n_classes=group_n_classes,
            group_cldice_weight=group_cldice_weight,
            group_cldice_iters=group_cldice_iters,
            group_cldice_downsample=group_cldice_downsample,
            group_cldice_include_bg=group_cldice_include_bg,
            group_cldice_classes=group_cldice_classes,
            group_tversky_weight=group_tversky_weight,
            group_tversky_alpha=group_tversky_alpha,
            group_tversky_beta=group_tversky_beta,
            group_tversky_classes=group_tversky_classes,
            loss_log_terms=loss_log_terms,
            ce_class_weight_overrides=ce_class_weight_overrides,
            valid_cfg=valid_cfg,
            meshnetme=cfg.model.use_me,
            db_host=db_host,
            wandb_team=cfg.wandb.team,
            maxshape=cfg.model.maxshape,
            hparams=hparams,
            use_refiner=use_refiner,
            refiner_kwargs=refiner_kwargs,
            refiner_delta_lambda=refiner_delta_lambda,
            refiner_delta_target=refiner_delta_target,
            refiner_freeze_epochs=(
                refiner_freeze_epochs_after_transition
                if use_refiner and experiment > 0 and cubesizes[experiment] != cubesizes[experiment - 1]
                else 0
            ),
            refiner_blend=refiner_blends[experiment],
            refiner_bypass_prob=refiner_bypass_prob,
            refiner_base_loss_lambda=refiner_base_loss_lambda,
            me_kwargs=me_kwargs,
            me_weight_diversity_lambda=me_weight_diversity_lambda,
            use_se=use_se,
            se_kwargs=se_kwargs,
            use_checkpoint=use_checkpoint,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            accum_steps=accum_steps,
            amp_dtype=amp_dtype,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_spatial_ae=use_spatial_ae,
            spatial_ae_mult=spatial_ae_mult,
            spatial_ae_down=spatial_ae_down,
            spatial_ae_up=spatial_ae_up,
            sched_pct_start=sched_pct_start,
            sched_div_factor=sched_div_factor,
            sched_final_div=sched_final_div,
            jdx_kwargs=jdx_kwargs,
        )
        # Knowledge distillation (optional): 18-class/16ch teacher -> student.
        # Attached to the instance (not threaded through __init__) and built
        # lazily on the first training batch, once the engine device is known.
        _distill_cfg = cfg.model.get("distill", None)
        runner.distill_cfg = (
            OmegaConf.to_container(_distill_cfg, resolve=True)
            if _distill_cfg is not None else None
        )
        _two_head_cfg = cfg.model.get("two_head", None)
        runner.two_head_cfg = (
            OmegaConf.to_container(_two_head_cfg, resolve=True)
            if _two_head_cfg is not None else None
        )
        # Periodic real-data (e.g. MindfulTensors/MRN) boundary-metric eval,
        # attached the same way as distill_cfg/two_head_cfg above so it survives
        # DDP mp.spawn (instance attrs set before runner.run() are pickled into
        # each rank; class-level state set elsewhere in main() is not). No-op
        # unless real_eval.enabled=True AND the runner's get_callbacks() wires it
        # up -- currently only FastRunner (curriculum_training_fast.py) does.
        _real_eval_cfg = cfg.get("real_eval", None)
        runner.real_eval_cfg = (
            OmegaConf.to_container(_real_eval_cfg, resolve=True)
            if _real_eval_cfg is not None else {}
        )
        try:
            from hydra.core.hydra_config import HydraConfig
            runner.config_name = HydraConfig.get().job.config_name
        except Exception:
            runner.config_name = None
        runner.run()

        shutil.copy(
            logdir + "/model.last.pth",
            logdir
            + "/model.last."
            + str(subvolume_shape[0])
            + f".run{experiment:02}.curriculum.pth",
        )

        model_path = logdir + "model.last.pth"

        # Release this rep's model/optimizer/runner before building the next one,
        # so GPU allocations don't accumulate across curriculum reps.
        del runner
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
