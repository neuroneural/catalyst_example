import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import os
import random
import shutil
from packaging import version
import yaml
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dice import faster_dice, DiceLoss
from meshnet import enMesh_checkpoint, enMesh, enMesh_checkpoint_SE, enMesh_SE
from meshnet_gn import enMesh_checkpoint as enMesh_checkpoint_gn
from meshnetme import MeshNetME_checkpoint
from refiner import enDynamicMesh_checkpoint, enDynamicMesh
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

import sys
import time
from pymongo.errors import OperationFailure

# Monkey-patching MongoDataset and MongoheadDataset to be robust against transient socket/EOF errors
def make_safe_getitem(original_getitem, class_name):
    def safe_getitem(self, batch, *args, **kwargs):
        max_retries = 10
        last_exception = None
        for attempt in range(max_retries):
            try:
                return original_getitem(self, batch, *args, **kwargs)
            except (EOFError, OperationFailure, RuntimeError, Exception) as e:
                last_exception = e
                print(
                    f"\n[Warning] {class_name}.__getitem__ failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Closing MongoClient to force reconnection...",
                    file=sys.stderr,
                    flush=True
                )
                
                # Close the MongoClient connection
                try:
                    if hasattr(self, "collection") and isinstance(self.collection, dict) and "bin" in self.collection:
                        client = self.collection["bin"].database.client
                        client.close()
                        print(f"[Info] MongoClient closed successfully.", file=sys.stderr, flush=True)
                except Exception as close_err:
                    print(f"[Warning] Failed to close MongoClient: {close_err}", file=sys.stderr, flush=True)
                
                time.sleep(1)
        
        raise last_exception
    return safe_getitem

# Apply monkey-patches to resolve fork-safety connection issues
MongoDataset.__getitem__ = make_safe_getitem(MongoDataset.__getitem__, "MongoDataset")
MongoheadDataset.__getitem__ = lambda self, batch, *args, **kwargs: MongoDataset.__getitem__(self, batch, *args, **kwargs)


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
        prefetches=8,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        lowprecision=False,
        meshnetme=False,
        lossweight=[1, 0],
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
        self.loss_weight = lossweight
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
        self._local_epoch_index = 0
        self._refiner_frozen = False

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

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            return dl.DistributedDataParallelEngine(
                # mixed_precision="fp16",
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

        tsampler = (
            DBBatchSampler(tdataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(tdataset, batch_size=self.num_volumes)
        )

        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                sampler=tsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                prefetch_factor=4,
                num_workers=4,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        vdataset = MongoDataset(
            range(32),
            self.funcs["mytransform"],
            None,
            self.db_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )

        vsampler = (
            DBBatchSampler(vdataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(
                vdataset, batch_size=self.num_volumes, seed=SEED
            )
        )

        vdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                vdataset,
                sampler=vsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                prefetch_factor=2,
                num_workers=4,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        return {"train": tdataloader, "valid": vdataloader}

    def get_model(self):
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
                    else {}
                )
                model = modelClass(
                    in_channels=1,
                    n_classes=self.n_classes,
                    channels=self.n_channels,
                    config_file=self.config_file,
                    **extra_kwargs,
                )
        return model

    def get_criterion(self):
        class_weight = torch.FloatTensor(
            [self.off_brain_weight] + [1.0] * (self.n_classes - 1)
        ).to(self.engine.device)
        ce_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=0.01
        )
        dice_criterion = DiceLoss()

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

    def get_optimizer(self, model):
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.onecycle_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=100,
            pct_start=0.1,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )
        return scheduler

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
            "checkpoint": dl.CheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        keys = ["loss", "macro_dice", "learning rate"]
        if getattr(self, "use_refiner", False):
            keys += ["refiner_iters_mean", "refiner_iters_min", "refiner_iters_max", "refiner_iters_relative_mean", "refiner_residual_mean", "refiner_residual_min", "refiner_residual_max", "refiner_avg_kernel_delta", "refiner_avg_alpha", "refiner_coeff_abs_mean", "refiner_coeff_abs_max", "refiner_delta_penalty", "refiner_scheduled_blend", "refiner_effective_blend", "refiner_bypassed", "refiner_bypass_prob", "refiner_base_loss", "refiner_base_loss_weighted"]
        if self.is_rbp_model():
            keys += ["rbp_iters", "rbp_iters_relative", "rbp_residual", "rbp_state_delta"]
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in keys
        }

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
        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):
        # Add synchronization before processing
        if self.engine.is_ddp:
            torch.cuda.synchronize()
        
        sample, label = batch
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
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        y_hat = self.model.forward(sample)
                        # print("y_hat.shape: ", y_hat.shape)
                        # print("label.shape: ", label.shape)
                        # stop

                        loss = self.criterion(y_hat, label)
                        loss = loss + self.me_weight_diversity_lambda * self.get_me_weight_diversity_penalty(loss.device)
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
                    scaler.scale(loss).backward()
                else:
                    y_hat = self.model.forward(sample)
                    loss = self.criterion(y_hat, label)
                    loss = loss + self.me_weight_diversity_lambda * self.get_me_weight_diversity_penalty(loss.device)
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
                    loss.backward()
            if not self.optimize_inline:
                if self.bit16:
                    self.zero_refiner_grads()
                    scaler.step(self.optimizer)
                    self.scheduler.step()
                    scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.zero_refiner_grads()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
        else:
            self.set_refiner_blend(self.refiner_blend)
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
                loss = loss + self.me_weight_diversity_lambda * self.get_me_weight_diversity_penalty(loss.device)
                refiner_delta_penalty = self.get_refiner_delta_penalty(loss.device)
        with torch.inference_mode():
            result = torch.squeeze(torch.argmax(y_hat, 1)).long()
            labels = torch.squeeze(label)
            dice = torch.mean(
                faster_dice(result, labels, range(self.n_classes))
            )

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
    use_refiner = cfg.model.get("use_refiner", False)
    use_se = cfg.model.get("use_se", False)
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
    model_path = cfg.paths.model if cfg.paths.loadcheckpoint else ""
    logdir = cfg.paths.logdir
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    # MongoDB parameters
    validation_percent = cfg.mongo.validation_percent

    wandb_project = cfg.wandb.project

    bit16 = cfg.bit16

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
        wandb_experiment = (
            f"{start_experiment + experiment:02} cube "
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
            client_creator=client_creator,
            off_brain_weight=weights[experiment],
            prefetches=prefetches[experiment],
            indexid=cfg.mongo.index_id,
            db_collection=collections[experiment],
            db_name=databases[experiment],
            db_fields=dbfields[experiment],
            subvolume_shape=subvolume_shape,
            lowprecision=bit16,
            lossweight = [w / sum(cfg.model.loss_weight) for w in cfg.model.loss_weight] if sum(cfg.model.loss_weight) != 0 else ValueError("The sum of loss weights cannot be zero."),
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
        )
        runner.run()

        shutil.copy(
            logdir + "/model.last.pth",
            logdir
            + "/model.last."
            + str(subvolume_shape[0])
            + f".run{experiment:02}.curriculum.pth",
        )

        model_path = logdir + "model.last.pth"


if __name__ == "__main__":
    main()
