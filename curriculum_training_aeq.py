"""Training entrypoint for the Adaptive Equilibrium MeshNet (Variant M).

Layers on top of curriculum_training_fast.py exactly the way that file layers
on curriculum_training.py, so the whole mongo/wirehead + Catalyst + wandb + DDP
stack is reused unchanged. What this adds:

  * get_model()      -> AEQMeshNet (aeq_meshnet.py) built from cfg.model.aeq
  * get_criterion()  -> wrapper adding L_aux (Sec. 7.3) + the Hutchinson
                        Jacobian penalty to the repo's CE/Dice criterion
  * phase schedule   -> Sec. 7.1 warm-up: A unrolled -> B solver+softplus
                        -> C solver+ELU -> D site gating
  * empirical gate   -> Sec. 7.2 random-freeze-mask convergence check; site
                        gating is NOT enabled until it passes
  * aeq/* wandb logging (NFE, residuals, backward iters, m(y) spread,
                        active fraction, row sums)

Run (18 classes, 16 channels):

    source ~/venv/torch/bin/activate
    python curriculum_training_aeq.py --config-dir=conf --config-name=aeq_siam18_16ch

torch.compile and channels_last are force-disabled (env vars below reach the
mp.spawn DDP ranks): Dynamo cannot trace the data-dependent solver loop or the
custom autograd.Function, and would silently fall back to eager anyway.
"""

import json
import os
import sys

# BEFORE importing curriculum_training_fast: these env defaults are what the
# mp.spawn DDP ranks read (cfg.perf never reaches them -- see the fast file).
os.environ.setdefault("FAST_COMPILE", "0")
os.environ.setdefault("FAST_COMPILE_LOSS", "0")
os.environ.setdefault("FAST_CHANNELS_LAST", "0")

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

import curriculum_training as base
import curriculum_training_fast as fast
from aeq_meshnet import AEQMeshNet


def _load_aeq_env():
    try:
        return json.loads(os.environ.get("AEQ_CFG_JSON", "{}"))
    except Exception:
        return {}


class AEQCriterion(torch.nn.Module):
    """Wraps the repo criterion; adds the AEQ extra losses stashed on the
    module during forward. Plugs into base.handle_batch untouched, because
    there the loss is simply `self.criterion(y_hat, label)`."""

    def __init__(self, base_criterion, runner, lambda_aux, gamma_jac):
        super().__init__()
        self.base = base_criterion
        self._runner = [runner]          # list -> not registered as submodule
        self.lambda_aux = float(lambda_aux)
        self.gamma_jac = float(gamma_jac)

    def _module(self):
        m = self._runner[0].model
        m = getattr(m, "module", m)      # unwrap DDP
        return m

    def forward(self, y_hat, label):
        loss = self.base(y_hat, label)
        m = self._module()
        if not isinstance(m, AEQMeshNet):
            return loss
        aux_logits, jac = m.pop_extra_losses()
        if aux_logits is not None and self.lambda_aux > 0:
            # L_aux target: per-class volume fraction of the label (Sec. 7.3).
            with torch.no_grad():
                n = aux_logits.shape[1]
                flat = label.reshape(label.shape[0], -1)
                p = torch.stack([(flat == c).float().mean(dim=1)
                                 for c in range(n)], dim=1)
            aux = -(p * F.log_softmax(aux_logits.float(), dim=1)).sum(dim=1).mean()
            loss = loss + self.lambda_aux * aux
            m.stats["aux_loss"] = float(aux.detach())
        if jac is not None and self.gamma_jac > 0:
            loss = loss + self.gamma_jac * jac
            m.stats["jac_penalty"] = float(jac.detach())
        return loss


class AEQRunner(fast.FastRunner):
    # Seeded from env so mp.spawn DDP ranks (which re-import this module and
    # never run main()) see the same config as the launcher.
    aeq_cfg = _load_aeq_env()

    # ------------------------------------------------------------- plumbing
    def _aeq_module(self):
        m = getattr(self, "model", None)
        if m is None:
            return None
        m = getattr(m, "module", m)
        return m if isinstance(m, AEQMeshNet) else None

    def get_model(self):
        if not self.aeq_cfg.get("enabled", True):
            return super().get_model()
        model = AEQMeshNet(
            in_channels=1,
            n_classes=self.n_classes,
            channels=self.n_channels,
            aeq=self.aeq_cfg,
        )
        model.use_checkpoint = False   # harmless attr; our solver manages memory
        try:
            model._register_load_state_dict_pre_hook(fast._strip_orig_mod_prefix)
        except Exception:
            pass
        n = sum(p.numel() for p in model.parameters())
        print(f"[aeq] AEQMeshNet C={model.C} C_y={model.C_y} "
              f"coupling={model.coupling} gate_position={model.gate_position} "
              f"dilations={model.dilations} params={n}",
              file=sys.stderr, flush=True)
        return model

    # ------------------------------------------------- validation on/off
    def _validation_off(self):
        """validation.enabled=false -> skip the valid loader entirely.
        Cached on first call because get_loaders() below MUTATES valid_cfg,
        and Catalyst may call get_callbacks() either side of it."""
        if not hasattr(self, "_valid_off_cached"):
            self._valid_off_cached = not bool(
                (self.valid_cfg or {}).get("enabled", True))
        return self._valid_off_cached

    def get_loaders(self):
        if not self._validation_off():
            return super().get_loaders()
        # Blank valid_cfg first so the base builder takes its cheap legacy
        # branch instead of querying the real-data Mongo host for ids (that
        # query would still run -- and could fail -- for a loader we discard).
        self.valid_cfg = {}
        loaders = super().get_loaders()
        loaders.pop("valid", None)
        print("[aeq] validation DISABLED (validation.enabled=false): no valid "
              "loader; checkpoints key on TRAIN macro_dice",
              file=sys.stderr, flush=True)
        return loaders

    def get_callbacks(self):
        cbs = super().get_callbacks()
        if not self._validation_off():
            return cbs
        # save_best would otherwise watch a loader that never runs.
        params = {
            "save_best": True,
            "metric_key": "macro_dice",
            "loader_key": "train",
            "minimize": False,
        }
        if self.model_path:
            params["resume_model"] = self.model_path
        cbs["checkpoint"] = base.CompileSafeCheckpointCallback(
            self._logdir, **params)
        return cbs

    def get_criterion(self):
        crit = super().get_criterion()
        if not self.aeq_cfg.get("enabled", True):
            return crit
        loss_cfg = self.aeq_cfg.get("loss", {}) or {}
        return AEQCriterion(
            crit, self,
            lambda_aux=loss_cfg.get("lambda_aux", 0.1),
            gamma_jac=loss_cfg.get("gamma_jac", 0.1),
        )

    # ------------------------------------------------------- phase schedule
    def _apply_phase(self, epoch_idx):
        m = self._aeq_module()
        if m is None:
            return
        ph = self.aeq_cfg.get("phases", {}) or {}
        a_end = int(ph.get("unroll_epochs", 5))
        b_end = int(ph.get("softplus_epochs", 10))     # cumulative
        d_start = int(ph.get("gating_epoch", 15))
        # Each curriculum rep builds a FRESH runner, so the local epoch index
        # restarts at 0 -- which would drag a warm-started model back to
        # phase A (unrolled + softplus after it had reached ELU). Set
        # phases.epoch_offset on a resumed stage to enter at the right phase
        # (e.g. offset >= gating_epoch => start already in phase D).
        epoch_idx = epoch_idx + int(ph.get("epoch_offset", 0))

        if epoch_idx < a_end:
            mode, act, phase = "unroll", "softplus", "A"
        elif epoch_idx < b_end:
            mode, act, phase = "solve", "softplus", "B"
        elif epoch_idx < d_start:
            mode, act, phase = "solve", "elu", "C"
        else:
            mode, act, phase = "solve", "elu", "D"

        gate_wanted = (phase == "D"
                       and (self.aeq_cfg.get("site_gating", {}) or {})
                       .get("enabled", False))
        changed = (m.mode != mode or m.act_name != act
                   or getattr(self, "_aeq_phase", None) != phase)
        m.mode, m.act_name = mode, act
        self._aeq_gate_wanted = gate_wanted
        if not gate_wanted:
            m.gating_on = False
        # NOTE: gating_on flips True only after the Sec. 7.2 random-mask gate
        # passes (checked periodically in handle_batch).
        if changed:
            self._aeq_phase = phase
            print(f"[aeq] epoch {epoch_idx}: phase {phase} "
                  f"(mode={mode}, act={act}, gate_wanted={gate_wanted})",
                  file=sys.stderr, flush=True)

    def on_epoch_start(self, runner):
        epoch_idx = int(getattr(self, "_local_epoch_index", 0))
        super().on_epoch_start(runner)
        self._apply_phase(epoch_idx)

    # ------------------------------------------------------------ the batch
    def handle_batch(self, batch):
        m = self._aeq_module()
        if m is None:
            return super().handle_batch(batch)

        # Sec. 7.2 empirical gate, every stability.async_gate_every steps,
        # while site gating is wanted. Runs two extra no_grad solves on the
        # current batch -- a few percent overhead at the default cadence.
        if m.training and getattr(self, "_aeq_gate_wanted", False):
            stab = self.aeq_cfg.get("stability", {}) or {}
            every = int(stab.get("async_gate_every", 50))
            frac = float(stab.get("async_gate_mask_frac", 0.5))
            self._aeq_step = getattr(self, "_aeq_step", 0) + 1
            if self._aeq_step % max(1, every) == 1:
                sample = batch[0]
                passed, gap = m.random_mask_gate(sample, mask_frac=frac)
                # DDP: make the verdict global (all ranks must agree, and the
                # gate opens only if EVERY rank's batch passed), so gating_on
                # never diverges across replicas. All ranks hit this at the
                # same step, so the collective cannot deadlock.
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        t = torch.tensor(
                            [1.0 if passed else 0.0], device=sample.device)
                        dist.all_reduce(t, op=dist.ReduceOp.MIN)
                        passed = bool(t.item() > 0.5)
                except Exception:
                    pass
                if passed and not m.gating_on:
                    print(f"[aeq] random-mask gate PASSED (gap={gap:.2e}) "
                          f"-> site gating ON", file=sys.stderr, flush=True)
                if not passed and m.gating_on:
                    print(f"[aeq] random-mask gate FAILED (gap={gap:.2e}) "
                          f"-> site gating OFF; tighten rowsum_target/m_max",
                          file=sys.stderr, flush=True)
                m.gating_on = passed

        # Decide BEFORE the forward whether this step collects the sync-costly
        # scalar stats (m_mean, y_absmax, per-sweep active fraction). On every
        # other step the solver skips them entirely.
        every = int((self.aeq_cfg.get("log", {}) or {}).get("every_n_steps", 20))
        self._aeq_log_step = getattr(self, "_aeq_log_step", 0) + 1
        want_log = (m.training and self._is_main()
                    and self._aeq_log_step % max(1, every) == 0)
        m.collect_stats = want_log
        # Amortize the Hutchinson penalty (see AEQMeshNet.jac_every). All ranks
        # use the same step counter, so they agree on which steps pay it --
        # important under DDP, where a rank skipping it would change which
        # parameters have grads.
        m.jac_this_step = (self._aeq_log_step % m.jac_every == 0)

        out = super().handle_batch(batch)

        # wandb-only diagnostics (off the tqdm bar)
        if want_log:
            try:
                import wandb
                if wandb.run is not None:
                    s = m.stats
                    log = {f"aeq/{k}": v for k, v in s.items()
                           if isinstance(v, (int, float))}
                    curve = s.get("active_frac_curve") or []
                    if curve:
                        log["aeq/active_frac_mean"] = sum(curve) / len(curve)
                    log.update({f"aeq/{k}": v
                                for k, v in m.rowsum_report().items()})
                    if torch.cuda.is_available():
                        # allocated = live tensors; reserved = caching-allocator
                        # pool (what nvidia-smi/nvitop shows). A climbing
                        # ALLOCATED curve is a real leak; flat allocated with
                        # high reserved is fragmentation/caching.
                        g = 1024 ** 3
                        log["aeq/mem_alloc_gb"] = torch.cuda.memory_allocated() / g
                        log["aeq/mem_reserved_gb"] = torch.cuda.memory_reserved() / g
                        log["aeq/mem_max_alloc_gb"] = (
                            torch.cuda.max_memory_allocated() / g)
                    wandb.log(log, commit=False)
            except Exception:
                pass
        return out


@hydra.main(config_path="conf", config_name="aeq_siam18_16ch", version_base=None)
def main(cfg: DictConfig):
    aeq = cfg.model.get("aeq", {})
    aeq = OmegaConf.to_container(aeq, resolve=True) if aeq else {}
    # Ship the AEQ config to the DDP ranks through the environment (mp.spawn
    # children inherit env; they re-import this module and read it there).
    os.environ["AEQ_CFG_JSON"] = json.dumps(aeq)
    AEQRunner.aeq_cfg = aeq

    # Same perf setup as fast.main, minus compile/channels_last (the solver
    # loop and the custom autograd.Function are not compilable, and eager GN
    # has no NHWC 3D kernel). NOTE: we cannot call fast.main.__wrapped__ --
    # it would reset base.CustomRunner to FastRunner.
    perf = cfg.get("perf", {}) or {}
    if perf.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
    if perf.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    AEQRunner.perf_compile = False
    AEQRunner.perf_compile_loss = False
    AEQRunner.perf_channels_last = False
    AEQRunner.perf_verbose = bool(perf.get("verbose", False))
    AEQRunner.perf_profile = bool(perf.get("profile", False))
    AEQRunner.perf_profile_steps = int(perf.get("profile_steps", 10))
    AEQRunner.checkpoint_segments = cfg.model.get("checkpoint_segments", None)
    AEQRunner.checkpoint_keep_layers = int(cfg.model.get("checkpoint_keep_layers", 0))

    base.CustomRunner = AEQRunner
    return base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
