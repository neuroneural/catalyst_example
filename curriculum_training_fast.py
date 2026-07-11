"""Throughput-optimized training entrypoint.

Same training logic as ``curriculum_training.py`` but with GPU-efficiency
features layered on top, all toggleable from the hydra config:

  * ``torch.backends.cudnn.benchmark``  -> let cuDNN autotune conv algorithms
    for the fixed 256^3 shapes (big win for the unusual dilation rates).
  * TF32 matmul / cuDNN paths on Ampere+.
  * ``channels_last_3d`` memory format for model + input, so cuDNN picks its
    faster NHWC 3D conv kernels (this workload is memory-bandwidth bound, so
    layout matters a lot).
  * ``torch.compile`` to fuse the conv -> GroupNorm -> GELU chains and the
    argmax/dice tail into fewer kernels, attacking the launch-overhead +
    small-kernel pattern.
  * ``model.checkpoint_segments`` to coarsen gradient checkpointing (fewer,
    larger recompute segments => less recompute + fewer launches, at the cost
    of higher peak training memory).

Nothing here changes the math of the model or the loss; it only changes how
the work is scheduled on the GPU. Inference (e.g. the browser deployment) is
untouched.

Run it exactly like the base trainer, e.g.:

    python curriculum_training_fast.py --config-name gn_hdc_deep

Config knobs (all optional, with the defaults shown):

    model:
      checkpoint_segments: null   # null/0 => per-layer (original behavior)

    perf:
      cudnn_benchmark: true
      tf32: true
      channels_last: true
      compile: true
      compile_mode: "default"     # or "max-autotune", "reduce-overhead"
      verbose: false              # print per-step timing & diagnostics
"""

import os
import sys
import time

# IMPORTANT: set before torch/inductor spins up its compile-worker pool, and at
# MODULE scope so it runs in every process -- including the DDP ranks that
# Catalyst creates with mp.spawn(start_method="spawn"). Those ranks re-import
# this module but never execute main(), so anything set only in main() (as in an
# earlier version) never reaches them.
#
# Why it matters: the mindfultensors DataLoader workers each hold a live
# MongoClient. TorchInductor's default async-compile pool forks subprocess
# workers on the first torch.compile (== first batch), exactly while those Mongo
# workers are alive. pymongo is not fork-safe, so the inherited client gets
# closed -> "Cannot use MongoClient after close" on the next cursor read.
# COMPILE_THREADS=1 makes inductor compile in-process (no pool, no fork).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import hydra
import torch
from omegaconf import DictConfig

import curriculum_training as base


def _strip_orig_mod_prefix(state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs):
    """load_state_dict pre-hook: drop the '_orig_mod.' prefix that
    torch.compile's OptimizedModule wrapper adds to checkpoint keys. Lets us
    resume checkpoints that were saved while the model was compile-wrapped
    (e.g. from an earlier run) into a plain or in-place-compiled model."""
    for k in [k for k in state_dict if "_orig_mod." in k]:
        state_dict[k.replace("_orig_mod.", "")] = state_dict.pop(k)


# Defaults; overwritten from cfg.perf / cfg.model in main() before any runner
# is constructed. Kept as class attributes so they survive the per-rep
# create/destroy cycle in base.main().
class FastRunner(base.CustomRunner):
    perf_channels_last = True
    perf_compile = True
    perf_compile_mode = "default"
    perf_verbose = False
    perf_profile = False
    perf_profile_steps = 10
    checkpoint_segments = None
    checkpoint_keep_layers = 0

    def get_model(self):
        model = super().get_model()

        # Tolerate checkpoints saved with torch.compile's "_orig_mod." key
        # prefix (e.g. from a prior run that used wrapper-style compile). This
        # is what Catalyst's resume_model loads into.
        try:
            model._register_load_state_dict_pre_hook(_strip_orig_mod_prefix)
        except Exception as exc:  # pragma: no cover
            print(f"[fast] could not register state_dict prefix hook: {exc}",
                  file=sys.stderr, flush=True)

        # Coarsen gradient checkpointing. Honored by CheckpointMixin variants;
        # plain (manual-backprop) variants simply ignore the attribute.
        if self.checkpoint_segments:
            try:
                model.checkpoint_segments = int(self.checkpoint_segments)
                if self.perf_verbose:
                    print(
                        f"[fast] checkpoint_segments={model.checkpoint_segments}",
                        file=sys.stderr, flush=True,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[fast] could not set checkpoint_segments: {exc}",
                      file=sys.stderr, flush=True)

        # Partial checkpointing: keep the first N layers un-checkpointed (no
        # recompute) and checkpoint the rest. Spends spare GPU memory to cut the
        # recompute tax. 0 (default) = full checkpointing (unchanged behavior).
        try:
            model.checkpoint_keep_layers = int(self.checkpoint_keep_layers or 0)
            if self.perf_verbose and model.checkpoint_keep_layers:
                print(f"[fast] checkpoint_keep_layers={model.checkpoint_keep_layers}",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[fast] could not set checkpoint_keep_layers: {exc}",
                  file=sys.stderr, flush=True)

        if self.perf_channels_last:
            try:
                model = model.to(memory_format=torch.channels_last_3d)
                if self.perf_verbose:
                    print("[fast] model -> channels_last_3d", file=sys.stderr, flush=True)
            except Exception as exc:  # pragma: no cover
                print(f"[fast] channels_last_3d failed: {exc}",
                      file=sys.stderr, flush=True)

        # NOTE: torch.compile is intentionally NOT applied here. Under DDP,
        # Catalyst wraps the model returned by get_model() *after* this point, so
        # compiling now would give DDP(torch.compile(model)) -- compile inside,
        # DDP outside -- which disables Dynamo's DDPOptimizer and is the
        # unsupported ordering. We instead compile lazily on the first batch
        # (see handle_batch), once self.model is the final, DDP-wrapped module,
        # yielding the supported torch.compile(DDP(model)) ordering.
        return model

    def _maybe_compile_model(self):
        if not self.perf_compile or getattr(self, "_compiled", False):
            return
        # Don't compile the shape > maxshape branch: that path uses a different
        # (manual-backprop) model and self.model.no_sync(), which an
        # OptimizedModule wrapper would hide. compile only helps the normal AMP
        # checkpointed path anyway.
        if self.shape > self.maxshape:
            self._compiled = True  # skip permanently for this rep
            if self.perf_verbose:
                print("[fast] skipping torch.compile (shape > maxshape path)",
                      file=sys.stderr, flush=True)
            return
        try:
            # In-place compile (nn.Module.compile) rather than rebinding to
            # torch.compile(...)'s OptimizedModule wrapper. Two reasons:
            #  1) state_dict keys stay clean -- a wrapper prefixes every key with
            #     "_orig_mod.", so Catalyst would save checkpoints that then fail
            #     to load into a fresh (uncompiled) model on the next curriculum
            #     rep ("Missing/Unexpected key(s) ... _orig_mod.*"). Clean keys
            #     also matter for exporting weights to the browser model.
            #  2) self.model stays the DDP module, so .no_sync()/.module still
            #     work. Compiling the DDP module's forward still engages Dynamo's
            #     DDPOptimizer.
            self.model.compile(mode=self.perf_compile_mode)
            if self.perf_verbose:
                print(f"[fast] in-place compile(mode={self.perf_compile_mode}) "
                      f"on {self.model.__class__.__name__} (post-DDP)",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[fast] torch.compile failed, running eager: {exc}",
                  file=sys.stderr, flush=True)
        self._compiled = True

    def get_callbacks(self):
        cbs = super().get_callbacks()
        # Make checkpoint resume verifiable: did Catalyst get a resume path, and
        # does the file actually exist? (Silent no-op resume => training from
        # scratch => low starting dice.)
        mp = getattr(self, "model_path", "") or ""
        if self.perf_verbose:
            if mp:
                exists = os.path.exists(mp)
                extra = f" size={os.path.getsize(mp)}B" if exists else ""
                print(f"[fast/resume] resume_model={mp} exists={exists}{extra}",
                      file=sys.stderr, flush=True)
            else:
                print("[fast/resume] NO resume_model (paths.loadcheckpoint=False or "
                      "empty paths.model) -> starting from scratch",
                      file=sys.stderr, flush=True)
        return cbs

    def _weight_fingerprint(self):
        import itertools
        with torch.no_grad():
            ps = list(itertools.islice(
                (p for p in self.model.parameters() if p.requires_grad), 5))
            return sum(float(p.float().abs().sum()) for p in ps)

    def _is_main(self):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        return True

    def _maybe_profiler_enter(self):
        """Lazily start a one-shot torch.profiler window (main rank only).
        Captures wait+warmup then `perf_profile_steps` active steps, dumps an op
        table + chrome trace, and disables itself. Off unless perf.profile=True.
        """
        if not self.perf_profile or not self._is_main():
            return
        if getattr(self, "_prof_done", False) or hasattr(self, "_prof"):
            return
        try:
            from torch.profiler import profile, ProfilerActivity, schedule
            wait, warmup, active = 5, 3, int(self.perf_profile_steps)
            self._prof_total = wait + warmup + active
            self._prof_n = 0
            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                record_shapes=True, profile_memory=True, with_stack=False,
            )
            self._prof.__enter__()
            print(f"[fast/profile] capturing {active} steps after "
                  f"{wait + warmup} warmup steps...", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[fast/profile] could not start profiler: {exc}",
                  file=sys.stderr, flush=True)
            self._prof_done = True

    def _maybe_profiler_step(self):
        prof = getattr(self, "_prof", None)
        if prof is None:
            return
        prof.step()
        self._prof_n += 1
        if self._prof_n >= self._prof_total:
            try:
                prof.__exit__(None, None, None)
                table = prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=25)
                print("[fast/profile] top ops by CUDA time:\n" + table,
                      file=sys.stderr, flush=True)
                path = os.path.abspath("profile_trace.json")
                prof.export_chrome_trace(path)
                print(f"[fast/profile] chrome trace -> {path} "
                      f"(open in chrome://tracing or ui.perfetto.dev)",
                      file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[fast/profile] dump failed: {exc}",
                      file=sys.stderr, flush=True)
            finally:
                self._prof_done = True
                del self._prof

    def handle_batch(self, batch):
        # Compile here (not in get_model): by the first batch self.model is the
        # fully-prepared, DDP-wrapped module, so we get torch.compile(DDP(model)).
        self._maybe_compile_model()

        # One-time fingerprint AFTER resume has run, to confirm weights loaded.
        if not getattr(self, "_fp_logged", False):
            if self.perf_verbose:
                try:
                    print(f"[fast/resume] first-batch weight fingerprint(first5 "
                          f"params abs-sum)={self._weight_fingerprint():.4f} "
                          f"(compare across runs; random init differs from resumed)",
                          file=sys.stderr, flush=True)
                except Exception as exc:
                    print(f"[fast/resume] fingerprint failed: {exc}",
                          file=sys.stderr, flush=True)
            self._fp_logged = True

        # Match the input layout to the model so cuDNN keeps the NHWC fast path
        # instead of inserting layout-conversion copies every step.
        if self.perf_channels_last:
            sample, label = batch
            sample = sample.contiguous(memory_format=torch.channels_last_3d)
            batch = (sample, label)

        # One-shot profiler window (gated by perf.profile; main rank only).
        self._maybe_profiler_enter()

        # Fast path (production): no instrumentation, no per-step sync.
        if not self.perf_verbose:
            out = super().handle_batch(batch)
            self._maybe_profiler_step()
            return out

        # --- verbose-only timing: separates data-wait from GPU compute, but
        # costs a per-step torch.cuda.synchronize() (serializes CPU<->GPU), so
        # it must never run in production. Rolling window = steady state.
        now = time.perf_counter()
        prev_end = getattr(self, "_t_prev_end", None)
        if prev_end is not None:
            self._t_wait = getattr(self, "_t_wait", 0.0) + (now - prev_end)
        t0 = time.perf_counter()
        out = super().handle_batch(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        WINDOW = 20
        self._t_compute = getattr(self, "_t_compute", 0.0) + (t1 - t0)
        n = getattr(self, "_step_count", 0) + 1
        self._step_count = n
        if n % WINDOW == 0:
            w_compute = self._t_compute - getattr(self, "_t_compute_mark", 0.0)
            w_wait = getattr(self, "_t_wait", 0.0) - getattr(self, "_t_wait_mark", 0.0)
            avg_c = 1000 * w_compute / WINDOW
            avg_w = 1000 * w_wait / WINDOW
            bound = "DATA-bound" if avg_w > avg_c else "COMPUTE-bound"
            print(f"[fast/timing] step {n}: compute~{avg_c:.0f}ms/step "
                  f"data-wait~{avg_w:.0f}ms/step (last {WINDOW}) -> {bound}",
                  file=sys.stderr, flush=True)
            self._t_compute_mark = self._t_compute
            self._t_wait_mark = getattr(self, "_t_wait", 0.0)
        self._maybe_profiler_step()
        self._t_prev_end = time.perf_counter()
        return out


@hydra.main(config_path="conf", config_name="gn_hdc_deep", version_base=None)
def main(cfg: DictConfig):
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

    FastRunner.perf_channels_last = bool(perf.get("channels_last", True))
    FastRunner.perf_compile = bool(perf.get("compile", True))
    FastRunner.perf_compile_mode = str(perf.get("compile_mode", "default"))
    FastRunner.perf_verbose = bool(perf.get("verbose", False))
    FastRunner.perf_profile = bool(perf.get("profile", False))
    FastRunner.perf_profile_steps = int(perf.get("profile_steps", 10))
    FastRunner.checkpoint_segments = cfg.model.get("checkpoint_segments", None)
    FastRunner.checkpoint_keep_layers = int(cfg.model.get("checkpoint_keep_layers", 0))

    # NB: TORCHINDUCTOR_COMPILE_THREADS is set at module import (top of file) so
    # it also applies in the spawned DDP rank processes. Don't move it here --
    # main() does not run in those ranks.

    if FastRunner.perf_verbose:
        print(
            "[fast] cudnn_benchmark={} tf32={} channels_last={} compile={}({}) "
            "checkpoint_segments={}".format(
                torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32,
                FastRunner.perf_channels_last,
                FastRunner.perf_compile,
                FastRunner.perf_compile_mode,
                FastRunner.checkpoint_segments,
            ),
            file=sys.stderr, flush=True,
        )

    # Route base.main() through the optimized runner.
    base.CustomRunner = FastRunner
    return base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
