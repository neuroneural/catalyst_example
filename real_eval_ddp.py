"""Periodic real-data evaluation, split into a GPU phase and a CPU phase.

Why this file exists
--------------------
The first cut of this feature launched ``eval_real_metrics.py`` as a detached
subprocess on rank 0 while training kept running. That was a bad design:

  * it reloaded the checkpoint into a SECOND CUDA context on GPU 0, so rank 0
    paid both the memory and the SM contention while ranks 1..N-1 idled
    through the same wall-clock;
  * the expensive part (surface distance transforms) is pure CPU, but it was
    stuck behind a serial, single-process GPU inference loop;
  * "runs in parallel with training" meant "competes with training for the one
    GPU that also drives the collective" -- the worst rank to slow down, since
    every allreduce waits for the slowest rank.

This file does the obvious better thing, in two phases:

  PHASE 1 (GPU, synchronous, ALL ranks, on the critical path -- but short)
    Every rank runs inference on its own shard ``ids[rank::world_size]`` of the
    real subjects, using the LIVE in-memory model. No checkpoint reload, no
    second CUDA context, no extra process on the GPU. With W ranks the
    inference is W-way parallel, so 24 subjects on 4 GPUs is 6 forwards per
    rank -- seconds, not minutes. Each rank writes its (pred, gt) label pair to
    a spool directory as cropped uint8 and drops a ``rank_<r>.done`` marker.

  PHASE 2 (CPU, asynchronous, rank 0 launches and forgets)
    Rank 0 spawns ``real_eval_reduce.py`` -- a detached, CUDA-FREE, nice'd
    process that waits for the spool to fill, then runs Dice/NSD/HD95/ASSD over
    subjects in a ``multiprocessing.Pool``. Training starts the next epoch
    immediately. The reducer only ever touches CPU cores and the spool dir, so
    it cannot contend for GPU or perturb the collective.

Deadlock safety
---------------
This callback adds **zero collective operations** to the training loop. There is
no ``dist.barrier()``, and inference runs on the UNWRAPPED module (DDP's forward
can broadcast buffers -- a collective -- and ``torch.compile``'s wrapper would
recompile under ``no_grad``; both are avoided by unwrapping). Ranks therefore
never have to agree on anything, so an exception on one rank cannot hang the
others, and shard sizes are equalized only for tidiness, not for correctness.
Rank 0 tells the reducer how many ranks to wait for, and the reducer times out
rather than blocking forever if a rank died.

Config (``real_eval:`` block in the yaml, reaches here as ``runner.real_eval_cfg``):
    enabled: false
    every_n_epochs: 10
    max_subjects: 24
    collection: "MRN"
    labelfield: "label104"
    datafield: "T1"
    tau: 1.0                 # NSD tolerance, mm
    spacing: "1,1,1"
    fail_dice: 0.5
    exclude_classes: ""      # e.g. "16,17" -- kept per-class, off the headline
    num_workers: 2           # loader workers for the eval shard (per rank)
    prefetch_factor: 2
    amp: true                # run inference under the training autocast dtype
    jobs: 0                  # reducer pool size; 0 => auto (see below)
    nice: 10                 # reducer niceness -- training's loader wins ties
    spool_dir: null          # default <logdir>/real_eval/epoch_NNNNN/spool
    keep_spool: false
    timeout_s: 1800          # reducer gives up waiting for missing ranks
    compress: false          # savez_compressed: ~4x smaller, ~10x slower write
    aux_missing: false       # if true, spool TwoHeadMeshNet's missing prediction
    reference_missing_id: null  # e.g. 104 for a post-op target; null for healthy
"""
import contextlib
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from catalyst import dl

_HERE = os.path.dirname(os.path.abspath(__file__))


def _unwrap(model):
    """Peel torch.compile's OptimizedModule and DDP off to reach the raw nn.Module.

    Both wrappers are actively harmful here. DDP.forward() may broadcast buffers
    (a COLLECTIVE -- and our ranks run different numbers of eval forwards, so it
    would desync). OptimizedModule.forward() under ``no_grad`` is a new guard
    configuration, so torch.compile would trigger a full inductor recompile
    on the critical path, every eval round, on every rank.
    """
    for _ in range(4):
        inner = getattr(model, "_orig_mod", None)
        if inner is None:
            inner = getattr(model, "module", None)
        if inner is None or inner is model:
            break
        model = inner
    return model


def _rank_world():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


class RealEvalCallback(dl.Callback):
    """See module docstring. Order=External so it runs after the checkpointer."""

    def __init__(self):
        # External (100) > Checkpoint (50): we run after the epoch's checkpoint
        # is on disk, though we deliberately evaluate the LIVE weights rather
        # than re-reading that file.
        super().__init__(order=dl.CallbackOrder.External)
        self._local_epoch = 0
        self._proc = None
        self._ids = None
        self._loader = None
        self._loader_key = None
        self._logged = set()
        self._base_epoch = None
        self._startup_done = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _cfg(runner):
        return getattr(runner, "real_eval_cfg", None) or {}

    @staticmethod
    def _root(runner):
        return os.path.join(getattr(runner, "_logdir", "."), "real_eval")

    @staticmethod
    def _counter_path(runner):
        """Cross-restart epoch counter.

        This run is launched in 4h SLURM slices at ~35 min/epoch, i.e. ~6 epochs
        per job, and resume loads MODEL WEIGHTS ONLY -- so Catalyst's epoch
        counters (and a naive in-callback counter) restart from 1 every job. With
        every_n_epochs > epochs-per-job that means the eval NEVER FIRES. Persist
        the count instead so `every_n_epochs` means what it says across restarts.
        """
        return os.path.join(RealEvalCallback._root(runner), "epochs_seen")

    @staticmethod
    def _marker(runner):
        """Filesystem 'a reduce job is in flight' flag.

        Lives on disk rather than in rank 0's memory so that EVERY rank makes
        the same skip/run decision without a collective: if the previous round's
        reducer is still chewing, nobody spools a new round on top of it.
        """
        return os.path.join(RealEvalCallback._root(runner), "REDUCING")

    def _log(self, msg):
        print(f"[real-eval] {msg}", file=sys.stderr, flush=True)

    # --------------------------------------------------------------- startup
    def on_epoch_start(self, runner):
        """Read the persisted epoch count, and clean up after a killed job.

        Deliberately on_epoch_START: every rank reads the counter file here, which
        strictly precedes rank 0's first write at the end of that same epoch. So
        all ranks agree on the base for the whole job without a collective. Doing
        this lazily in on_epoch_end would be a race.
        """
        cfg = self._cfg(runner)
        if not cfg.get("enabled", False) or self._startup_done:
            return
        self._startup_done = True
        root = self._root(runner)
        try:
            os.makedirs(root, exist_ok=True)
            try:
                with open(self._counter_path(runner)) as f:
                    self._base_epoch = int((f.read().strip() or "0"))
            except (OSError, ValueError):
                self._base_epoch = 0
            rank, _ = _rank_world()
            if rank == 0:
                self._reap_previous_job(runner)
            # Past rounds are already in history.csv; don't re-push them to the
            # new wandb run. Recap the latest one so the log shows where we are.
            prev = sorted(d for d in os.listdir(root)
                          if os.path.isfile(os.path.join(root, d, "summary.json")))
            self._logged.update(prev)
            self._log(f"epoch count resumes at {self._base_epoch}; "
                      f"{len(prev)} completed round(s) on disk"
                      + (f", latest {prev[-1]}" if prev else ""))
        except Exception as exc:
            self._base_epoch = self._base_epoch or 0
            self._log(f"startup check failed (ignored): {exc}")

    def _keep_spool(self, runner):
        try:
            return bool(self._cfg(runner).get("keep_spool", False))
        except Exception:
            return False

    def _reap_previous_job(self, runner):
        """A reducer cannot outlive its SLURM step: when the 4h wall clock kills
        the job mid-reduce, it leaves a REDUCING marker that would make EVERY
        later job skip the eval forever, plus a spool of npz files that nothing
        can reduce any more. Clear both, loudly."""
        marker = self._marker(runner)
        if os.path.exists(marker):
            self._log("clearing a REDUCING marker left behind by a previous job "
                      "(its reducer died with the SLURM step)")
            try:
                os.remove(marker)
            except OSError:
                pass
        freed = 0
        for d in glob.glob(os.path.join(self._root(runner), "epoch_*", "spool")):
            # keep_spool means the spool is a deliberate artifact (error_attrib.py
            # reads {pred,gt} straight out of it), so a job restart must not wipe
            # it. Only reap a spool whose round never produced a summary.json,
            # i.e. one that really is unreducible. Without keep_spool the old
            # behaviour is unchanged: reap everything.
            if self._keep_spool(runner) and os.path.exists(
                    os.path.join(os.path.dirname(d), "summary.json")):
                continue
            for p in glob.glob(os.path.join(d, "*")):
                try:
                    freed += os.path.getsize(p)
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(d)
            except OSError:
                pass
        if freed:
            self._log(f"deleted {freed / 1e6:.0f} MB of orphaned spool files "
                      f"(unreducible without the job that wrote them)")

    # ------------------------------------------------------------------- ids
    def _subject_ids(self, runner, cfg):
        if self._ids is not None:
            return self._ids
        from mindfultensors.mongoloader import MongoClient
        db_name = "MindfulTensors"
        col = str(cfg.get("collection", "MRN"))
        index_id = runner.index_id
        mc = MongoClient("mongodb://" + runner.db_host + ":27017")
        try:
            try:
                ids = sorted(int(x) for x in mc[db_name][f"{col}.meta"].distinct(index_id))
            except Exception:
                ids = []
            if not ids:
                ids = sorted(int(x) for x in mc[db_name][f"{col}.bin"].distinct(index_id))
        finally:
            mc.close()
        ids = ids[: int(cfg.get("max_subjects", 24))]
        _, world = _rank_world()
        if world > 1 and len(ids) >= world:
            # Equal shards: keeps the wall-clock of phase 1 == one rank's share
            # instead of one unlucky rank doing an extra subject while the rest
            # wait at the next allreduce.
            keep = (len(ids) // world) * world
            if keep != len(ids):
                self._log(f"truncating {len(ids)} -> {keep} subjects for {world} even shards")
            ids = ids[:keep]
        self._ids = ids
        return ids

    def _build_loader(self, runner, cfg, my_ids):
        key = (tuple(my_ids), cfg.get("datafield"), cfg.get("labelfield"),
               cfg.get("collection"))
        if self._loader is not None and self._loader_key == key:
            return self._loader
        import curriculum_training as base
        from mindfultensors.mongoloader import MongoDataset
        from mindfultensors.utils import unit_interval_normalize, DBBatchSampler
        from torch.utils.data import DataLoader

        cc = base.ClientCreator(runner.db_host)
        cc.set_database("MindfulTensors")
        cc.set_collection(str(cfg.get("collection", "MRN")))
        cc.set_shape([256, 256, 256])
        cc.set_num_subcubes(1)
        fields = (str(cfg.get("datafield", "T1")), str(cfg.get("labelfield", "label104")))
        ds = MongoDataset(list(my_ids), cc.mytransform, None, fields,
                          normalize=unit_interval_normalize, id=runner.index_id)
        sampler = DBBatchSampler(ds, batch_size=1, seed=42)
        # persistent_workers=False on purpose: these workers each hold a Mongo
        # client, and they must NOT stay resident through the 10 epochs between
        # eval rounds (this repo has already been OOM-killed once by resident
        # loader workers).
        self._loader = DataLoader(
            ds, sampler=sampler, collate_fn=cc.mycollate_full, pin_memory=False,
            worker_init_fn=cc.create_client, persistent_workers=False,
            prefetch_factor=int(cfg.get("prefetch_factor", 2)) if int(cfg.get("num_workers", 2)) > 0 else None,
            num_workers=int(cfg.get("num_workers", 2)),
        )
        self._loader_key = key
        return self._loader

    # -------------------------------------------------------------- phase 1
    def _infer_shard(self, runner, cfg, spool, my_ids):
        """Run the live model over this rank's subjects and spool uint8 labels."""
        rank, world = _rank_world()
        n_classes = int(runner.n_classes)
        assert n_classes <= 256, "spool dtype is uint8; widen it for >256 classes"
        device = next(_unwrap(runner.model).parameters()).device
        model = _unwrap(runner.model)
        was_training = model.training
        channels_last = bool(getattr(runner, "perf_channels_last", False))
        use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
        amp_name = str(getattr(runner, "amp_dtype", "float16")).lower()
        amp_dtype = torch.bfloat16 if amp_name in ("bf16", "bfloat16") else torch.float16
        compress = bool(cfg.get("compress", False))

        loader = self._build_loader(runner, cfg, my_ids)
        model.eval()
        n_done = 0
        t0 = time.time()
        try:
            for i, batch in enumerate(loader):
                sample, label = batch
                sample = sample.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                if channels_last:
                    sample = sample.contiguous(memory_format=torch.channels_last_3d)
                with torch.no_grad():
                    # nullcontext rather than autocast(enabled=False): a CPU-only
                    # fallback must not construct a cuda autocast region at all.
                    ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                           if use_amp else contextlib.nullcontext())
                    with ctx:
                        if bool(cfg.get("aux_missing", False)):
                            logits, aux_logits = model(sample, return_aux=True)
                        else:
                            logits, aux_logits = model(sample), None
                    pred = torch.argmax(logits.float(), dim=1).squeeze()
                    del logits, sample
                    raw_gt = label.squeeze().to(torch.int64)
                    ref_id = cfg.get("reference_missing_id", None)
                    ref_missing = ((raw_gt == int(ref_id)) if ref_id is not None
                                   else None)
                    aux_pred = (torch.argmax(aux_logits.float(), dim=1).squeeze()
                                if aux_logits is not None else None)
                    del aux_logits
                    gt = raw_gt
                    gt = torch.where(gt < n_classes, gt, torch.zeros_like(gt))
                    del label, raw_gt
                    # Crop to the union foreground box (+margin) before leaving
                    # the GPU. Everything outside is background in BOTH volumes,
                    # so no per-class surface distance can change; it just makes
                    # the spooled file and every downstream EDT ~2-3x smaller.
                    fg = (gt > 0) | (pred > 0)
                    if aux_pred is not None:
                        fg |= aux_pred > 0
                    if ref_missing is not None:
                        fg |= ref_missing
                    bbox = self._bbox(fg, margin=4)
                    if bbox is not None:
                        (z0, z1), (y0, y1), (x0, x1) = bbox
                        pred = pred[z0:z1, y0:y1, x0:x1]
                        gt = gt[z0:z1, y0:y1, x0:x1]
                        if aux_pred is not None:
                            aux_pred = aux_pred[z0:z1, y0:y1, x0:x1]
                        if ref_missing is not None:
                            ref_missing = ref_missing[z0:z1, y0:y1, x0:x1]
                    pred_np = pred.to(torch.uint8).cpu().numpy()
                    gt_np = gt.to(torch.uint8).cpu().numpy()
                    aux_np = (aux_pred.to(torch.uint8).cpu().numpy()
                              if aux_pred is not None else None)
                    ref_np = (ref_missing.to(torch.uint8).cpu().numpy()
                              if ref_missing is not None else None)
                    del pred, gt, fg

                sid = int(my_ids[i]) if i < len(my_ids) else -(rank * 1000 + i)
                path = os.path.join(spool, f"subj_r{rank}_{i:03d}_{sid}.npz")
                # Write-then-rename: the reducer may be scanning this directory
                # while we write, and must never see a half-written array.
                tmp = path + ".tmp.npz"
                saver = np.savez_compressed if compress else np.savez
                with open(tmp, "wb") as fh:
                    payload = {"pred": pred_np, "gt": gt_np,
                               "sid": np.int64(sid)}
                    if aux_np is not None:
                        payload["aux_missing"] = aux_np
                    if ref_np is not None:
                        payload["ref_missing"] = ref_np
                    saver(fh, **payload)
                os.replace(tmp, path)
                n_done += 1
        finally:
            model.train(was_training)
        dt = time.time() - t0
        self._log(f"rank {rank}/{world}: {n_done} subjects inferred in {dt:.1f}s "
                  f"({dt / max(1, n_done):.1f}s/subj) -> {spool}")
        with open(os.path.join(spool, f"rank_{rank}.done"), "w") as f:
            f.write(json.dumps({"rank": rank, "n": n_done, "seconds": dt}))
        return n_done

    @staticmethod
    def _bbox(mask, margin=4):
        if not bool(mask.any()):
            return None
        out = []
        for ax in range(3):
            d0, d1 = [d for d in range(3) if d != ax]   # d0 < d1
            proj = mask.any(dim=d1).any(dim=d0)          # reduce the larger axis first
            idx = torch.nonzero(proj, as_tuple=False)
            lo = int(idx[0]) - margin
            hi = int(idx[-1]) + 1 + margin
            out.append((max(0, lo), min(mask.shape[ax], hi)))
        return tuple(out)

    # -------------------------------------------------------------- phase 2
    def _launch_reducer(self, runner, cfg, out_dir, spool, n_ranks, n_subj):
        script = os.path.join(_HERE, "real_eval_reduce.py")
        jobs = int(cfg.get("jobs", 0))
        if jobs <= 0:
            # Leave the training loader's workers their cores. os.cpu_count()
            # over-reports under cgroups/SLURM, so prefer the affinity mask.
            try:
                ncpu = len(os.sched_getaffinity(0))
            except Exception:
                ncpu = os.cpu_count() or 4
            reserved = int(getattr(runner, "num_workers", 4)) + 2
            jobs = max(1, min(16, ncpu - reserved))
        cmd = [
            sys.executable, script,
            "--spool", spool,
            "--out", out_dir,
            "--n-classes", str(int(runner.n_classes)),
            "--tau", str(cfg.get("tau", 1.0)),
            "--spacing", str(cfg.get("spacing", "1,1,1")),
            "--fail-dice", str(cfg.get("fail_dice", 0.5)),
            "--exclude-classes", str(cfg.get("exclude_classes", "") or ""),
            "--expect-ranks", str(int(n_ranks)),
            "--expect-subjects", str(int(n_subj)),
            "--jobs", str(jobs),
            "--nice", str(int(cfg.get("nice", 10))),
            "--timeout", str(int(cfg.get("timeout_s", 1800))),
            "--marker", self._marker(runner),
        ]
        if bool(cfg.get("keep_spool", False)):
            cmd.append("--keep-spool")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""       # hard guarantee: no GPU contention
        env["OMP_NUM_THREADS"] = "1"           # the parallelism is the pool, not BLAS
        env["MKL_NUM_THREADS"] = "1"
        try:
            logf = open(os.path.join(out_dir, "reduce.log"), "w")
            self._proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=_HERE, env=env,
                start_new_session=True,
            )
            self._log(f"reducer pid={self._proc.pid} jobs={jobs} -> {out_dir} "
                      f"(CPU only; training continues now)")
        except Exception as exc:
            self._log(f"failed to launch reducer: {exc}")
            try:
                os.remove(self._marker(runner))
            except OSError:
                pass

    # ------------------------------------------------- report finished rounds
    def _drain(self, runner):
        """Pick up summaries the reducer finished since last epoch and log them
        onto the live training run. Cheap (a couple of stats + a json read)."""
        root = self._root(runner)
        if not os.path.isdir(root):
            return
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            summ = os.path.join(d, "summary.json")
            if name in self._logged or not os.path.isfile(summ):
                continue
            try:
                with open(summ) as f:
                    s = json.load(f)
            except Exception:
                continue
            self._logged.add(name)
            self._log(
                f"{name}: Dice={s.get('macro_dice', float('nan')):.4f} "
                f"VolW={s.get('volw_dice', float('nan')):.4f} "
                f"NSD={s.get('macro_nsd', float('nan')):.4f} "
                f"HD95={s.get('macro_hd95', float('nan')):.2f}mm "
                f"worst-subj={s.get('worst_subject_fg_dice', float('nan')):.4f} "
                f"fail={s.get('failure_rate', float('nan')):.3f}"
            )
            try:
                import wandb
                if wandb.run is not None:
                    payload = {f"real_eval/{k}": v for k, v in s.items()
                               if isinstance(v, (int, float))}
                    step = getattr(runner, "global_batch_step", None)
                    try:
                        if step:
                            wandb.log(payload, step=int(step))
                        else:
                            wandb.log(payload)
                    except Exception:
                        # e.g. wandb refuses a non-monotonic step -- fall back to
                        # an uncommitted point rather than losing the number.
                        wandb.log(payload)
            except Exception:
                pass

    # ---------------------------------------------------------------- driver
    def on_epoch_end(self, runner):
        self._local_epoch += 1
        cfg = self._cfg(runner)
        if not cfg.get("enabled", False):
            return
        rank, world = _rank_world()
        try:
            self._drain(runner)
        except Exception as exc:
            self._log(f"drain failed (ignored): {exc}")

        every = max(1, int(cfg.get("every_n_epochs", 10)))
        # Global (cross-restart) epoch, so every_n_epochs is not defeated by the
        # job being sliced into 4h chunks. See _counter_path().
        epoch = (self._base_epoch or 0) + self._local_epoch
        if rank == 0:
            try:
                tmp = self._counter_path(runner) + ".tmp"
                with open(tmp, "w") as f:
                    f.write(str(epoch))
                os.replace(tmp, self._counter_path(runner))
            except Exception as exc:
                self._log(f"could not persist epoch counter: {exc}")
        if epoch % every != 0:
            return

        marker = self._marker(runner)
        if os.path.exists(marker):
            # Same decision on every rank, no collective needed. Age-guarded so a
            # marker orphaned by a crash can only cost one round, not the run.
            try:
                age = time.time() - os.path.getmtime(marker)
            except OSError:
                age = 0.0
            if age < int(cfg.get("timeout_s", 1800)) + 600:
                self._log(f"epoch {epoch}: previous round still reducing "
                          f"({age:.0f}s), skipping")
                return
            self._log(f"epoch {epoch}: REDUCING marker is {age:.0f}s old -- "
                      f"treating as stale and proceeding")
            if rank == 0:
                try:
                    os.remove(marker)
                except OSError:
                    pass

        try:
            ids = self._subject_ids(runner, cfg)
            if not ids:
                self._log(f"epoch {epoch}: no subjects found, skipping")
                return
            my_ids = ids[rank::world]
            out_dir = os.path.join(self._root(runner), f"epoch_{epoch:05d}")
            spool = str(cfg.get("spool_dir") or os.path.join(out_dir, "spool"))
            os.makedirs(spool, exist_ok=True)
            if rank == 0:
                # Claim the round BEFORE inference so a crash mid-phase-1 does
                # not leave the next round racing this one.
                with open(marker, "w") as f:
                    f.write(str(epoch))
            if my_ids:
                self._infer_shard(runner, cfg, spool, my_ids)
            if rank == 0:
                self._launch_reducer(runner, cfg, out_dir, spool, world, len(ids))
        except Exception as exc:
            import traceback
            self._log(f"epoch {epoch}: FAILED (training continues): {exc}")
            traceback.print_exc(file=sys.stderr)
            if rank == 0:
                try:
                    os.remove(self._marker(runner))
                except OSError:
                    pass
