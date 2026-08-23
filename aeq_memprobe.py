"""Per-phase GPU memory breakdown for AEQMeshNet. Answers "is it a leak?"
without waiting on a cluster job.

Runs N training steps of the real model+loss at a chosen cube size in each
phase, and reports for every step:

    alloc   = torch.cuda.memory_allocated()      live tensors RIGHT NOW
    peak    = torch.cuda.max_memory_allocated()  high-water mark this step
    reserved= torch.cuda.memory_reserved()       caching-allocator pool
                                                 (this is what nvitop shows)

How to read it:
  * ALLOC climbing step over step  -> real leak (something retains graphs).
  * ALLOC flat, RESERVED >> peak   -> fragmentation / allocator caching, not
                                      a leak. Try PYTORCH_CUDA_ALLOC_CONF=
                                      expandable_segments:True.
  * PEAK near device capacity      -> genuinely needs that much; the next
                                      resolution will OOM (memory scales ~8x
                                      per doubling of the cube edge).

Run (one GPU is enough; no Mongo/wandb involved):

    python aeq_memprobe.py --cube 128 --channels 16 --classes 18 --steps 6
    python aeq_memprobe.py --cube 256 --steps 3          # will it fit at all?
    python aeq_memprobe.py --cube 128 --phase B --steps 8
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from aeq_meshnet import AEQMeshNet

G = 1024 ** 3

PHASES = {   # (mode, act) per design Sec. 7.1
    "A": ("unroll", "softplus"),
    "B": ("solve", "softplus"),
    "C": ("solve", "elu"),
    "D": ("solve", "elu"),
}


def probe(args, phase):
    mode, act = PHASES[phase]
    aeq = {
        "C_y": args.cy,
        "dilations": [int(d) for d in args.dilations.split(",")],
        "solver": {"max_iter": args.max_iter, "eps": 1e-3,
                   "compile_step": bool(args.compile_step),
                   "check_every": 3},
        "backward": {"max_iter": args.bwd_iter, "eps": 1e-4},
        "site_gating": {"enabled": phase == "D", "tau_0": 5e-4},
        "loss": {"gamma_jac": args.gamma_jac},
        "phases": {"unroll_K": args.unroll_k},
    }
    dev = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AEQMeshNet(1, args.classes, args.channels, aeq=aeq).to(dev)
    model.mode, model.act_name = mode, act
    model.gating_on = (phase == "D")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n_par = sum(p.numel() for p in model.parameters())
    after_model = torch.cuda.memory_allocated() / G
    print(f"\n=== phase {phase} (mode={mode}, act={act}, gating={phase=='D'}) "
          f"cube={args.cube} C={args.channels} classes={args.classes} ===")
    print(f"  params {n_par/1e6:.3f}M -> {after_model:.3f} GiB resident")
    print(f"  {'step':>4} {'alloc':>9} {'peak':>9} {'reserved':>9}  "
          f"{'d_alloc':>9}  nfe")

    x = torch.randn(1, 1, *(args.cube,) * 3, device=dev)
    lab = torch.randint(0, args.classes, (1, *(args.cube,) * 3), device=dev)
    prev_alloc, rows = None, []
    amp = torch.bfloat16 if args.bf16 else None

    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        model.collect_stats = False
        if amp is not None:
            with torch.autocast("cuda", dtype=amp):
                out = model(x)
                loss = F.cross_entropy(out.float(), lab)
        else:
            out = model(x)
            loss = F.cross_entropy(out, lab)
        aux, jac = model.pop_extra_losses()
        if aux is not None:
            loss = loss + 0.1 * F.cross_entropy(
                aux, torch.zeros(1, dtype=torch.long, device=dev))
        if jac is not None:
            loss = loss + 0.1 * jac
        loss.backward()
        opt.step()

        a = torch.cuda.memory_allocated() / G
        p = torch.cuda.max_memory_allocated() / G
        r = torch.cuda.memory_reserved() / G
        d = "" if prev_alloc is None else f"{a - prev_alloc:+.4f}"
        print(f"  {s:>4} {a:>9.3f} {p:>9.3f} {r:>9.3f}  {d:>9}  "
              f"{model.stats.get('nfe','-')}")
        rows.append(a)
        prev_alloc = a

    # verdict: is live memory growing after the first two warmup steps?
    tail = rows[2:] if len(rows) > 3 else rows[1:]
    drift = (tail[-1] - tail[0]) if len(tail) > 1 else 0.0
    verdict = ("LEAK: live tensors grew {:+.3f} GiB over {} steps"
               .format(drift, len(tail) - 1) if drift > 0.01 else
               "no leak: live tensors flat after warmup ({:+.4f} GiB)"
               .format(drift))
    print(f"  -> {verdict}")
    print(f"  -> peak {torch.cuda.max_memory_allocated()/G:.3f} GiB, "
          f"reserved {torch.cuda.memory_reserved()/G:.3f} GiB, "
          f"device {torch.cuda.get_device_properties(0).total_memory/G:.1f} GiB")
    del model, opt, x, lab
    torch.cuda.empty_cache()
    return {"phase": phase, "peak_gb": p, "reserved_gb": r, "drift_gb": drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, default=128)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--classes", type=int, default=18)
    ap.add_argument("--cy", type=int, default=64)
    ap.add_argument("--dilations", default="1,3,9,27")
    ap.add_argument("--phase", default="all", help="A|B|C|D|all")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--bwd-iter", type=int, default=32)
    ap.add_argument("--unroll-k", type=int, default=5)
    ap.add_argument("--gamma-jac", type=float, default=0.1)
    ap.add_argument("--compile-step", type=int, default=0,
                    help="1 to include torch.compile (adds warmup noise)")
    ap.add_argument("--bf16", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    phases = list(PHASES) if args.phase == "all" else [args.phase.upper()]
    out = []
    for ph in phases:
        try:
            out.append(probe(args, ph))
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  phase {ph}: OOM -- {str(e)[:120]}")
            torch.cuda.empty_cache()
            out.append({"phase": ph, "oom": True})
    print("\nsummary:", json.dumps(out))


if __name__ == "__main__":
    main()
