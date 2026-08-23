"""Print the forward solve's residual trajectory for a real checkpoint.

Answers the one question the wandb scalars cannot: WHY does the solve exit on
the iteration cap? There are only three possibilities, and they need opposite
responses:

  1. rho >= 1  -> the map is NOT contracting. The residual grows or stalls
                  high. No solver setting fixes this; lower m_max further.
  2. bf16 floor-> the residual falls to ~2-5e-3 and stops, because f(z)-z is a
                  difference of nearly-equal bf16 numbers. Already converged;
                  raise solver.eps above the measured floor.
  3. too slow  -> the residual is still falling at sweep 12. Raise
                  solver.max_iter (or lower m_max to speed contraction).

Runs plain damped Jacobi iteration (exactly what window_m=1 does), in bf16
autocast and in fp32 side by side, so the floor is visible as the gap.

    python aeq_solvetrace.py --ckpt /path/model.last.pth --cube 128 --sweeps 40

Without --ckpt it uses a random init (wiring check only -- an untrained map
tells you nothing about the trained one).
"""

import argparse
import contextlib

import torch

from aeq_meshnet import AEQMeshNet, _block_norm, _state_norm


def trace(model, x, sweeps, amp_dtype=None):
    dev = x.device
    ctx = (torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype
           else contextlib.nullcontext())
    out = []
    with torch.no_grad(), ctx:
        xin = model.stem(x)
        y = model._y0(xin)
        z = torch.zeros(x.shape[0], model.C, *x.shape[2:],
                        device=dev, dtype=xin.dtype)
        for k in range(sweeps):
            fz, fy = model._H(z, y, xin)
            rel = float(_block_norm(fz - z, fy - y) / _state_norm(fz, fy))
            out.append(rel)
            beta = model.sol_cfg["beta"]
            z = beta * fz + (1 - beta) * z
            y = beta * fy + (1 - beta) * y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--cube", type=int, default=128)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--classes", type=int, default=18)
    ap.add_argument("--cy", type=int, default=64)
    ap.add_argument("--dilations", default="1,3,9,27")
    ap.add_argument("--m-max", type=float, default=0.5)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--act", default="elu", help="softplus (phase A/B) | elu (C/D)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aeq = {"C_y": args.cy, "m_max": args.m_max,
           "dilations": [int(d) for d in args.dilations.split(",")],
           "solver": {"max_iter": args.sweeps, "eps": 0.0, "window_m": 1,
                      "check_every": 1, "compile_step": False},
           "loss": {"gamma_jac": 0.0}}
    model = AEQMeshNet(1, args.classes, args.channels, aeq=aeq).to(dev)
    model.act_name = args.act
    model.eval()

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=dev)
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.ckpt}\n  {len(missing)} missing, "
              f"{len(unexpected)} unexpected key(s)")
        if len(unexpected) > 4:
            print(f"  WARNING many unexpected keys -- config mismatch? "
                  f"{sorted(unexpected)[:4]}")
    else:
        print("NO --ckpt: random init, wiring check only")

    torch.manual_seed(0)
    x = torch.randn(1, 1, *(args.cube,) * 3, device=dev)

    rows = {}
    for name, dt in (("bf16", torch.bfloat16), ("fp32", None)):
        try:
            rows[name] = trace(model, x, args.sweeps, dt)
        except Exception as e:
            print(f"  {name} trace failed: {type(e).__name__}: {e}")

    print(f"\n{'sweep':>5} {'bf16 rel_res':>14} {'fp32 rel_res':>14}")
    for k in range(args.sweeps):
        b = rows.get("bf16", [None] * args.sweeps)[k]
        f = rows.get("fp32", [None] * args.sweeps)[k]
        print(f"{k:>5} {('%.3e' % b) if b is not None else '-':>14} "
              f"{('%.3e' % f) if f is not None else '-':>14}")

    # rho at the last iterate
    with torch.no_grad():
        xin = model.stem(x)
        y = model._y0(xin)
        z, y = model._solve(xin, y)
    r = model._rho_estimate(z, y, xin, iters=25)
    print(f"\nrho(dF/dz) at the iterate: {r:.4f}"
          + ("   <-- >= 1: NOT CONTRACTING, lower m_max" if r >= 1.0 else ""))

    for name in ("bf16", "fp32"):
        v = rows.get(name)
        if not v:
            continue
        tail = v[-5:]
        falling = tail[-1] < tail[0] * 0.9
        floor = min(v)
        print(f"{name}: min={floor:.3e} last={v[-1]:.3e} "
              + ("still falling at the cap -> raise max_iter"
                 if falling else f"plateaued -> set solver.eps above {floor:.1e}"))


if __name__ == "__main__":
    main()
