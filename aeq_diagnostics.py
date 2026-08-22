"""Offline diagnostics for AEQMeshNet (Sec. 8 of the design doc).

Implements:
  * Sec. 8.1 exact Schur complement S = (I - G_y) - G_z (I - F_z)^{-1} F_y,
    formed explicitly column by column (C_y inner linear solves).
    -> log cond(S), its spectrum, and ||G_z (I-F_z)^{-1} F_y||.
       If that last norm is ~0, the outer contour is DECORATIVE and the
       central claim of Sec. 1 is false.
  * rho(F_z) via power iteration, per input (Sec. 8.2 metric: mult should
    move rho across inputs; add should not).
  * Sec. 8.3 truncation bias: parameter-gradient cosine K_b=0 vs K_b=64.

Run on the training server (GPU recommended), e.g.:

    python aeq_diagnostics.py --ckpt logs/.../model.last.pth --cube 64

With no --ckpt it uses a random-init model (useful for wiring checks only).
Cost scales with C_y * inner-solve iters * one-f-eval; use --cube 48..96,
not 256.
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

from aeq_meshnet import AEQMeshNet


def _fixed_point(model, x):
    model.eval()
    with torch.no_grad():
        xin = model.stem(x)
        y0 = model._y0(xin)
        z, y = model._solve(xin, y0)
    return xin, z, y


def _jvp_f_z(model, z, y, xin, u):
    """J_f/dz . u (forward mode)."""
    def f(z_):
        return model._f(z_, y, xin)
    _, out = torch.func.jvp(f, (z,), (u,))
    return out


def _jvp_f_y(model, z, y, xin, e):
    def f(y_):
        return model._f(z, y_, xin)
    _, out = torch.func.jvp(f, (y,), (e,))
    return out


def _jvp_g_z(model, z, y, u):
    def g(z_):
        return model._g(y, z_)
    _, out = torch.func.jvp(g, (z,), (u,))
    return out


def _jac_g_y(model, z, y):
    def g(y_):
        return model._g(y_, z)
    return torch.autograd.functional.jacobian(g, y, vectorize=True)


def schur_report(model, x, inner_iters=40, tol=1e-6):
    """Sec. 8.1. Returns dict with cond(S), spectrum, coupling norm."""
    xin, z, y = _fixed_point(model, x)
    B, C_y = y.shape
    assert B == 1, "run with batch 1"
    dev = y.device

    Gy = _jac_g_y(model, z, y).reshape(C_y, C_y)   # d g / d y (B=1)

    cols = []
    for i in range(C_y):
        e = torch.zeros_like(y)
        e[0, i] = 1.0
        a = _jvp_f_y(model, z, y, xin, e)          # F_y e_i  (z-space)
        u = torch.zeros_like(a)
        for k in range(inner_iters):               # u <- F_z u + a
            u_next = _jvp_f_z(model, z, y, xin, u) + a
            rel = (u_next - u).norm() / (u.norm() + 1e-12)
            u = u_next
            if rel < tol:
                break
        b = _jvp_g_z(model, z, y, u)               # G_z (I-F_z)^{-1} F_y e_i
        cols.append(b.reshape(C_y))
    Bmat = torch.stack(cols, dim=1)                # (C_y, C_y)

    S = torch.eye(C_y, device=dev) - Gy - Bmat
    ev = torch.linalg.eigvals(S)
    sv = torch.linalg.svdvals(S)
    return {
        "coupling_norm": float(Bmat.norm()),           # THE number (Sec. 8.1)
        "coupling_absmax": float(Bmat.abs().max()),
        "cond_S": float(sv.max() / sv.min().clamp(min=1e-12)),
        "S_eig_real_min": float(ev.real.min()),
        "S_eig_real_max": float(ev.real.max()),
        "Gy_norm": float(Gy.norm()),
    }


def rho_f_z(model, x, iters=30):
    """Power-iteration estimate of rho(F_z) at the fixed point."""
    xin, z, y = _fixed_point(model, x)
    v = torch.randn_like(z)
    v /= v.norm()
    rho = float("nan")
    for _ in range(iters):
        Jv = _jvp_f_z(model, z, y, xin, v)
        rho = float(Jv.norm())
        v = Jv / (Jv.norm() + 1e-12)
    return rho


def truncation_bias(model, x, label, kb_ref=64):
    """Sec. 8.3: cosine(grad @ K_b=0, grad @ K_b=kb_ref)."""
    def grads(kb):
        model.zero_grad(set_to_none=True)
        model.train()
        old = dict(model.bw_cfg)
        model.bw_cfg["max_iter"] = max(kb, 1) if kb > 0 else 1
        model.bw_cfg["eps"] = 0.0 if kb > 0 else 1e9   # eps=inf => 1 iter = JFB
        out = model(x)
        F.cross_entropy(out, label).backward()
        model.bw_cfg.update(old)
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters()])
    g0, gref = grads(0), grads(kb_ref)
    cos = float(F.cosine_similarity(g0, gref, dim=0))
    ratio = float(g0.norm() / (gref.norm() + 1e-12))
    return {"cos_K0_vs_ref": cos, "norm_ratio": ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="state_dict checkpoint (optional)")
    ap.add_argument("--cube", type=int, default=48)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--classes", type=int, default=18)
    ap.add_argument("--cy", type=int, default=64)
    ap.add_argument("--coupling", default="mult")
    ap.add_argument("--dilations", default="1,4,16")
    ap.add_argument("--n-inputs", type=int, default=4,
                    help="inputs for the rho(F_z) variance check")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aeq = {
        "C_y": args.cy,
        "coupling": args.coupling,
        "dilations": [int(d) for d in args.dilations.split(",")],
        "solver": {"max_iter": 40, "eps": 1e-5},
        "backward": {"max_iter": 64, "eps": 1e-7},
    }
    model = AEQMeshNet(1, args.classes, args.channels, aeq=aeq).to(dev)
    model.act_name = "elu"
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=dev)
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.ckpt}: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", file=sys.stderr)

    torch.manual_seed(0)
    xs = [torch.randn(1, 1, *(args.cube,) * 3, device=dev)
          for _ in range(args.n_inputs)]

    print("== rho(F_z) across inputs (Sec. 8.2: mult should move it) ==")
    rhos = [rho_f_z(model, x) for x in xs]
    print(f"   rho = {['%.4f' % r for r in rhos]}  "
          f"mean={sum(rhos)/len(rhos):.4f} "
          f"spread={max(rhos)-min(rhos):.4f}")

    print("== Schur complement (Sec. 8.1) ==")
    rep = schur_report(model, xs[0])
    for k, v in rep.items():
        print(f"   {k:18s} {v:.6g}")
    if rep["coupling_norm"] < 1e-4:
        print("   WARNING: ||G_z (I-F_z)^{-1} F_y|| ~ 0 -> outer contour "
              "is DECORATIVE (Sec. 1 claim fails on this checkpoint).")

    print("== truncation bias (Sec. 8.3) ==")
    lab = torch.randint(0, args.classes, (1, *(args.cube,) * 3), device=dev)
    rep = truncation_bias(model, xs[0], lab)
    for k, v in rep.items():
        print(f"   {k:18s} {v:.6g}")


if __name__ == "__main__":
    main()
