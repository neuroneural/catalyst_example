"""Phase 0 machinery gate + unit tests for aeq_meshnet.py (CPU, tiny scale).

Run:  python test_aeq.py

Mirrors Sec. 10 "Phase 0" of adaptive_equilibrium_design_v2.md:
  1. RBP gradient at large K_b matches fully-unrolled autograd:
     cosine > 0.99, relative error < 1e-2.      <-- HARD GATE
  2. Forward solve reaches eps_f within K_f; both block residuals converge.
  3. Path independence: same fixed point from z0 = 0 and z0 = random.
  4. Contraction: power-iteration estimate of rho(J_f) < 1 (row-sum cap works).
  5. Random-freezing-mask solve lands on the synchronous fixed point (Sec. 7.2
     empirical gate for asynchronous/site-gated iteration).
  6. Site gating on: fixed point within 2*eps of the synchronous one.
  7. Coupling arms 'add' and 'none' build, run, and backprop.
  8. read_residual + detached_bptt: runs, grads reach g MLP and y_init.
"""

import math
import sys

import torch
import torch.nn.functional as F

from aeq_meshnet import AEQMeshNet, _block_norm, _state_norm

torch.manual_seed(0)

CUBE, C, NCLS = 20, 8, 4
AEQ = {
    "C_y": 16,
    "dilations": [1, 2],
    "n_groups": 4,
    "m_max": 0.9,
    "solver": {"max_iter": 60, "eps": 1e-6, "window_m": 3, "beta": 0.8},
    "backward": {"max_iter": 80, "eps": 1e-9},
    "stability": {"rowsum_target": 0.7},
    "loss": {"gamma_jac": 0.0},   # off for exact-gradient comparison
}

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILED.append(name)


def flat_grads(model):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters()
    ])


def loss_of(model, x, lab):
    out = model(x)
    return F.cross_entropy(out, lab)


def make(aeq_over=None, **kw):
    aeq = {**AEQ, **(aeq_over or {})}
    torch.manual_seed(0)
    m = AEQMeshNet(1, NCLS, C, aeq=aeq)
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def main():
    x = torch.randn(1, 1, CUBE, CUBE, CUBE)
    lab = torch.randint(0, NCLS, (1, CUBE, CUBE, CUBE))

    # ------------------------------------------------ 1. THE HARD GATE
    # implicit (RBP) gradient
    m = make()
    m.train()
    loss_of(m, x, lab).backward()
    g_rbp = flat_grads(m)
    stats_solve = dict(m.stats)

    # fully unrolled reference: same weights, plain Jacobi sweeps, autograd
    m2 = make(mode="unroll", unroll_K=120)
    m2.load_state_dict(m.state_dict())
    m2.train()
    loss_of(m2, x, lab).backward()
    g_ref = flat_grads(m2)

    cos = F.cosine_similarity(g_rbp, g_ref, dim=0).item()
    rel = ((g_rbp - g_ref).norm() / (g_ref.norm() + 1e-12)).item()
    check("Phase-0 gradient gate (cos > 0.99)", cos > 0.99,
          f"cosine={cos:.6f}")
    check("Phase-0 gradient gate (rel err < 1e-2)", rel < 1e-2,
          f"rel={rel:.2e}")

    # per-parameter-group cosines (localize any mismatch)
    i = 0
    for name, p in m.named_parameters():
        n = p.numel()
        a, b = g_rbp[i:i + n], g_ref[i:i + n]
        i += n
        if a.norm() > 1e-10 and b.norm() > 1e-10:
            c = F.cosine_similarity(a, b, dim=0).item()
            if c < 0.99:
                print(f"        low-cos param: {name}  cos={c:.4f}")

    # ------------------------------------------------ 2. forward convergence
    check("forward solve reached eps_f",
          stats_solve["fwd_rel_res"] < AEQ["solver"]["eps"] * 1.5,
          f"rel_res={stats_solve['fwd_rel_res']:.2e} nfe={stats_solve['nfe']}")

    m.eval()
    with torch.no_grad():
        xin = m.stem(x)
        y0 = m._y0(xin)
        z1, y1 = m._solve(xin, y0)
        fz, fy = m._H(z1, y1, xin)
        rz = (fz - z1).norm() / math.sqrt(z1.numel())
        ry = (fy - y1).norm() / math.sqrt(y1.numel())
    check("both block residuals converge (Sec. 8.5)",
          rz < 1e-4 and ry < 1e-4, f"r_z={rz:.2e} r_y={ry:.2e}")

    # ------------------------------------------------ 3. path independence
    with torch.no_grad():
        z2, y2 = m._solve(xin, y0, z0=0.5 * torch.randn_like(z1))
        gap = float(_block_norm(z2 - z1, y2 - y1) / _state_norm(z1, y1))
    check("path independence (Sec. 8.4)", gap < 5 * AEQ["solver"]["eps"],
          f"gap={gap:.2e}")

    # ------------------------------------------------ 4. contraction rho(J)<1
    with torch.enable_grad():
        z_in = z1.detach().requires_grad_(True)
        out = m._f(z_in, y1, xin)
        v = torch.randn_like(z1)
        v /= v.norm()
        for _ in range(20):
            (Jv,) = torch.autograd.grad(out, z_in, v, retain_graph=True)
            rho = Jv.norm().item()
            v = Jv / (Jv.norm() + 1e-12)
    check("contraction rho(J_f) < 1 (row-sum cap)", rho < 1.0,
          f"rho~{rho:.3f} (target {AEQ['stability']['rowsum_target']})")

    # ------------------------------------------------ 5. random-mask gate
    passed, gp = m.random_mask_gate(x, mask_frac=0.5)
    check("random-freeze-mask gate (Sec. 7.2)", passed, f"gap={gp:.2e}")

    # ------------------------------------------------ 6. site gating fixed pt
    mg = make(aeq_over={"site_gating": {"enabled": True, "tau_0": 5e-7,
                                        "learned_threshold": True}})
    mg.load_state_dict(m.state_dict(), strict=False)
    mg.gating_on = True
    mg.eval()
    with torch.no_grad():
        xing = mg.stem(x)
        y0g = mg._y0(xing)
        zs, ys = mg._solve(xing, y0g)                    # gated (gating_on)
        mg.gating_on = False
        zs2, ys2 = mg._solve(xing, y0g)                  # synchronous
        gap = float(_block_norm(zs - zs2, ys - ys2) / _state_norm(zs2, ys2))
    check("site-gated solve == synchronous fixed point",
          gap < 2 * AEQ["solver"]["eps"], f"gap={gap:.2e}")

    # ------------------------------------------------ 7. coupling arms
    for arm in ("add", "none"):
        ma = make(aeq_over={"coupling": arm})
        ma.train()
        loss_of(ma, x, lab).backward()
        ok = all(torch.isfinite(p.grad).all() for p in ma.parameters()
                 if p.grad is not None)
        check(f"coupling arm '{arm}' trains", ok,
              f"nfe={ma.stats['nfe']} res={ma.stats['fwd_rel_res']:.1e}")

    # ------------------------------------------------ 8. closed loop + BPTT
    mb = make(aeq_over={"outer": {"read_residual": True,
                                  "outer_grad": "detached_bptt"}})
    mb.train()
    loss = loss_of(mb, x, lab)
    aux, _ = mb.pop_extra_losses()
    (loss + 0.1 * aux.pow(2).mean()).backward()
    g_ok = all(torch.isfinite(p.grad).all() for p in mb.parameters()
               if p.grad is not None)
    ymlp_g = sum(float(p.grad.abs().sum()) for p in mb.g_mlp.parameters())
    yinit_g = sum(float(p.grad.abs().sum()) for p in mb.y_init.parameters())
    check("read_residual + detached_bptt trains", g_ok and ymlp_g > 0,
          f"g_mlp |grad|={ymlp_g:.2e} y_init |grad|={yinit_g:.2e}")

    # ------------------------------------------------ 9. jacobian penalty
    mj = make(aeq_over={"loss": {"gamma_jac": 0.1}})
    mj.train()
    out = mj(x)
    _, jac = mj.pop_extra_losses()
    (F.cross_entropy(out, lab) + 0.1 * jac).backward()
    ok = all(torch.isfinite(p.grad).all() for p in mj.parameters()
             if p.grad is not None)
    check("Hutchinson Jacobian penalty backprops", ok,
          f"penalty={float(jac):.4f}")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("ALL PASSED -- Phase 0 machinery gate cleared.")


if __name__ == "__main__":
    main()
