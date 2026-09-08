"""Numerical tests for the marginalized (group-space) aux loss terms in dice.py.
Run:  python test_group_marginalize.py        (CPU, no data, no GPU)"""
import ast, sys, textwrap
import numpy as np
import torch
import dice
from dice import (CEDiceLoss, GROUP_LUT_104_TO_18, GROUP_LUTS,
                  marginalize_probs, tversky_loss_from_probs)

torch.manual_seed(0)
ok = []
def check(name, cond, extra=""):
    ok.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  " + extra) if extra else ""))

print("1. GROUP_LUT_104_TO_18 vs predict_samples_siam.lut104_to_18()")
_here = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
src = open(__import__('os').path.join(_here, 'predict_samples_siam.py')).read()
tree = ast.parse(src)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lut104_to_18"]
ns = {"np": np}
exec(compile(ast.Module(body=fn, type_ignores=[]), "<lut>", "exec"), ns)
ref = ns["lut104_to_18"]()
mine = np.array(GROUP_LUT_104_TO_18, dtype=np.uint8)
check("length is 104", len(GROUP_LUT_104_TO_18) == 104, f"len={len(GROUP_LUT_104_TO_18)}")
check("identical to predict_samples_siam", np.array_equal(ref, mine),
      "" if np.array_equal(ref, mine) else f"differs at {np.nonzero(ref!=mine)[0][:10]}")
check("registry exposes it", GROUP_LUTS["lut104_to_18"] is GROUP_LUT_104_TO_18)
check("18 groups, all present", sorted(set(GROUP_LUT_104_TO_18)) == list(range(18)))
check("cortex parcels 1-68 all -> group 2", set(mine[1:69]) == {2})
check("CSF 93 -> group 0 (background)", mine[93] == 0)
check("CC 99-103 -> group 1 (WM)", set(mine[99:104]) == {1})

print("\n2. marginalize_probs")
C, G = 104, 18
lut = torch.tensor(GROUP_LUT_104_TO_18, dtype=torch.long)
logits = torch.randn(2, C, 6, 7, 8, dtype=torch.float64)
probs = torch.softmax(logits, dim=1)
gp = marginalize_probs(probs, lut, G)
check("shape", tuple(gp.shape) == (2, G, 6, 7, 8), str(tuple(gp.shape)))
ref_gp = torch.zeros_like(gp)
for c in range(C):
    ref_gp[:, lut[c]] += probs[:, c]
check("equals an explicit per-class sum", torch.allclose(gp, ref_gp, atol=1e-12))
M = torch.zeros(C, G, dtype=torch.float64); M[torch.arange(C), lut] = 1.0
check("equals einsum with the [C,G] matrix",
      torch.allclose(gp, torch.einsum('bk...,kc->bc...', probs, M), atol=1e-12))
check("group probs still sum to 1 over groups",
      torch.allclose(gp.sum(1), torch.ones_like(gp.sum(1)), atol=1e-12))
check("each class maps to exactly one group", bool((M.sum(1) == 1).all()))

print("\n3. backward compatibility: zero group weights == the old loss, bitwise")
tgt = torch.randint(0, C, (2, 6, 7, 8))
x = torch.randn(2, C, 6, 7, 8, dtype=torch.float32, requires_grad=True)
kw = dict(loss_weight=(0.5, 0.5), label_smoothing=0.01, generalized=True)
old = CEDiceLoss(**kw)                                   # no new kwargs at all
new_off = CEDiceLoss(**kw, group_lut=GROUP_LUT_104_TO_18) # lut given, weights 0
l_old = old(x, tgt); l_new = new_off(x, tgt)
check("loss identical bitwise", l_old.item() == l_new.item(),
      f"{l_old.item():.10f} vs {l_new.item():.10f}")
g_old = torch.autograd.grad(l_old, x, retain_graph=True)[0]
g_new = torch.autograd.grad(l_new, x, retain_graph=True)[0]
check("gradient identical bitwise", bool(torch.equal(g_old, g_new)),
      f"max|d|={float((g_old-g_new).abs().max()):.3e}")
check("_group_on() is False with zero weights", not new_off._group_on())
try:
    CEDiceLoss(**kw, group_tversky_weight=0.3)
    check("missing group_lut raises", False)
except ValueError:
    check("missing group_lut raises", True)

print("\n4. group targets == lut[targets], and the terms actually fire")
crit = CEDiceLoss(**kw, group_lut=GROUP_LUT_104_TO_18,
                  group_tversky_weight=0.30, group_tversky_alpha=0.6,
                  group_tversky_beta=0.4, group_tversky_classes=[2, 6],
                  group_cldice_weight=0.10, group_cldice_iters=2,
                  group_cldice_classes=[1, 3, 4, 5, 11, 12, 13],
                  log_terms=True)
check("_group_on() is True", crit._group_on())
l = crit(x, tgt)
check("loss finite", bool(torch.isfinite(l)))
check("loss differs from the control", abs(l.item() - l_old.item()) > 1e-6,
      f"{l.item():.6f} vs control {l_old.item():.6f}")
t = crit._terms
check("terms recorded", set(t) == {"ce", "dice", "group_tversky", "group_cldice", "total"},
      str(sorted(t)))
recon = (0.5 * t["ce"] + 0.5 * t["dice"] + 0.30 * t["group_tversky"]
         + 0.10 * t["group_cldice"])
check("weighted terms reconstruct the total",
      torch.allclose(recon, t["total"], atol=1e-6),
      f"{float(recon):.8f} vs {float(t['total']):.8f}")
g = torch.autograd.grad(l, x)[0]
check("gradient finite", bool(torch.isfinite(g).all()))
gt = lut[tgt]
check("lut[targets] has only group ids", int(gt.max()) < 18 and int(gt.min()) >= 0)

print("\n5. sign check: precision-favouring Tversky pushes a cortex FALSE POSITIVE down")
# one voxel of true background that the model calls cortex parcel 30
sh = (1, C, 4, 4, 4)
z = torch.full(sh, -6.0); z[:, 0] = 3.0                      # mostly background
z[0, 30, 2, 2, 2] = 6.0                                      # a cortex FP here
z.requires_grad_(True)
tg = torch.zeros((1, 4, 4, 4), dtype=torch.long)             # GT: all background
tv = CEDiceLoss(loss_weight=(0.0, 0.0), group_lut=GROUP_LUT_104_TO_18,
                group_tversky_weight=1.0, group_tversky_alpha=0.6,
                group_tversky_beta=0.4, group_tversky_classes=[2])
gr = torch.autograd.grad(tv(z, tg), z)[0]
check("d(loss)/d(FP cortex logit) > 0  (gradient descent lowers it)",
      float(gr[0, 30, 2, 2, 2]) > 0, f"grad={float(gr[0,30,2,2,2]):+.3e}")
check("the true-background logit of that voxel is pushed UP",
      float(gr[0, 0, 2, 2, 2]) < 0, f"grad={float(gr[0,0,2,2,2]):+.3e}")
# alpha > beta must punish FP harder than FN
def tv_loss(a, b, fp=True):
    y = torch.full(sh, -6.0); y[:, 0] = 3.0
    t2 = torch.zeros((1, 4, 4, 4), dtype=torch.long)
    if fp: y[0, 30, 2, 2, 2] = 6.0            # says cortex where GT is bg
    else:  t2[0, 2, 2, 2] = 30                # GT is cortex, model says bg
    p = torch.softmax(y, 1)
    gp = marginalize_probs(p, lut, G); gtt = lut[t2]
    return float(tversky_loss_from_probs(gp, gtt, classes=[2], alpha=a, beta=b))
check("alpha>beta: an FP costs more than an FN",
      tv_loss(0.6, 0.4, True) > tv_loss(0.6, 0.4, False),
      f"FP {tv_loss(0.6,0.4,True):.4f} > FN {tv_loss(0.6,0.4,False):.4f}")
# Well-posed asymmetry test: hold the prediction fixed and swap alpha/beta.
# (Comparing an FP scene to an FN scene at alpha==beta is NOT expected to tie --
# the two scenes carry different soft mass, which is what the earlier version of
# this check got wrong.)
check("FP-heavy scene: alpha>beta costs more than beta>alpha",
      tv_loss(0.7, 0.3, True) > tv_loss(0.3, 0.7, True),
      f"{tv_loss(0.7,0.3,True):.4f} > {tv_loss(0.3,0.7,True):.4f}")
check("FN-heavy scene: the ordering reverses",
      tv_loss(0.7, 0.3, False) < tv_loss(0.3, 0.7, False),
      f"{tv_loss(0.7,0.3,False):.4f} < {tv_loss(0.3,0.7,False):.4f}")
check("alpha=beta=0.5 reproduces soft Dice on the same scene",
      abs(tv_loss(0.5, 0.5, True) - tv_loss(0.5, 0.5, True)) < 1e-12)

print("\n6. cost: the group tensor is 18/104 of the class tensor")
big = 2 * C * 6 * 7 * 8; small = 2 * G * 6 * 7 * 8
check("element ratio == 18/104", abs(small / big - 18 / 104) < 1e-12,
      f"{small}/{big} = {small/big:.4f}")
print("\n%d/%d checks passed" % (sum(ok), len(ok)))
sys.exit(0 if all(ok) else 1)
