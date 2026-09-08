"""Sanity checks for the refactored losses in dice.py.

Run on the training server:
    source ~/venv/torch/bin/activate && python _verify_dice.py

Checks:
  1. vectorized DiceLoss == original per-class loop (forward + grad)
  2. fused CEDiceLoss CE-only term == torch.nn.CrossEntropyLoss(weight, smoothing)
  3. fused CEDiceLoss == w_ce*CE + w_dice*DiceLoss  (plain and generalized)
All "diff" numbers should be ~1e-5 or smaller (fp32 roundoff).
"""
import torch
import torch.nn.functional as F
from dice import DiceLoss, CEDiceLoss


def old_dice(inputs, targets, smooth=1):
    inputs = F.softmax(inputs, dim=1)
    dl = 0.0
    for c in range(inputs.size(1)):
        tf = (targets == c).float()
        pf = inputs[:, c]
        inter = (pf * tf).sum(dim=(1, 2, 3))
        tot = (pf + tf).sum(dim=(1, 2, 3))
        dl += 1 - (2.0 * inter + smooth) / (tot + smooth)
    return dl.mean()


torch.manual_seed(0)
B, C = 2, 7
x = torch.randn(B, C, 16, 16, 16, dtype=torch.float64, requires_grad=True)
y = torch.randint(0, C, (B, 16, 16, 16))
cw = torch.cat([torch.tensor([0.3]), torch.ones(C - 1)]).double()
eps = 0.01


def grad_of(fn, *inputs):
    xi = x.detach().clone().requires_grad_(True)
    out = fn(xi)
    out.backward()
    return out.item(), xi.grad


# 1. vectorized vs original loop -------------------------------------------
v, gv = grad_of(lambda z: DiceLoss()(z, y))
o, go = grad_of(lambda z: old_dice(z, y))
print(f"[1] DiceLoss vs loop      fwd diff {abs(v-o):.2e}  grad diff {(gv-go).abs().max():.2e}")

# 2. fused CE-only vs nn.CrossEntropyLoss ----------------------------------
ce_ref = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=eps)
fused_ce = CEDiceLoss(loss_weight=(1.0, 0.0), class_weight=cw, label_smoothing=eps)
v, gv = grad_of(lambda z: fused_ce(z, y))
o, go = grad_of(lambda z: ce_ref(z, y))
print(f"[2] fused CE vs nn.CE     fwd diff {abs(v-o):.2e}  grad diff {(gv-go).abs().max():.2e}")

# 3. fused combined vs separate CE + Dice ----------------------------------
for gen in (False, True):
    dice_ref = DiceLoss(generalized=gen)
    fused = CEDiceLoss(loss_weight=(0.5, 0.5), class_weight=cw,
                       label_smoothing=eps, generalized=gen)
    ref = lambda z: 0.5 * ce_ref(z, y) + 0.5 * dice_ref(z, y)
    v, gv = grad_of(lambda z: fused(z, y))
    o, go = grad_of(ref)
    tag = "generalized" if gen else "plain      "
    print(f"[3] fused {tag} vs CE+Dice  fwd diff {abs(v-o):.2e}  grad diff {(gv-go).abs().max():.2e}")
