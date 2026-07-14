"""Probe: does compiling CEDiceLoss (a) stay numerically identical and (b) run
faster? The Dice loss is ~18% of GPU time, dominated by two scatter_add_ over
16.7M voxels into 18 bins (atomic-contention bound) plus an fp32 log_softmax,
all running EAGER outside the compiled model graph. Inductor should fuse the
log_softmax->exp->gather chain and generate a privatized (contention-free)
scatter. This isolates that question -- standalone, single GPU, seconds.

Mirrors compile_probe.py. Run on the server:
    source /trdapps/linux-x86_64/envs/plis_venv/torch/bin/activate
    python loss_compile_probe.py          # PROBE_CUBE=256 by default
"""
import os
import time
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = False   # surface any fallback

from dice import CEDiceLoss

DEV = "cuda"
C = 18                                          # n_classes (siam18)
CUBE = int(os.environ.get("PROBE_CUBE", "256"))
DT = torch.bfloat16                             # matches amp_dtype

torch.manual_seed(0)

def make_loss():
    # same construction as get_criterion: class_weight = [off_brain] + 1.0*...,
    # generalized Dice, label_smoothing 0.01, 0.5/0.5 CE/Dice split.
    cw = torch.tensor([0.2] + [1.0] * (C - 1), device=DEV)
    return CEDiceLoss(loss_weight=(0.5, 0.5), class_weight=cw,
                      label_smoothing=0.01, generalized=True).to(DEV)

logits = torch.randn(1, C, CUBE, CUBE, CUBE, device=DEV, dtype=DT)
labels = torch.randint(0, C, (1, CUBE, CUBE, CUBE), device=DEV)

def run(crit, x_src):
    x = x_src.detach().clone().requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=DT):
        loss = crit(x, labels)
    loss.backward()
    return float(loss), x.grad.detach()

# --- eager reference ---
crit_e = make_loss()
le, ge = run(crit_e, logits)

# --- compiled (same weights) ---
crit_c = make_loss()
crit_c.load_state_dict(crit_e.state_dict())
crit_c.compile(mode="default")
for _ in range(3):                              # first call triggers compile
    lc, gc = run(crit_c, logits)

print(f"cube={CUBE}  dtype={DT}")
print(f"eager    loss = {le:.6f}")
print(f"compiled loss = {lc:.6f}")
print(f"loss |diff|   = {abs(le - lc):.3e}   (rel {abs(le-lc)/max(abs(le),1e-9):.2e})")
print(f"grad max|diff|= {(ge - gc).abs().max().item():.3e}")
print("-> numerically safe if rel loss diff and grad diff are ~1e-3 or smaller (bf16)")

def bench(crit, n=30):
    for _ in range(5):
        run(crit, logits)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        run(crit, logits)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0

te = bench(crit_e)
tc = bench(crit_c)
print(f"\neager    fwd+bwd: {te:6.2f} ms/call")
print(f"compiled fwd+bwd: {tc:6.2f} ms/call   ({te / tc:.2f}x)")
