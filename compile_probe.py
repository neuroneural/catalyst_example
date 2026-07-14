"""Isolate the torch.compile behavior of the GN MeshNet trunk.

Why: in full DDP training, `[fast] in-place compile` prints, no graph breaks are
logged, yet the profiler shows only eager ATen kernels (native_group_norm,
cudnn_convolution) and zero triton_* fused kernels. That pattern == Dynamo is
silently falling back to eager (torch._dynamo.config.suppress_errors defaults to
True, which swallows inductor compile errors). This probe compiles the exact same
model OUTSIDE Catalyst/DDP with suppress_errors=False so the real error surfaces,
and reports whether triton kernels were actually generated.

Run on the server (small cube, ~seconds):
    source /trdapps/linux-x86_64/envs/plis_venv/torch/bin/activate
    TORCH_LOGS="graph_breaks,recompiles" python compile_probe.py
Add TORCH_LOGS="output_code" to dump the generated triton (or confirm none is).
"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn as nn
import torch._dynamo
from torch._dynamo.utils import counters

# Surface inductor/dynamo errors instead of silently running eager.
torch._dynamo.config.suppress_errors = False
torch._dynamo.reset()

from meshnet_gn import enMesh_checkpoint as Mesh

DEV = "cuda"
CFG = "./modelAE_hdc_deep.json"      # same file the training config points at
CUBE = 64                             # small: compile behavior is shape-agnostic

model = Mesh(in_channels=1, n_classes=18, channels=16,
             config_file=CFG, affine=True).to(DEV)
model.use_checkpoint = False          # matches the fast/turbo config
model.train()

x = torch.randn(1, 1, CUBE, CUBE, CUBE, device=DEV)
y = torch.randint(0, 18, (1, CUBE, CUBE, CUBE), device=DEV)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print(">> compiling (in-place, mode=default)...", flush=True)
model.compile(mode="default")

def step(i):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = crit(out, y)
    loss.backward()
    opt.step()
    return float(loss)

try:
    for i in range(3):            # first step triggers compile; 2-3 confirm steady state
        L = step(i)
        print(f">> step {i} ok  loss={L:.4f}", flush=True)
    print(">> forward/backward ran under compile without raising")
except Exception as exc:
    import traceback
    print(">> COMPILE RAISED (this is the real reason it falls back to eager):\n",
          flush=True)
    traceback.print_exc()

print("\n==== dynamo/inductor tallies ====")
print("unique_graphs :", counters["stats"].get("unique_graphs", 0))
print("graph_breaks  :", dict(counters["graph_break"]))
print("inductor      :", dict(counters["inductor"]))
print("If unique_graphs>0 and graph_breaks is empty and no exception above,",
      "compile is working in isolation -> the problem is the DDP wrapping.")
