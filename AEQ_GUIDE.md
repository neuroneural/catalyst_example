# Adaptive Equilibrium MeshNet — design analysis & training guide

Companion to `adaptive_equilibrium_design_v2.md` (v2.1). This is the Variant M
implementation for 18-class segmentation, wired into the existing
`curriculum_training_fast.py` stack.

## TL;DR

```bash
source ~/venv/torch/bin/activate
python test_aeq.py                       # Phase 0 machinery gate (CPU, ~2 min)
python curriculum_training_aeq.py --config-dir=conf --config-name=aeq_siam18_16ch
```

Phase 0 has already been run on the delivered code: **RBP gradient vs.
fully-unrolled autograd: cosine 1.000000, relative error 3e-7**; forward
converges in ~13 NFE to 5e-7; path independence, contraction (ρ≈0.36 under a
0.7 row-sum target), random-mask gate, site-gating fixed-point equality, and
the add/none arms all pass.

## Files

| file | role |
|---|---|
| `aeq_meshnet.py` | the model: joint solver, RBP backward, stability caps, gating |
| `curriculum_training_aeq.py` | entrypoint; subclasses `FastRunner`, adds phases, L_aux, γ‖F_z‖², the §7.2 empirical gate, `aeq/*` wandb logging |
| `conf/aeq_siam18_16ch.yaml` | 18-class, 16-channel config (synth18 wirehead data, same endpoints as the siam18 runs) |
| `test_aeq.py` | Phase 0 gate + unit tests (design §10) — run after ANY solver edit |
| `aeq_diagnostics.py` | §8.1 Schur complement, §8.2 ρ(F_z) spread, §8.3 truncation bias — offline, on a checkpoint |

---

## 1. Design verdict

The design is sound and was implemented essentially as written: RBP (Alg. 1)
with block-wise stopping, ∞-norm row-sum stability, block-normalized Anderson
residual (§5.1 — implemented exactly; it is the easiest silent failure),
detached per-site gating with reactivation, the four-phase warm-up, and the
mandatory random-mask empirical gate before gating is enabled. Two findings
materially change the architecture, and a handful of smaller fixes were needed
to make the pseudocode real. Everything below is config-switchable so each
deviation can be measured rather than trusted.

### 1.1 Major — gate placement vs. GroupNorm (quality)

§3.4 orders the block `m(y)·(W*z) → +x → GroupNorm → act`. GroupNorm then
standardizes each group of channels — and with C=16 / 8 groups, a group is a
*pair*. A per-channel multiplicative gate applied immediately before a
per-pair standardization is mostly **divided back out by the group statistics**:
what survives is only the within-pair ratio and the interaction with the `+x`
shift. The doc's central mechanism — `y` moving `∂f/∂z`'s spectrum — is
cancelled at exactly the point it is supposed to act, and the §8.2 headline
ablation would be biased toward "multiplicative coupling does nothing."

Fix (default `gate_position: post_norm`):

```
z' = act( m(y) ⊙ GN(W*z) + x )
```

Now `J = D_act · diag(m) · J_GN · J_conv`: the gate scales Jacobian rows
directly, so both the intended effect and the row-sum bound are clean.
`gate_position: pre_norm` reproduces §3.4 literally for the ablation.

### 1.2 Major — single dilation d=4 recreates the MDEQ receptive-field bound (efficiency)

§3.5 correctly identifies MDEQ's dominant cost: `f` must be applied until the
stacked receptive field covers the input. But M-A's single 3³ conv at d=4
grows the RF by only 8 voxels per sweep → **~32 sweeps to span a 256 cube**,
while the config caps `K_f = 24`. The truncated solve never sees the whole
volume — the exact failure §3.5 quotes.

Default here: a weight-tied **ternary micro-stack** `dilations: [1, 3, 9, 27]`
applied every sweep — RF grows 80/sweep, 4 sweeps for full 256³ coverage, and
Anderson history (the memory item) is unchanged. This *is* Bet 2 in sharpened
form. The specific set carries the `modelAE_hdc_deep` lesson over correctly:
with weight tying, the composed dilation schedule is *the per-sweep set
repeated K times*, so the no-gridding-holes property must hold within the set
itself — and 1,3,9,27 with 3³ kernels tiles exactly (each stage's taps land
on lattices fully covered by the previous stage's footprint), giving a
hole-free per-sweep RF at every sweep. Reach and the ramp's up-down symmetry,
by contrast, come free from iteration, so the 13-layer ramp itself is not
needed. `dilations: [4]` recovers doc-literal M-A;
`dilations: [1,3,5,7,13,19,31,19,13,7,5,3,1]` iterates the existing explicit
schedule as the equilibrium map (ablation 8; drop `solver.max_iter` to ~6).
The per-layer row-sum target is `rowsum_target^(1/L)` so the composed product
keeps the same bound.

### 1.3 Moderate fixes

1. **§5.2 safeguard doubles NFE as written** — testing the Anderson candidate
   needs an extra evaluation of `H` every sweep. Implemented retrospectively:
   accept the candidate, and if the *next* sweep's (already paid-for) residual
   increased, revert to the stored plain step and truncate history. Same
   protection, zero extra evaluations (logged as `aeq/anderson_rejects`).
2. **§6.2 pseudocode bug** — `n_z = λ_z_next − λ_z` reads `λ_next` before it
   exists; increments are computed from `v + g` before assignment.
3. **`y_init` never trains under `outer_grad: implicit`** — the fixed point is
   init-independent, so implicit differentiation sends exactly zero gradient
   to `y_0 = tanh(Linear(pool(x)))`. It stays at random init (a harmless
   deterministic seed). Under `detached_bptt` it gets a `(1−α)^K`-vanishing
   trajectory gradient (measured ~1e-20 at K≈13). The doc reads as if this
   Linear is learned; it is not. If a learned start ever matters, tie it to
   the aux head instead.
4. **`λ_ponder` is not implementable as specified** — iterations-per-site is
   produced inside the `no_grad` solver behind a detached mask; the only
   differentiable route is through τ against stored *per-site, per-sweep*
   residual volumes (a memory item the design elsewhere forbids). Left out;
   active fraction is logged instead. The doc itself flags the collapse
   hazard and calls ponder a proxy before Phase 3 — nothing is lost yet.
5. **Residual norms must be fp32 under bf16 autocast** — bf16's ~3 significant
   digits alias the `1e-3` relative stopping test and the safeguard
   comparison. All norms/Gram matrices are computed in fp32; only the Anderson
   *history* is bf16 (§5.2's intent, stated explicitly nowhere).
6. **GN gamma belongs in the row-sum monitor** — the house style is
   `use_affine: true`, and γ scales Jacobian rows exactly like `m`.
   `stability.include_gn_gamma: true` folds `max|γ|` into the cap.
7. **Damping form** — with only map outputs kept (halves history memory), the
   β-damping mixes toward the current iterate rather than the doc's stored
   previous iterates. Same fixed point, same safeguard.
8. **torch.compile is off** — Dynamo cannot trace the data-dependent solver
   loop or the custom `autograd.Function`; with the repo's suppress-errors
   default it would *silently* run eager anyway (the failure mode you already
   documented for DDP). The env vars are set at module scope in
   `curriculum_training_aeq.py` so the mp.spawn ranks see them too. A later
   optimization: compile `_f` alone and call it from the eager loop.
9. **DDP unused-parameter hazard** — `tau_net`, `y_init` (and `aux_head` when
   λ_aux=0) sit outside the autograd graph; a zero-valued guard term keeps DDP
   from erroring without touching the math.
10. **Eval never enters the autograd.Function** — validation runs a plain
    `no_grad` solve, so `inference_mode` wrappers and Catalyst's valid loop
    are safe.

### 1.4 What matches the doc exactly

Block-normalized joint residual (§5.1) in solver *and* stopping test; ridge on
the Gram matrix; map-outputs-in-history; tuple state flattened only in the
least-squares; z₀=0 and pooled-x y₀; Jacobi sweeps; detached reactivating
site mask with τ = τ₀·exp(−Linear(y)); RBP with one joint VJP per iteration
and block-wise freezing; O(1) tape (one evaluation of `H`); K_b=0 ≡ JFB
degenerate case (used by `aeq_diagnostics.py --truncation`); ∞-norm rationale
(the same constraint licenses asynchronous forward and backward); phases A–D;
L_aux on per-class volume fractions; Hutchinson γ‖F_z‖²_F with one Rademacher
probe; the mandatory §7.2 random-mask gate wired into the trainer at
`async_gate_every` cadence — **site gating cannot turn on until it passes**,
and it turns itself back off (with a stderr warning to tighten
`rowsum_target`/`m_max`) if the gate later fails.

---

## 2. Training on 18 classes

### 2.1 Run

```bash
source ~/venv/torch/bin/activate
# allocator taming only needed because persistent_workers=true in the config
MALLOC_TRIM_THRESHOLD_=0 MALLOC_ARENA_MAX=2 \
python curriculum_training_aeq.py --config-dir=conf --config-name=aeq_siam18_16ch
```

Data/validation blocks are copied from the siam18 configs (wirehead synth18
for training, MindfulTensors/MRN `labelfused` + surface metrics for real-data
validation). Point `paths.logdir` wherever you keep runs. Slurm: reuse
`submit-job.sh` with the new entrypoint name.

### 2.2 Phase schedule (§7.1), driven by `model.aeq.phases`

| phase | epochs (default) | mode | act | gating |
|---|---|---|---|---|
| A | 0–4 | unrolled K=5, ordinary backprop (per-sweep checkpointing) | softplus | off |
| B | 5–9 | Anderson + RBP | softplus | off |
| C | 10–14 | Anderson + RBP | ELU | off |
| D | 15+ | Anderson + RBP | ELU | on **after** the random-mask gate passes |

A failure isolates to the single change that phase introduced. Expect a
transient loss bump at the B→C activation switch (seen in the smoke run
here); it recovers within a few hundred steps — don't panic-revert.

### 2.3 What to watch (wandb `aeq/*`, logged every 20 steps)

| metric | healthy | pathological |
|---|---|---|
| `aeq/fwd_rel_res` | < `solver.eps` (1e-3) | plateaus above eps → raise `max_iter` or lower `rowsum_target` |
| `aeq/nfe` | 6–12, drifting down as training settles | pinned at `max_iter` → not converging |
| `aeq/bwd_iters` | 10–25 | pinned at `backward.max_iter` → ρ too close to 1; push `rowsum_target` down (§6.3: ρ is the compute knob) |
| `aeq/anderson_rejects` | rare (<5% of sweeps) | frequent → extrapolation flipping the regime; lower `beta` |
| `aeq/m_mean`, `aeq/m_std_across_batch` | m_std grows from 0 | m_mean → 0 = the §3.2 collapse; m_std stuck at 0 = `y` input-independent (persistent-excitation failure, raise λ_aux) |
| `aeq/aux_loss` | decreasing | flat at ln(18)≈2.9 → y carries no information |
| `aeq/rowsum_l*` | at/below the per-layer target | — (the cap enforces this; watch whether it is *binding*: if far below target, capacity is being wasted) |
| `aeq/async_gate_gap` | < 2e-3 | above → gating stays off; tighten `rowsum_target`, `m_max` |
| `aeq/active_frac_last`, `aeq/active_frac_mean` | falling within a solve after phase D | never < ~1 → raise `tau_0` |
| `aeq/jac_penalty` | drifting down | exploding → raise γ |

Phase transitions and gate decisions are also printed to stderr (`[aeq] ...`).

### 2.4 The experiments the design exists for

Headline ablation (§8.2) — three arms, everything else identical:

```bash
python curriculum_training_aeq.py --config-dir=conf --config-name=aeq_siam18_16ch \
    model.aeq.coupling=mult wandb.run_tag=aeq_mult paths.logdir=.../aeq_mult/
python curriculum_training_aeq.py ... model.aeq.coupling=add  wandb.run_tag=aeq_add
python curriculum_training_aeq.py ... model.aeq.coupling=none wandb.run_tag=aeq_none
```

Judge with `aeq_diagnostics.py` on the resulting checkpoints:

```bash
python aeq_diagnostics.py --ckpt .../model.last.pth --cube 64
```

which prints the §8.1 numbers — `coupling_norm` = ‖G_z(I−F_z)⁻¹F_y‖ (**the**
measurement; ~0 ⇒ the outer contour is decorative and Bet 1 is lost),
cond(S), the ρ(F_z) spread across inputs (mult should move it, add should
not), and the §8.3 K_b=0-vs-64 gradient cosine. Note `coupling_norm` is
*exactly zero at random init* (the gate Linear is zero-initialized, FiLM
style) and must grow during training — a clean monotone probe of the claim.

Other cheap ablations, all pure CLI overrides: `gate_position=pre_norm`
(measures §1.1), `dilations=[4]` (doc-literal M-A / Bet 2 control),
`backward.max_iter` sweep {1,3,10,24,64} (§8.3), `stability.rowsum_target`
sweep {0.5,0.7,0.9} (§11.4), `loss.lambda_aux=0` (§11.5),
`outer.read_residual=true outer.outer_grad=detached_bptt` (§11.9/11.10,
Milestone 2), `site_gating.learned_threshold=false` (§11.7). Resolution
transfer (§11.6): train at `cubesizes_code: [128]*maxreps`, validate at 256 —
`y` is shape-free and the row-sum bound is shape-independent, so nothing else
changes.

### 2.5 Cost and memory (C=16, 256³, defaults)

Peak *training* memory: `z` (1.07 GB fp32 / 0.54 bf16) + Anderson history
2·m_f = 6 bf16 tensors ≈ 3.2 GB + one f-evaluation tape for the RBP backward
+ stem/head full-res buffers. Comfortably inside an A100-80.

Wall-clock honesty: the doc says 2–4× an explicit baseline; with these
defaults (≈10 sweeps × 3 convs forward + ~20 joint VJPs backward) expect
**4–8× per step vs. `gn_hdc_deep`** — and remember the explicit model also
enjoys torch.compile (~2.5×) which the solver forgoes. `accum_steps` buys
gradient quality, not throughput. Budget runs accordingly; §6.3 is the lever
(a lower `rowsum_target` directly cuts `bwd_iters` and permits fewer sweeps).

Peak *inference* memory — the point of the project: the weight-tied trunk
needs `z` + one conv temp + the injected `x`, independent of sweep count;
Anderson can be dropped at deploy time for plain damped iteration (no history
tensors) since ρ is bounded by construction. At 5 channels that is the same
order as the current browser model's per-layer buffer, with adaptive depth on
top. Export for WebGPU = unroll a fixed K (path independence, verified in
`test_aeq.py`, is what makes a fixed-K unroll safe) — the ONNX graph is K
copies of the same 3-conv block reading the same weights.

### 2.6 Failure table additions (extends §13)

| symptom | cause | action |
|---|---|---|
| loss spike right at epoch `softplus_epochs` | B→C ELU switch | expected, transient; if it does not recover in <1 epoch, extend B |
| `aeq/nfe` pinned at max from phase B on, phase A was fine | weights left the contraction regime during unrolled warm-up | lower `rowsum_target` (the cap re-bites immediately — it is forward-time, not a projection) |
| DDP "expected to mark a variable ready" / unused-parameter error | a new head/branch added outside the graph | extend `_ddp_zero_guard` |
| grads NaN only when `bwd_amp: true` | bf16 VJP accumulation | keep `bwd_amp: false` (default) |
| `coupling_norm` stays ~0 after many epochs | gate Linear stuck at zero-init | raise λ_aux; check `aeq/m_std_across_batch`; try `gate_position=post_norm` if you were running pre_norm |
| validation much worse than train dice at same residual | eval budget too small | raise `solver.eval_max_iter` (anytime-inference: more sweeps only helps, given path independence) |

### 2.7 Sanity ritual after ANY solver/model edit

```bash
python test_aeq.py    # must end: ALL PASSED -- Phase 0 machinery gate cleared.
```

The gradient gate (cosine > 0.99 vs. unrolled autograd) is the design's hard
Phase 0 gate; nothing downstream is sound without it.
