# Adaptive Equilibrium Networks: Two Contours with Multiplicative Coupling

**Version 2.1.** Supersedes v1. Changes from v1 in §0; v2.1 additions in §0.1.

Design document for implementation and testing.

---

## 0. What changed from v1

| # | Change | Reason |
|---|---|---|
| 1 | Backward solver is **RBP (Algorithm 1)**, not Neumann (Algorithm 2) | The two generate the same sequence, but RBP's error obeys `e_{k+1} = Jᵀe_k` and is self-correcting under rounding; Neumann accumulates into a running sum. Matters in bf16. |
| 2 | Stability constraint is **∞-norm (row-sum)**, not spectral 2-norm | Asynchronous (per-site) iteration requires `ρ(|J|) < 1`, which is stronger than `ρ(J) < 1`. Since `ρ(|J|) ≤ ‖J‖_∞`, a row-sum bound suffices. A 2-norm bound does **not** give this. |
| 3 | **Per-site residual gating** added to the forward solve | Different regions of the input converge at very different rates. Freezing converged sites is asynchronous relaxation, with convergence theory from Chazan & Miranker. |
| 4 | **Block-wise adjoint stopping** | The two contours converge at different rates; freeze `λ_y` when its increment is small and keep iterating `λ_z`. |
| 5 | Backbone abstracted; **two instantiations** | Variant M (volumetric MeshNet) and Variant T (transformer). All solver, stability, and diagnostic machinery is shared. |
| 6 | Row-sum normalization replaces power iteration | Cheaper (no persistent buffer, no matmuls) and **shape-independent**, which fixes v1's resolution-transfer caveat. |

## 0.1 Added in v2.1

| # | Addition | Reason |
|---|---|---|
| 7 | `g` may read the **residual** as an input (§3.2), behind a flag | Lets the outer contour condition on *how the solve is going*, not just where it is. This is the difference between open-loop adaptive compute (read the input, predict a budget) and closed-loop (watch the run). |
| 8 | **`outer_grad` three-way flag** (§6.5) | Once `g` reads residuals, a trajectory gradient path opens that the adjoint does not capture. Detached-BPTT on `y` costs ~10⁻⁵ of the memory of the joint state, so it is nearly free — but it is a bias, and it needs an ablation switch. |
| 9 | Explicit **collapse warning**: residual is an input, never a target (§3.2, §7.3, §13) | If `y` is rewarded for driving residuals down, the optimum is `m(y) → 0` — a near-constant map that converges in one sweep to a useless fixed point. |
| 10 | **Active-set shrinkage** diagnostic and the `N^(2/3)` prediction (§5.3, §8.6) | The late-iteration live set should be a boundary surface, not a volume. If it is, large `K_f` is far cheaper than `K_f × per-sweep` suggests. Testable and cheap. |
| 11 | **Halting granularity** table and Variant T-A stub (§3.5) | Per-head halting is only meaningful when the fixed-point state is the attention matrix (SELF-Transformer), not token representations. |

---

## 1. Scope and the central claim

Build a network with two coupled fixed points:

- an **inner contour** carrying activations `z`,
- an **outer contour** carrying a small vector `y` that **multiplies** into the inner map.

Train by implicit differentiation. Anderson-accelerated fixed-point iteration forward, RBP backward. No backpropagation through the solver trajectory. No Broyden.

**The claim under test.** Every two-state equilibrium system in the literature couples its states *additively* — MDEQ sums resampled features at each scale's junction, HRM injects the high-level state into the low-level block, looped transformers inject the input additively. Multiplicative modulation exists everywhere as FiLM/gating/hypernetworks, but the modulator is always a feedforward function of the input, never a second state with its own fixed point.

The bet is that multiplicative coupling is qualitatively different: `y` changes `∂f/∂z` — the inner contour's *spectrum*, its contraction rate, and (Variant M) its effective receptive field. Additive coupling can only shift where the inner fixed point sits. If that difference does not show up in the measurements of §8, the outer contour is decorative and should be removed.

**Non-goals.** Beating a SOTA number. Weight-tying costs accuracy against a tuned explicit baseline; the deliverable is a working two-contour system with measured properties.

**Novelty caveat for the implementer.** Do not write "first to do X" without a systematic literature pass. Known near-neighbours are listed in §12. In particular **DeltaDEQ already exploits per-dimension heterogeneous convergence** as an inference-acceleration trick; our contribution there is coupling the halting threshold to the outer contour, not the observation that convergence is heterogeneous.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `z` | inner state (activations). Variant M: `(B,C,D,H,W)`. Variant T: `(B,N,d)`. |
| `y` | outer state, `(B, C_y)`, `C_y` small (default 64) |
| `x` | input injection, precomputed once by the stem |
| `f(z,y,x)` | inner map |
| `g(y,z)` | outer map |
| `H(z,y,x) = (f, g)` | joint map |
| `F_z, F_y, G_z, G_y` | Jacobian blocks of `H` at the fixed point |
| `J` | joint Jacobian |
| `m(y)` | modulation signal — the multiplier |
| `s` | a **site**: one voxel (Variant M) or one token (Variant T) |

Joint fixed point: `z = f(z,y,x)` and `y = g(y,z)` simultaneously.

---

## 3. Architecture

### 3.1 Shared skeleton

```
x       = stem(input)                # computed once, injected every evaluation
z*, y*  = solve(H, x)                # §5
out     = head(z*)
aux     = aux_head(y*)               # §7.3
```

Stem and heads are explicit and outside the loop.

### 3.2 Outer map `g` (identical in both variants)

```
s   = global_pool(z)                       # spatial mean (M) or token mean (T) → (B, C)
q   = [pool(‖r_z‖), log(‖r_z‖/‖r_z_prev‖)] if read_residual else []   # (B, 2)
u   = MLP([y, s, q])                       # 2 layers, hidden 2*C_y → (B, C_y)
y'  = (1 - α) * y + α * tanh(u)            # α = 0.5
```

**`read_residual` (default off) is what makes the adaptivity closed-loop.** Without it, `y` sees only *where* the inner loop has gotten to; with it, `y` also sees *how fast it is getting there* — the residual magnitude and its contraction ratio between sweeps. Standard ACT reads the input once and predicts a budget; this reads the run as it unfolds and can extend or cut it mid-solve.

Two consequences, both handled elsewhere: the residual is zero at the fixed point, so it only carries information under truncation (§6.5), and it opens a trajectory gradient path (§6.5).

> **Collapse warning.** The residual is an **input** to `y`, never a target. Do not add a loss term rewarding `y` for small residuals, and keep `λ_ponder` small or zero. The optimum of "minimize the residual" is `m(y) → 0`: a near-constant inner map that converges in one sweep to a useless fixed point. The task loss decides whether fast convergence is worth it.

Damped-tanh rather than `LayerNorm(y + …)`: it gives the clean bound `‖G_y‖_∞ ≤ (1-α) + α‖MLP'‖_∞` with no normalization Jacobian in the way.

`g` reads `z` **only through a global pool**. `y` is a regime selector, not a second segmentation. This also keeps `G_z` cheap and makes `y` shape-independent, which is what buys resolution transfer.

### 3.3 Modulation `m(y)`

```
m = m_max * sigmoid(Linear(y))       # (B, C), in (0, m_max)
```

`m_max` default `0.9`. This is the **only** place the outer contour touches the inner one.

**Equivalence note.** Per-output-channel gating of activations equals per-filter-bank scaling of weights: `m_c·(W_c * z) = (m_c W_c) * z`. Implement the activation form; never materialize scaled weights. `y` need not be parameter-shaped.

### 3.4 Variant M — volumetric MeshNet

```
h = conv3d(z, W, dilation=d)         # d from config
h = m(y) * h                         # broadcast (B,C,1,1,1)
h = h + x                            # input injection
z' = act(groupnorm(h))
```

Sub-variants, config-selected:

- **M-A (default)**: single dilation `d = 4`.
- **M-B**: dilation mixture. `branches = [conv3d(z, W_d, dilation=d) for d in [1,2,4,8,16]]`, `w = softmax(Linear(y))`, `h = Σ_d w_d · branches[d]`. The outer contour schedules receptive-field growth. Costs the sum of branch FLOPs per evaluation unless top-k sparsified.
- **M-C**: rank-`r` fast weights, `W_eff = W + U(y)V(y)ᵀ`, `r = 4`. Only if gating proves too weak (it can rescale but not rotate).

`act`: softplus during warm-up, ELU after (§7.1). `z` starts at zero and ReLU-family kinks at the origin destabilize early solves.

Normalization: **GroupNorm**, 8 groups. Not BatchNorm — batch statistics are layer-indexed, meaningless when layers are implicit, and BN scales the Jacobian badly. Keep the learnable affine.

Dropout: if used, **variational only** — one mask per training step reused across every evaluation of `f`. Fresh masks per call make `f(z*) = z*` unattainable. Default off.

### 3.5 Variant T — transformer

```
h  = z + m_a(y) * Attn(LN(z), temp=T(y))
z' = h + m_m(y) * MLP(LN(h)) + x
```

with `T(y) = T_0 · exp(Linear(y))` an attention temperature and `m_a, m_m` per-channel gates.

Weight-tied single block, input injection every evaluation (both required for path independence — see §8.4).

**Why the transformer variant may be the better vehicle.** MDEQ's dominant cost was a receptive-field lower bound: with `3×3` convolutions, `f` must be applied enough times for the stacked receptive field to cover the input, or each further application admits new pixels and disrupts the equilibrium — roughly 30 evaluations on CIFAR-10, over 100 on Cityscapes. **Attention has full receptive field in one application.** That bound simply does not exist here, so the forward solve should need far fewer evaluations.

**Why the ∞-norm framework fits attention well.** The softmax matrix `A` is row-stochastic, so `‖A‖_∞ = 1` exactly and the value path costs nothing to bound: `‖A V W_O‖_∞ ≤ ‖W_V W_O‖_∞`. The `Q`/`K` paths go through the softmax Jacobian and scale with inverse temperature — which is precisely what `T(y)` controls. So in Variant T the outer contour's modulation knob **is** the stability knob. Elegant, but verify empirically (§7.2); the three-term Jacobian is not fully bounded by the above argument.

Per-**token** halting replaces per-voxel halting (§5.3). Everything else is unchanged.

**Scope.** Variant T is a **standard DEQ-transformer** — state is token representations, weight-tied block, input injection — plus exactly two additions: the outer contour `y`, and per-token halting. Nothing else about it is novel, and that is deliberate: it is the testbed, and the machinery under test should be the only unfamiliar part.

**Halting granularity depends on what the state is.**

| model | fixed-point state | natural halting unit |
|---|---|---|
| Variant M | voxel features | voxel |
| Variant T | token representations | token |
| Variant T-A (future) | per-head alignment matrix | head, or query row within head |

**Variant T-A — SELF-Transformer / FPSA (arXiv:2507.13569), not in scope for the first build.** There the fixed-point state is the attention matrix itself, iteratively refined rather than produced in one pass. Per-head halting is meaningful only in that setting; in Variant T it means nothing, because heads do not have separate states. T-A is a genuinely different architecture, not a config switch — treat it as a follow-on once the machinery is validated.

---

## 4. What is iterated

One **Jacobi sweep** updates both states from tick-`k` values:

```
z_{k+1} = f(z_k, y_k, x)
y_{k+1} = g(y_k, z_k)
```

No nested inner solve. Nested and flat schedules have the same equilibrium; flat is the default because Anderson wants a joint residual and because the nested schedule wastes its first inner solve at an uninformative `y`.

Keep `z` and `y` as a **tuple**, not a concatenation — different shapes, dimensionalities, semantics. Flatten only inside the Anderson least-squares.

Initialization: `z_0 = 0`; `y_0 = tanh(Linear(global_pool(x)))`, **not random**. `y` sets `ρ(F_z)`; a random `y` can start the inner contour outside its contraction regime.

---

## 5. Forward solve

### 5.1 Residual and block scaling

```
r_z = f(z,y,x) - z
r_y = g(y,z)   - y
r_flat = concat( r_z.flatten()/sqrt(numel(z)),
                 r_y.flatten()/sqrt(numel(y)) )
```

**Critical.** With `|z| ≈ 10⁹` and `|y| = 64`, an unnormalized concatenation makes `r_y` about 10⁻⁷ of the norm. Anderson then optimizes `z` alone and **`y` silently never converges** — the model trains, produces plausible metrics, and the outer contour is inert. This is the easiest way to get a wrong implementation that looks right. Use the same normalization in the stopping criterion.

### 5.2 Safeguarded Anderson

Type-II, window `m_f = 3`, damping `β = 0.8`, bf16 history.

```
z, y = init(x)
Zh, Rh = [], []
for k in 0..K_f:
    Fz, Fy = H(z, y, x)
    r = residual(...)                          # block-normalized
    if ‖r‖/‖state‖ < ε_f: break
    push (Fz,Fy) to Zh; push r to Rh           # store MAP OUTPUT, not iterate
    if len(Rh) > 1:
        R = stack(Rh); G = R @ R.T
        G += λ_ridge * trace(G)/n * I          # λ_ridge = 1e-4
        v = torch.linalg.solve(G, ones(n)); α = v / v.sum()
        cand = β·Σ αᵢ Zh[i] + (1-β)·Σ αᵢ (previous iterates)
    else:
        cand = (Fz, Fy)
    if ‖residual(cand)‖ > ‖r‖:                 # safeguard
        cand = (Fz, Fy); truncate history to 1
    z, y = apply_site_gate(cand, z, y, r)      # §5.3
```

Three things not to shortcut:

1. **Ridge on `G`.** `R Rᵀ` is routinely near-singular once iterates align. A bare `torch.inverse` will NaN mid-training.
2. **Safeguarding.** Anderson extrapolates linearly. With multiplicative modulation an extrapolated iterate can land where `m(y)` has flipped the contraction regime. Reject on residual increase, fall back to a plain step.
3. **bf16 history.** `Zh`/`Rh` are `2·m_f` tensors the size of `z`. The least-squares is `m_f × m_f` and precision there is irrelevant.

`ε_f = 1e-3` relative, `K_f = 24`. Log the achieved residual every step — do not assume convergence.

### 5.3 Per-site residual gating (asynchronous relaxation)

```
r_site  = ‖r_z‖ reduced over the channel axis      # (B,1,D,H,W) or (B,N,1)
τ       = τ_0 * exp(-Linear(y))                    # outer contour sets the threshold
active  = (r_site > τ).detach()                    # NOT in the gradient path
z_new   = where(active, cand_z, z)                 # freeze converged sites
```

This is an **asynchronous fixed-point iteration**: a frozen site's stale value still feeds its updating neighbours through the convolution (M) or attention (T). Convergence theory is classical — Chazan & Miranker's chaotic relaxation, Bertsekas & Tsitsiklis — and the condition is `ρ(|J|) < 1`, enforced in §7.2.

Four implementation requirements:

- **The mask is detached.** At convergence, frozen sites sit at their fixed point, so the implicit derivative is unaffected. Under truncation the mask introduces bias; that is measured in §8.3.
- **Sites must be able to reactivate.** A frozen site whose neighbours have since moved is no longer converged. Recompute `r` everywhere each sweep.
- **Phase 1 saves no FLOPs.** Computing `r` everywhere means evaluating `f` everywhere. The gating buys *adaptive-depth semantics* and a spatial compute map, not speed. FLOP savings require DeltaDEQ-style caching of linear-operation results with sparse recomputation — treat as a separate optimization phase, and benchmark it, because sparse conv on GPU is often slower than dense unless sparsity is high.
- **`τ_0` default `0.5·ε_f`**, i.e. sites freeze at a stricter threshold than global convergence.

**What actually shrinks.** The kernel does not shrink — dilation and receptive field are unchanged. What shrinks is the **set of sites the kernel is evaluated at**. A frozen voxel must still be *readable* by its live neighbours, so the full volume stays resident: **memory stays flat at `|z|`, only FLOPs fall.** Do not budget for a shrinking activation tensor.

**Expected shape of the curve (Variant M).** Sweep 1 computes everywhere. Background and air freeze almost immediately; homogeneous interior next; by late sweeps only boundaries and ambiguous structures are still live. The mechanism is geometric rather than semantic: the receptive field grows with each sweep, so a voxel's iteration count is roughly **the distance to the evidence that disambiguates it**. Air has local evidence; a subcortical boundary may need context from centimetres away and cannot converge until it arrives.

**Prediction to test.** The late-iteration live set is a *surface*, not a volume, so the active fraction should fall roughly as `N^(2/3)`. If it holds, a large `K_f` costs far less than `K_f × per-sweep` implies, and raising `K_f` becomes cheap. Plot active fraction vs. sweep index on log axes (§8.6).

**Reactivation is not a bug.** A voxel that froze early can reactivate when long-range information reaches its neighbours — exactly the case where the model *should* spend more. So shrinkage is not strictly monotone, and whether the reactivation rate is low enough for sparse recompute to pay is Phase 3's empirical question.

---

## 6. Backward: RBP with block-wise stopping

### 6.1 Why RBP and not Neumann, and why not Anderson

RBP (Algorithm 1) and Neumann (Algorithm 2) generate the same sequence in exact arithmetic — `z_{k+1} - z_k` in Alg 1 equals `v_k` in Alg 2, so even the convergence test is the same quantity. Both are equally adaptive. **The tiebreaker is numerical**: RBP's error obeys `e_{k+1} = Jᵀ e_k` and is damped by subsequent steps; Neumann accumulates into a running sum where early error persists. In bf16, prefer RBP.

Not Anderson: the adjoint system `(I - Jᵀ)λ = g` is **linear**. RBP needs two tensors, O(1). Anderson would cost `2·m_b·|z|` for nothing structural — and Anderson with full memory on a linear fixed-point iteration is essentially GMRES, so if you ever want acceleration here, use a Krylov solver directly rather than Anderson.

### 6.2 Algorithm

```
λ_z, λ_y = 0, 0
frozen_z = frozen_y = False
for k in 1..K_b:
    v_z, v_y = vjp(H, (z*, y*), (λ_z, λ_y))       # ONE VJP through the joint map
    n_z, n_y = λ_z_next - λ_z, λ_y_next - λ_y     # per-block increments
    if not frozen_z: λ_z = v_z + g_z
    if not frozen_y: λ_y = v_y + g_y
    if ‖n_z‖/sqrt(numel(z)) < ε_b: frozen_z = True
    if ‖n_y‖/sqrt(numel(y)) < ε_b: frozen_y = True
    if frozen_z and frozen_y: break
∂L/∂θ = vjp_θ(H, (z*,y*,θ), (λ_z, λ_y))
∂L/∂x = vjp_x(H, (z*,y*,x), (λ_z, λ_y))
```

Apply the block transpose in **one** VJP call — do not implement `F_zᵀ`, `G_zᵀ` separately. `λ` is a tuple; PyTorch handles it.

**Block-wise stopping is the point.** The two contours converge at different rates; there is no reason to keep iterating `λ_y` after it has settled. Freezing a converged block is again asynchronous adjoint iteration, licensed by the same condition — note `ρ(|Jᵀ|) = ρ(|J|)`, so **one constraint covers asynchronous forward and asynchronous backward**.

`K_b = 32` max, `ε_b = 1e-4`. Note `K_b = 0` recovers the one-step / JFB gradient as the degenerate case.

### 6.3 The `ρ` → `K_b` relationship

Truncation error is `ρ^(K_b+1)/(1-ρ)`. Because `ρ` is bounded by construction:

| target `ρ` | `K_b` for 1e-3 relative error |
|---|---|
| 0.9 | ~87 |
| 0.7 | ~24 |
| 0.5 | ~10 |

**The stability constraint is therefore the primary compute knob, not merely a safety measure.** Push `ρ` down harder than feels necessary; back off only when accuracy complains.

### 6.4 PyTorch wiring

```python
with torch.no_grad():
    z_star, y_star = anderson_solve(H, x, ...)

z0 = z_star.detach().requires_grad_()
y0 = y_star.detach().requires_grad_()
z1, y1 = H(z0, y0, x)      # ONE differentiable application — the entire tape
# custom autograd.Function whose backward runs §6.2 RBP
```

Tape holds exactly one evaluation of `H`. Peak backward activation memory is one forward pass, independent of `K_f` and `K_b`.

### 6.5 The outer gradient path — mixed mode

There are two routes from the outer parameters to the loss:

1. **Through the equilibrium**: `y* → m(y*) → f → z*`. Captured exactly by the adjoint, O(1) memory. This is §6.2 and needs nothing extra.
2. **Through the trajectory**: how the path taken affects where the solve lands. **Identically zero when `read_residual` is off** — `g` then reads only `pool(z)` and the fixed point is path-independent. Nonzero the moment `y` reads residuals, because the residual exists only off-equilibrium.

Path 2 needs BPTT. The cost asymmetry decides how:

| | size (C=16, 256³) | BPTT over `T=24` sweeps |
|---|---|---|
| `z` | 1.07 GB | 25.7 GB — fatal |
| `y` + pooled summaries | ~320 B | ~7.7 KB |

Ratio ≈ 10⁻⁵ in memory, and the same story in compute: `f` is ~10¹¹ MACs, `g`'s MLP ~10⁴. **The outer loop is free in both currencies.** That is what licenses mixed mode: implicit differentiation on the big state, truncated BPTT on the small one.

```yaml
outer_grad: implicit | detached_bptt | full_joint
```

| setting | extra memory | extra compute | bias |
|---|---|---|---|
| `implicit` | 0 | 0 | `y` cannot learn to use residuals at all |
| `detached_bptt` (default when `read_residual`) | ~8 KB | ~10⁻⁵ | small, bounded |
| `full_joint` | 25.7 GB | large | none — **toy scale only** |

`detached_bptt` stores the pooled scalars `pool(z_k)` and `‖r_k‖` (tiny), backpropagates exactly through the `y → y → y` recurrence, and treats the `z` inputs as constants. It drops the `y → z_k → y` cross-term while keeping exact the part that carries the "how is this run going" signal.

`full_joint` exists so the bias of `detached_bptt` can be *measured* at toy scale rather than assumed. Run it once in Phase 0; do not enable it at full resolution.

---

## 7. Training

### 7.1 Warm-up

| Phase | Epochs | Mode |
|---|---|---|
| A | 0–5 | Unrolled weight-tied `K=5`, ordinary backprop, softplus, **site gating off** |
| B | 5–10 | Solver + RBP, softplus, site gating off |
| C | 10–15 | Solver + RBP, ELU, site gating off |
| D | 15+ | Solver + RBP, ELU, **site gating on** |

Four steps rather than one so that a failure isolates to a single change. Site gating goes last because it is the only component with no fallback.

### 7.2 Stability: ∞-norm, not 2-norm

Asynchronous iteration requires `ρ(|J|) < 1`. Since `ρ(|J|) ≤ ‖|J|‖_∞ = ‖J‖_∞`, a **row-sum bound suffices**. A spectral (2-norm) bound does not imply it.

**Variant M — analytic and cheap.** The row sum of the conv Jacobian at an output site is the sum of `|W|` over input channels and kernel taps, times the gain and the activation slope:

```python
rowsum = W.abs().sum(dim=(1,2,3,4)).max()          # per-output-channel L1
scale  = min(1.0, target / (m_max * rowsum))
W_eff  = W * scale
```

Cheaper than power iteration (no matmuls, no persistent buffer) and **shape-independent**, which removes v1's resolution-transfer caveat. Dilation does not change the tap count, so the bound is dilation-invariant — a small bonus for Variant M-B.

**Variant T.** Row-stochasticity of softmax gives `‖A‖_∞ = 1` free, so bound `‖W_V W_O‖_∞` by row sums. The `Q`/`K` paths through the softmax Jacobian scale with inverse temperature; keep `T(y)` bounded below. This argument does not cover the full three-term Jacobian — treat it as a heuristic and rely on the empirical gate below.

**The weak link, stated plainly.** GroupNorm/LayerNorm Jacobians are not exactly 1-Lipschitz, so the analytic product bound is a heuristic, not a proof. Therefore:

**Empirical gate (mandatory).** Every 50 steps, run the forward solve with a *random* freezing mask (freeze 50% of sites at random each sweep) and confirm it still converges to the same fixed point as the synchronous solve, within `2·ε_f`. This directly tests the property the analytic bound is a proxy for. If it fails, tighten `target` and `m_max` until it passes. **Do not enable site gating in training until this gate passes.**

### 7.3 Losses

```
L = L_task(head(z*), target)
  + λ_aux · L_aux(aux_head(y*), aux_target)
  + γ · ‖F_z‖_F²                                  # Hutchinson, 1 Rademacher probe
  + λ_ponder · mean(iterations_per_site)           # only when gating is on
```

- `L_task`: Dice + cross-entropy (M), task loss (T).
- **`L_aux` is not optional.** If `y` affects the objective only through its modulation of `z`, nothing pins it down and it drifts along directions the gradient cannot see — persistent-excitation failure, transplanted from adaptive control. Target: per-class volume fraction (M) or a sequence-level label (T). `λ_aux = 0.1`.
- Jacobian regularization `γ = 0.1`. Directly regularizes the modulation channel; published results took an MDEQ from 17 function evaluations to 6.
- Ponder cost `λ_ponder = 0.01`, and note it is only meaningful once §5.3 phase 2 actually saves FLOPs. Before that it is a proxy.

### 7.4 Optimizer

AdamW, lr `1e-3` cosine, weight decay `1e-4`.

---

## 8. Diagnostics

### 8.1 Exact Schur complement (once per epoch, one batch)

With `C_y = 64`, the outer sensitivity operator

```
S = (I - G_y) - G_z (I - F_z)^{-1} F_y
```

is `64 × 64` and can be **formed explicitly**: for each basis vector `e_i`, JVP `F_y e_i` into `z`-space, solve `(I - F_z)u = a` by RBP, VJP through `G_z`. Cost: `C_y` inner linear solves — too expensive per step, fine per epoch.

Log `cond(S)`, its spectrum, and `‖G_z(I-F_z)^{-1}F_y‖`. **If that last quantity is near zero, the outer contour is decorative** and the central claim of §1 is false. This is the single most important measurement in the project.

### 8.2 Additive vs. multiplicative coupling (the headline ablation)

Same parameter count, same NFE budget, three arms:

- **mult**: `m(y) ⊙ (W*z)` — as designed
- **add**: `(W*z) + Linear(y)` — broadcast additive injection, i.e. what MDEQ/HRM do
- **none**: outer contour removed

Report task metric, NFE to `ε_f`, and `‖G_z(I-F_z)^{-1}F_y‖`. The claim is that mult changes `ρ(F_z)` across inputs while add does not — measure `ρ` variance across the batch for each arm.

### 8.3 Truncation and mask bias (every 100 steps)

Cheap: parameter gradient at `K_b = 0` vs `K_b = 64`; report cosine similarity and norm ratio. Separately, gradient with site gating on vs off at the same `K_b`. The concern is that the whole outer-loop gain runs through `(I - F_z)^{-1}`, which `K_b = 0` replaces with the identity.

### 8.4 Path independence

Solve from three different `z_0` (zeros, random, another sample's equilibrium); confirm the same fixed point within `ε_f`. **Test-time compute scaling only works for path-independent models, and path independence requires weight tying and input injection** — both present here, but verify rather than assume. Any anytime-inference claim depends on this.

### 8.6 Closed-loop adaptivity (only meaningful when `read_residual` is on)

Three cheap checks that the outer contour is actually watching the run rather than just the input:

- **Does `τ` move within a solve?** Log `τ` per sweep for a fixed input. A constant `τ` across sweeps means `y` is ignoring the residual channel and the closed loop is decorative — fall back to `implicit` and save the complexity.
- **Does `τ` differ across inputs at the same sweep index?** Separates input-conditioned adaptivity (ordinary ACT) from run-conditioned adaptivity (the claim).
- **Active-fraction curve.** Plot live-site fraction vs. sweep index, log-log, and fit the exponent. The prediction is ≈ `N^(2/3)` (§5.3). Also log the **reactivation rate** — fraction of sites that unfreeze per sweep — since it sets whether sparse recompute can ever pay.

### 8.5 Per-block and per-site curves

Log `‖r_z‖/‖z‖` and `‖r_y‖/‖y‖` separately vs NFE — if `r_y` plateaus while `r_z` converges, §5.1's block scaling is wrong. Also log the spatial map of iterations-per-site; the expected signature (from recurrent-depth segmentation work) is boundaries and pathology receiving several times the iterations of homogeneous interior.

---

## 9. Memory and compute

Variant M at `256³`, fp32:

| item | count | bytes |
|---|---|---|
| `z`, C=16 | 2.7e8 | 1.07 GB |
| `z`, C=64 | 1.07e9 | 4.29 GB |
| conv weights, C=64, 3³ | 1.1e5 | 442 KB |
| `m(y)` | 64 | 256 B |

Anderson history is `2·m_f` tensors the size of `z`:

| C | `m_f=5` fp32 | `m_f=3` bf16 |
|---|---|---|
| 16 | 10.7 GB | 3.2 GB |
| 64 | 42.9 GB | 12.9 GB |

**MeshNet's low channel count is what makes this feasible at all.** Start at `C = 16`, `m_f = 3`, bf16. Do not scale `C` before the solver is validated.

Backward is O(1) in `K_b`: two tensors plus one forward pass of tape. Expect 2–4× wall-clock versus an explicit baseline at equal accuracy; the currency is solver iterations, not bytes.

---

## 10. Milestones

**Phase 0 — machinery gate.** Single contour (`y` disabled), tiny problem (`32³`, `C=8` for M; 2-layer, `d=64`, `N=64` for T).
- *Pass*: RBP gradient at `K_b = 64` matches fully-unrolled autograd (unroll 200) with cosine > 0.99, relative error < 1e-2.
- Hard gate. Nothing downstream is sound without it.

**Phase 1 — two contours, multiplicative, synchronous.** No site gating.
- *Pass*: forward reaches `ε_f` within `K_f`; random-mask convergence gate (§7.2) passes; `‖G_z(I-F_z)^{-1}F_y‖` materially non-zero; task metric within 5% of explicit baseline.

**Phase 2 — asynchronous site gating.**
- *Pass*: same fixed point as synchronous within `2·ε_f`; iterations-per-site map shows structure (boundaries > interior); no metric regression.

**Phase 3 — sparse recomputation.** DeltaDEQ-style caching for actual FLOP savings.
- *Pass*: measured wall-clock improvement. If dense-with-mask beats sparse on GPU, report that and stop.

**Phase 4 — Variant M-B (dilation mixture) or M-C (fast weights).**
- *Pass*: fewer NFE at matched per-iteration FLOPs, or better metric at matched NFE. If neither, report the negative result. The receptive-field-scheduling hypothesis is speculative.

### Which variant to build first

**Build Variant T first as the testbed, port to Variant M as the application.** T iterates in hours rather than days, has no receptive-field bound to confound NFE measurements, and lets §8.1–8.4 be run cheaply and often. M is where the memory wall is real and where deployment matters, but it is a poor place to debug a solver. All of §4–§8 is backbone-agnostic by construction, so the port is mechanical.

---

## 11. Ablations

1. **Multiplicative vs. additive vs. none** (§8.2) — the headline.
2. `K_b` sweep `{0,1,3,10,24,64}` — accuracy and wall-clock vs. truncation.
3. **Jacobi vs. Gauss–Seidel** at fixed NFE. GS reads the freshest `z` when computing `y`; costs one extra kernel of sequential depth per sweep but folds the joint loop out of the spectrum. No published comparison in an equilibrium model that I found. Cheap.
4. `ρ` target sweep `{0.5, 0.7, 0.9}` — the compute/accuracy frontier.
5. `λ_aux` on/off — does `y` drift without its own loss?
6. **Resolution / length transfer**: train `128³` test `256³` (M); train `N=512` test `N=2048` (T). `y` carries no spatial extent, so this should work; an activation-shaped outer state could not.
7. **Learned vs. fixed halting threshold** — is `τ = τ_0·exp(-Linear(y))` better than a constant `τ`? This is the specific claim that the outer contour should schedule *compute allocation*.
8. **Dilation preservation** (M): weight-tied single dilation vs. `L` distinct blocks with MeshNet's original dilation schedule as a dense-feedback multi-block state.
9. **`read_residual` on/off** — does closed-loop beat open-loop adaptivity? Judge on task metric at matched mean NFE, and on §8.6's within-solve `τ` variation. This is the specific claim that watching the run beats reading the input.
10. **`outer_grad`: `implicit` vs `detached_bptt`** at matched `read_residual=on`. If `implicit` matches, the detached BPTT is unnecessary machinery and should be cut. Run `full_joint` once at toy scale to bound the bias of both.

---

## 12. Prior art the implementer must read

| Work | Why |
|---|---|
| **MDEQ** (Bai, Koltun & Kolter, NeurIPS 2020, arXiv:2006.08656) | Closest system: multiple states, joint equilibrium, Jacobi schedule, O(1) memory. §3.3 for the GroupNorm / variational-dropout / softplus findings inherited here; Appendix B.2 for the receptive-field bound on iteration count. Coupling is **additive** — that is the contrast this design rests on. |
| **DeltaDEQ** | Already exploits per-dimension heterogeneous convergence for inference acceleration. Read before claiming novelty on §5.3. |
| **Chazan & Miranker (1969), Bertsekas & Tsitsiklis** | Asynchronous / chaotic relaxation. The `ρ(|J|) < 1` condition that §7.2 exists to enforce. |
| **Path Independent Equilibrium Models** (NeurIPS 2022) | Test-time compute scaling requires path independence, which requires weight tying and input injection. Basis for §8.4. |
| **Jacobian regularization for DEQs** (arXiv:2106.14342) | The `γ` term and its measured NFE effect. |
| **HRM / TRM** | Nested two-timescale with one-step gradient. TRM reported *better* results after dropping the one-step approximation — motivation for §8.3. Coupling is additive. |
| **RD-ViT** (2026 preprint) | Recurrent-depth ViT for medical segmentation with per-patch ACT halting; boundary regions received 3–4 iterations vs 1–2 for homogeneous ones. Nearest neighbour to §5.3 in 2D. Past my reliable knowledge — read directly. |
| **Almeida (1990), Pineda (1988)** | Original recurrent backpropagation. Algorithm 1. |
| **Walker & Ni** | Anderson ≈ GMRES on linear systems; why not to use Anderson on the backward. |

---

## 13. Failure modes

| Symptom | Cause | Action |
|---|---|---|
| `r_y` never converges, `r_z` fine | missing block scaling §5.1 | fix residual normalization |
| NaN in Anderson after N steps | singular Gram matrix | verify `λ_ridge` applied |
| Solve diverges mid-training | `ρ` crossed 1 | check monitor; tighten `m_max` and row-sum target; raise `γ` |
| Diverges from a step that looked fine | Anderson extrapolation flipped the regime | verify safeguard active |
| Synchronous solve fine, gated solve diverges | `ρ(|J|) ≥ 1` — 2-norm bound is insufficient | this is exactly why §7.2 uses ∞-norm; tighten `target` |
| Sites freeze then thrash | reactivation not implemented, or `τ` too tight | recompute `r` everywhere each sweep |
| Gradient cosine vs. unrolled < 0.9 | forward truncated too early | raise `K_f`, lower `ε_f`, re-run Phase 0 |
| `S` nearly diagonal | outer contour decorative | check `m(y)` varies across inputs; if not, the §1 claim fails |
| Good metric, `y` constant across inputs | persistent-excitation failure | raise `λ_aux`, shrink `C_y` |
| Sparse recompute slower than dense | GPU sparsity threshold | expected below ~70% sparsity; report and use dense |
| `m(y) → 0`, NFE collapses to 1–2, metric craters | `y` rewarded for small residuals | remove any residual-minimizing loss term; drop `λ_ponder`; residual is an input, not a target |
| `τ` constant across sweeps | `y` ignoring the residual channel | check `outer_grad ≠ implicit`; if still flat, closed loop is decorative — revert to `read_residual=off` |
| Memory grows with `K_f` after enabling site gating | budgeting for a shrinking activation tensor | frozen sites stay resident; only FLOPs fall, not memory |
| Active fraction never drops below ~1 | threshold too tight, or reactivation thrashing | raise `τ_0`; check §8.6 reactivation rate |

---

## 14. Configuration

```yaml
variant: T            # T (transformer, testbed) or M (meshnet, application)

model:
  C: 16               # M: channels;  T: d_model = 256
  C_y: 64
  m_max: 0.9
  alpha_outer: 0.5
  # M only
  n_groups: 8
  dilation: 4
  dilation_set: [1,2,4,8,16]
  rank_r: 4
  # T only
  n_heads: 8
  temp_0: 1.0

solver_forward:
  method: anderson
  window_m: 3
  beta: 0.8
  lambda_ridge: 1.0e-4
  eps: 1.0e-3
  max_iter: 24
  history_dtype: bfloat16
  safeguard: true

site_gating:
  enabled: false      # turn on at Phase D / Milestone 2
  tau_0: 5.0e-4
  learned_threshold: true
  sparse_recompute: false

outer_contour:
  read_residual: false      # closed-loop adaptivity; turn on at Milestone 2
  outer_grad: implicit      # implicit | detached_bptt | full_joint
                            # detached_bptt is the default once read_residual=true
                            # full_joint is toy-scale only (Phase 0 bias check)

solver_backward:
  method: rbp         # Algorithm 1, NOT Neumann
  max_iter: 32
  eps: 1.0e-4
  blockwise_stopping: true

stability:
  norm: inf           # row-sum, NOT spectral
  rowsum_target: 0.7
  async_gate_every: 50
  async_gate_mask_frac: 0.5

loss:
  lambda_aux: 0.1
  gamma_jac: 0.1
  lambda_ponder: 0.01

schedule:
  phase_A_epochs: 5
  phase_B_epochs: 10
  phase_C_epochs: 15
```

---

## 15. The two bets, stated so they can lose

**Bet 1 — multiplicative coupling is qualitatively different from additive.** Settled by §8.2 and §8.1. If `‖G_z(I-F_z)^{-1}F_y‖ ≈ 0`, or if the `add` arm matches the `mult` arm on every metric, the bet is lost and the honest write-up is a negative result on a natural hypothesis.

**Bet 2 (Variant M only) — dilation defuses MDEQ's receptive-field bound.** Dilation grows the receptive field exponentially rather than linearly, attacking the term that dominated high-resolution MDEQ's iteration count. Settled by comparing NFE-to-`ε_f` against a matched non-dilated equilibrium model at equal resolution. Variant M-B — letting the outer contour schedule the dilation — is the sharpened form.

Neither is assumed anywhere in the design. Both have explicit pass/fail.
