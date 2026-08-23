# Adaptive Equilibrium MeshNet — findings and post-mortem

Record of what was built, what was measured, and what it means. Companion to
`adaptive_equilibrium_design_v2.md` (the design) and `AEQ_GUIDE.md` (how to run
it). Written after ~6 multi-hour runs on 4×H100 at 128³/256³, 18-class synth.

**Verdict: the equilibrium formulation as specified does not work for this
problem, and the deeper reason is that it was solving the wrong problem.**
Nothing here says two-contour equilibrium models are a bad idea in general —
it says this instantiation, on volumetric segmentation with a
GroupNorm-normalized weight-tied trunk, is not the way to get robustness at
fixed inference memory.

---

## 1. The decisive experiment

Identical trunk, data, loss, and schedule. Only the training formulation differs.

| arm | formulation | macro_dice (train, 128³) |
|---|---|---|
| full AEQ | Anderson solve + RBP implicit gradient + `m(y)` gate | rose to 0.07 by 20k, **collapsed to 0.049 and stayed flat for 170k steps** |
| arm 1 | same trunk, unrolled K=5, ordinary backprop, no gate, no solver | 0.05 → **0.19 at 200k, still rising** |

0.049 is arithmetically "predict background everywhere" (background dice ≈ 0.9,
seventeen zeros, ÷18 ≈ 0.05). So the full system was not learning slowly — it
was not learning at all, while the same trunk without the equilibrium was.

Two independent conclusions:

1. **The equilibrium machinery actively suppressed learning.** Ratio matters
   more than the absolute numbers: 0.05 vs 0.19 on identical everything else.
2. **The trunk is also under-capacity.** 0.19 after 200k steps is not viable.
   4 tied 3³ kernels at 16ch = 54K params, vs `gn_hdc_deep`'s 13 distinct
   layers. Weight tying at this width costs more than it buys.

---

## 2. Root cause: GroupNorm inside the iterated map

The single most important technical finding, and the one that invalidated the
design's entire stability apparatus.

**GroupNorm is scale-invariant: `GN(αu) = GN(u)`, and its Jacobian carries a
compensating `1/α`.** With a GN after every conv inside the weight-tied map,
nothing outside the map can control the Jacobian's spectral radius:

| mechanism | intended effect | measured effect |
|---|---|---|
| §7.2 ∞-norm row-sum cap on conv weights | bound ρ | **inert.** Sweeping `rowsum_target` 0.9 → 0.1 (9×) moved ρ from 0.369 to 0.345 |
| `m_max` (gate ceiling, applied *after* GN) | bound ρ | **works** — ρ ∝ roughly `m_max^L` (0.9→0.050, 0.6→0.017, 0.3→0.0025). Survives because it is post-GN |
| `f_scale` (scalar on `f`'s output) | bound ρ | **destructive.** Reduces ρ only by making the *unscaled* input injection dominate, i.e. by amputating the recurrence. Ratcheted to its 1e-3 floor while the optimizer grew GN γ to compensate; macro_dice collapsed |

The only Jacobian factors that survive GN are those applied **after** it: the
gate `m`, the GN affine `γ`, and the activation slope. And `γ` is excluded from
weight decay by `get_optimizer`, so it grew unchecked. Consequence, measured on
a trained checkpoint with `aeq_solvetrace.py`:

```
rho(dF/dz) = 3.22          # expansive
residual  ~1.0 for 40 sweeps, DRIFTING UP (0.963 -> 1.047)
bf16 == fp32 to 4 digits   # not a precision artifact
```

**With ρ > 1 there is no fixed point, the solve cannot converge, and
`(I − J)⁻¹` has no convergent Neumann series — so every RBP gradient in every
run before this was taken at a non-equilibrium point of a divergent adjoint.**
That is the mechanical explanation for the flat 0.05.

The structural fix, if anyone revisits this: make the inner map residual with
the normalization **outside** the recurrence,

```
z' = z + η · act(conv(z)·m(y) + x)        η bounded, no GN inside f
```

so that ρ = ‖I + η·J_φ‖ is genuinely controlled by `η` and by the conv's
spectral norm — because with no GN to erase it, weight normalization finally
bites. This is the formulation §7.2's mathematics was actually written for.
**Not attempted here.**

---

## 3. The outer contour was decorative (Bet 1 lost, for a specific reason)

`aeq/m_std_across_inputs` — the spread of `m(y)` across the four *different*
volumes held by the four DDP ranks — read **exactly 0**, while `m_absmax` sat
pinned at `m_max` (0.898) and `m_absmin` fell to 0.0014. So `m(y)` became a
**static channel mask**: some channels learned fully on, others off, with no
dependence on the input. That is design §13's persistent-excitation failure.

The mechanism is not saturation (lowering `rowsum_target_y` to 0.4 took
`y_absmax` from 0.98 to ~0.85 and changed nothing). It is that **`g` reads `z`
only through a global pool, and the spatial mean of a feature map over an
entire brain varies only ~1–2% between subjects.** Measured, on synthetic
volumes:

| pooling | across-input std of the summary | relative to its own scale |
|---|---|---|
| `mean` | 0.00131 | 2.1% |
| `mean_std` | 0.00102 | 0.3% |

Adding std pooling does **not** help — pooling over a whole volume destroys
between-subject variance whichever moment you take. That ~1% signal is then
attenuated four more times (the row-sum cap on `g`'s output Linear, the α=0.5
damping, `tanh`, and the zero-initialized gate Linear) until it is
indistinguishable from zero.

`outer.center_pool` (centre the summary against an EMA of its own history,
divide by its EMA std) gave ~7× more absolute and ~20× more relative
across-input variation in `y` **at init**, on synthetic data. Never validated
in training. Available, not proven.

Related design-level worry: `L_aux` targets **per-class volume fractions**,
which are nearly identical across brains, so the auxiliary loss may be actively
teaching `y` to be constant. `aeq/aux_loss` sat flat at ~0.85 throughout.

**If the two-contour idea is retried, `y` must be conditioned on something with
real between-subject variance** — e.g. a low-resolution pass (downsample 4×,
tiny net) rather than a global pool. Note that this makes it FiLM conditioning,
which is well-trodden and cheap, and no longer needs an equilibrium.

---

## 4. Other measured results worth keeping

**The Jacobian (Hutchinson) penalty is not worth its cost here.** Measured at
128³: `gamma_jac > 0` costs **+54% step time and +86% peak memory** (8.73 s /
6.74 GiB vs 5.67 s / 3.61 GiB), because it is a double-backward through the
whole inner map. Its logged value was **~2 × 10⁻⁵**, contributing ~10⁻⁶ to the
loss. MDEQ's published 17→6 NFE win does not transfer: MDEQ had no post-GN gain
cap, so Jacobian regularization was its *only* handle on ρ. Default is now 0.0.

**bf16 puts a floor on the relative residual.** A fixed point converged to
7.7e-8 in fp32 reads **1.7e-3** when the state is rounded to bf16; a full bf16
solve plateaus ~1.1e-3 and never "converges". `f(z) − z` is a difference of
nearly-equal bf16 numbers, so cancellation eats every digit. **`eps_f = 1e-3`
is below what bf16 iterates can represent** — the design's threshold presumes
fp32 iterates while the trainer's `bit16: True` makes them bf16.

**Anderson acceleration hurt on this map.** When ρ was high, `anderson_rejects`
pinned at ~2 per solve — the extrapolation consistently made the residual
worse, exactly as §5.2 warns can happen under multiplicative modulation. Plain
damped iteration (`window_m: 1`) was strictly more robust, and it also frees the
`2·m_f` history tensors.

**The adjoint never needed acceleration.** `bwd_iters` ran 1–5 whenever the
forward converged, `bwd_hit_cap` ≡ 0. The doc's suggested Krylov adjoint would
have spent `m·|z|` of memory — the scarce resource — to speed up a
non-bottleneck. Not needed.

**Cost.** ~2.2 s/it at 256³ on 4×H100 (micro-batch, `accum_steps=4`);
~1.30 s/it at 128³ in solve mode vs 0.296 s/it in unrolled mode. Roughly 120
conv-equivalents per step against the explicit model's 13. A trained model is
genuinely cheaper than an untrained one — adaptive depth works — but the
constant factor is large.

**Solve mode is latency-bound, not bandwidth-bound.** Power fell to ~195 W of
700 W with memory bandwidth at 0–12% during the sequential adjoint loop, which
is why `torch.compile` on the step bought only 2.8% and sync-trimming
(`check_every`) bought less.

**GPU memory growth is benign.** Live allocation is flat (0.11–0.36 GiB against
a 3.7 GiB peak); what `nvitop` shows is the caching allocator's high-water mark,
which PyTorch never returns. 200k steps, no OOM. `expandable_segments:True` did
not change it, so the fragmentation theory was probably wrong too — it simply
plateaus.

---

## 5. The strategic error

**The equilibrium formulation buys O(1) *training* memory. The constraint in
this project is *inference* memory.**

A weight-tied recurrent trunk already gives the inference profile we want — one
`z` buffer plus temporaries, independent of how many times the block is applied.
That is true whether the weights were trained by implicit differentiation or by
ordinary BPTT through a fixed unroll. Implicit differentiation adds: a
contraction constraint on the map, a solver, a convergence criterion, path
independence, and a stability apparatus — **none of which the deployment
requires.**

Arm 1 demonstrates this concretely: it has exactly the inference memory profile
of the full AEQ model, trains with plain backprop and per-sweep gradient
checkpointing, has no ρ constraint, and learns 4× better.

---

## 6. Process lessons

1. **Instrument the mechanism before building on it.** `rho_est` — four lines of
   power iteration — was written *after* the row-sum cap, `m_max`, the
   Hutchinson penalty, and `f_scale`, all of which assumed ρ was being
   controlled. It immediately showed that none of them worked. It should have
   been the first thing written.
2. **A test suite that never enters the failing regime proves nothing.**
   `test_aeq.py` runs in fp32 and never triggered an Anderson rejection, so it
   missed both the bf16 residual floor and the safeguard-staleness bug. Tests
   must exercise the paths that fire in production (autocast, rejections,
   non-convergence).
3. **Implementing a design faithfully is not the same as validating it.** The
   ∞-norm apparatus was built exactly as specified in §7.2 and was inert from
   the first line, because §3.4 puts GroupNorm inside the map. The two sections
   are mutually incompatible and the doc never says so.
4. **Bisect the architecture before debugging its parts.** Arm 1 cost 4 hours
   and answered more than five rounds of solver fixes. It should have been run
   the day the first flat curve appeared.
5. **Get the baseline number first.** "Is 0.19 good?" was unanswerable for the
   whole investigation because `gn_hdc_deep`'s macro_dice at matched step count
   was never pulled. Every judgement about "slow" was made without a reference.

---

## 7. Where to go instead

Ordered by expected value against the actual goal (robust tiny models at fixed
browser inference memory).

**A. Recurrent refinement without the equilibrium.** Take arm 1 seriously as
the architecture: weight-tied block, fixed K unrolls, BPTT with per-sweep
checkpointing. Same inference memory, no solver, no ρ constraint. Then fix the
capacity problem, which is the real blocker — widen the tied block, deepen the
micro-stack, or use *partial* tying (2–3 distinct blocks applied in rotation)
so parameter count is not tied to a single kernel set. Compare against
`gn_hdc_deep` at matched inference memory, not matched parameters.

**B. Attack fragility where it actually lives.** This repo's own prior finding
(`gn-hdc-deep-sim-to-real`) is that the tiny model's failure on real data is
dura mis-labelling — a **sim-to-real distribution** problem, and that the fix is
pseudo-labelling real volumes with the 18ch teacher and mixing them in. No
architectural change addresses that, and it is likely worth more than anything
in this document. The project framing ("less fragile via creative architectural
modifications") may simply be aimed at the wrong cause.

**C. Adaptive compute without a fixed point.** If the appeal was spending more
computation on ambiguous voxels (§5.3), do it as an early-exit cascade: run the
tiny model, use a cheap uncertainty signal (logit margin), re-run refinement
only where it is low. No equilibrium, no contraction requirement, and the
per-site compute map the design wanted comes out directly.

**D. Test-time augmentation.** Averaging predictions over flips/rotations costs
wall-clock, not peak memory, and reliably improves robustness. Boring, and
probably a better robustness-per-effort ratio than anything above.

**E. If the two-contour hypothesis is still interesting**, test it *without* the
equilibrium first: FiLM-condition the recurrent trunk on a low-resolution
summary pass (real between-subject variance, unlike a global pool) and measure
whether multiplicative conditioning beats additive at matched parameters. If
that shows nothing, the equilibrium version cannot rescue it. If it shows
something, *then* consider making the conditioner a second state.

---

## 8. What is in the repo

| file | status |
|---|---|
| `aeq_meshnet.py` | works, all crash bugs fixed; `rho_control` OFF (destructive), `gamma_jac` 0.0 (not worth cost) |
| `curriculum_training_aeq.py` | works; phases, `validation.enabled`, DDP-safe diagnostics |
| `conf/aeq_siam18_16ch.yaml`, `..._pre128.yaml` | 256³ and 128³ stages |
| `test_aeq.py` | Phase-0 gate passes (RBP vs unrolled autograd, cosine 1.000000). **Runs fp32 only — does not exercise autocast or the reject path** |
| `aeq_solvetrace.py` | the tool that found ρ = 3.22. Per-sweep residual, bf16 vs fp32, ρ |
| `aeq_memprobe.py` | per-phase alloc/peak/reserved with a leak verdict |
| `aeq_diagnostics.py` | §8.1 Schur coupling norm, ρ spread, truncation bias. **Never run on a trained checkpoint** — the definitive Bet 1 measurement is still unmade |

The Phase-0 gradient gate genuinely passes: RBP matches fully-unrolled autograd
at cosine 1.000000. The implicit differentiation is correctly implemented. It is
correct machinery applied to a map that does not satisfy its preconditions.
