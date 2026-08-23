"""Adaptive Equilibrium MeshNet (Variant M of adaptive_equilibrium_design_v2.md).

Two coupled fixed points:
  inner contour  z : (B, C, D, H, W) voxel activations
  outer contour  y : (B, C_y)        regime-selector vector

Joint map (one Jacobi sweep):
  z' = f(z, y, x)   -- weight-tied dilated conv stack, multiplicatively gated by m(y)
  y' = g(y, z)      -- damped-tanh MLP on pooled z (optionally on solve residuals)

Forward: safeguarded type-II Anderson on the block-normalized joint residual
(Sec. 5), with optional per-site residual gating (Sec. 5.3, asynchronous
relaxation).  Backward: RBP (Algorithm 1) with block-wise stopping (Sec. 6),
wrapped in a custom autograd.Function so the tape holds exactly ONE evaluation
of H.  Stability: per-layer L-infinity (row-sum) weight scaling (Sec. 7.2).

Deviations from the doc (each measurable via config):
  * `gate_position: post_norm` (default) applies m(y) AFTER GroupNorm, because
    a per-channel gate applied before a per-channel/per-small-group
    normalization is largely cancelled by that normalization -- GN divides out
    the scale m just injected, so the intended spectrum modulation mostly dies.
    `pre_norm` reproduces the doc's Sec. 3.4 literally for the ablation.
  * `dilations: [1, 3, 9, 27]` (default) instead of the doc's single d=4: a
    single dilation-4 3^3 conv grows the receptive field by only 8 voxels per
    sweep, so covering a 256 cube needs ~32 sweeps -- the MDEQ receptive-field
    bound the doc itself warns about (Sec. 3.5).  The ternary weight-tied
    micro-stack grows RF by 80/sweep (4 sweeps cover 256^3) and each stage
    exactly tiles the previous one's footprint, so the per-sweep RF is
    HOLE-FREE at every sweep -- the same no-gridding doctrine as the
    modelAE_hdc_deep coprime ramp, applied to the per-sweep set (with weight
    tying, the composed schedule is the per-sweep set repeated K times, so
    density must hold within the set itself).  Set `dilations: [4]` to
    recover the doc's M-A, or the full 13-rate hdc_deep ramp to iterate the
    existing explicit schedule as the equilibrium map (ablation 8).
  * Anderson safeguarding is retrospective (reject at the NEXT sweep's already
    -paid residual evaluation) instead of the doc's immediate re-evaluation,
    which would double the per-sweep FLOPs.

The model obeys the repo's trainer contract: `model(x) -> logits` under
autocast; loss/backward handled by the runner.  Aux-head logits and the
Jacobian penalty are stashed on the module for the criterion wrapper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #

def _get(cfg, key, default):
    if cfg is None:
        return default
    v = cfg.get(key, default)
    return default if v is None else v


def _lin(layer, t, *a, **kw):
    """Apply a Linear whatever the input dtype. The RBP backward rebuilds the
    tape from tensors saved by an autocast forward, so bf16 activations can
    meet fp32 weights there ("mat1 and mat2 must have the same dtype"). Under
    autocast this cast is a no-op (autocast re-casts for the matmul anyway);
    outside it, the layer simply runs in its own precision."""
    return layer(t.to(layer.weight.dtype), *a, **kw)


def _act(name):
    if name == "softplus":
        return F.softplus
    if name == "elu":
        return F.elu
    if name == "gelu":
        return F.gelu
    raise ValueError(name)


def _block_norm(rz, ry):
    """Sec. 5.1 block-normalized joint residual norm (fp32 scalar tensor)."""
    nz = rz.float().norm() / math.sqrt(rz.numel())
    ny = ry.float().norm() / math.sqrt(ry.numel())
    return torch.sqrt(nz * nz + ny * ny)


def _state_norm(z, y):
    nz = z.float().norm() / math.sqrt(z.numel())
    ny = y.float().norm() / math.sqrt(y.numel())
    return torch.sqrt(nz * nz + ny * ny) + 1e-8


# --------------------------------------------------------------------------- #
# the fixed-point autograd.Function  (Sec. 6.4 wiring, Sec. 6.2 backward)
# --------------------------------------------------------------------------- #

class _AEQSolve(torch.autograd.Function):
    """no_grad Anderson solve forward; RBP with block-wise stopping backward.

    Params are passed as explicit inputs so DDP's grad hooks fire and so the
    returned grads land in .grad through the normal engine path.
    """

    @staticmethod
    def forward(ctx, x, y0, module, *params):
        with torch.no_grad():
            z_star, y_star = module._solve(x, y0)
        ctx.module = module
        # Record the forward's autocast state. The autograd engine runs
        # backward with autocast DISABLED, so without this the tape below is
        # built under a different dtype policy than the solve that produced
        # z*/y* -- bf16 saved tensors then meet fp32 Linear weights
        # ("mat1 and mat2 must have the same dtype"). Beyond the crash, the
        # implicit gradient must be taken w.r.t. the SAME map the forward
        # solved, so matching the policy is correctness, not just plumbing.
        _dev = "cuda" if x.is_cuda else "cpu"
        try:
            ctx.amp_enabled = bool(torch.is_autocast_enabled(_dev))
        except TypeError:      # older torch: no device arg
            ctx.amp_enabled = bool(torch.is_autocast_enabled()
                                   or torch.is_autocast_cpu_enabled())
        try:
            ctx.amp_dtype = torch.get_autocast_dtype("cuda" if x.is_cuda else "cpu")
        except Exception:                       # older torch
            ctx.amp_dtype = (torch.get_autocast_gpu_dtype() if x.is_cuda
                             else torch.bfloat16)
        ctx.amp_device = "cuda" if x.is_cuda else "cpu"
        ctx.save_for_backward(x.detach(), z_star.detach(), y_star.detach())
        return z_star, y_star

    @staticmethod
    def backward(ctx, grad_z, grad_y):
        module = ctx.module
        x, z_star, y_star = ctx.saved_tensors
        bw = module.bw_cfg
        if grad_z is None:
            grad_z = torch.zeros_like(z_star)
        if grad_y is None:
            grad_y = torch.zeros_like(y_star)

        params = module._eq_params()
        import contextlib
        # Rebuild the tape under the FORWARD's autocast policy (see forward).
        # bw_amp=True forces bf16 even if the forward ran in fp32.
        if getattr(ctx, "amp_enabled", False) or (module.bw_amp and x.is_cuda):
            amp_ctx = torch.autocast(
                getattr(ctx, "amp_device", "cuda" if x.is_cuda else "cpu"),
                dtype=getattr(ctx, "amp_dtype", torch.bfloat16))
        else:
            amp_ctx = contextlib.nullcontext()
        with torch.enable_grad(), amp_ctx:
            z_in = z_star.detach().requires_grad_(True)
            y_in = y_star.detach().requires_grad_(True)
            x_in = x.detach().requires_grad_(True)
            # ONE differentiable evaluation of the joint map -- the entire tape.
            fz, fy = module._H(z_in, y_in, x_in)

            lz, ly = grad_z, grad_y
            frozen_z = frozen_y = False
            iters = 0
            for k in range(int(bw["max_iter"])):
                iters = k + 1
                vz, vy = torch.autograd.grad(
                    (fz, fy), (z_in, y_in), (lz, ly),
                    retain_graph=True, allow_unused=True,
                )
                if vz is None:
                    vz = torch.zeros_like(lz)
                if vy is None:
                    vy = torch.zeros_like(ly)
                nlz, nly = vz + grad_z, vy + grad_y
                dz = (nlz - lz).norm() / math.sqrt(lz.numel())
                dy = (nly - ly).norm() / math.sqrt(ly.numel())
                if not frozen_z:
                    lz = nlz
                if not frozen_y:
                    ly = nly
                # Sec. 6.2 block-wise adjoint stopping
                frozen_z = frozen_z or bool(dz < bw["eps"])
                frozen_y = frozen_y or bool(dy < bw["eps"])
                if frozen_z and frozen_y:
                    break
            module.stats["bwd_iters"] = iters
            # Pinned at the cap => the adjoint is TRUNCATED, not converged.
            # Unlike the forward, this loop is plain unaccelerated Richardson
            # (e_{k+1} = J^T e_k), so it needs ~24 iterations at rho=0.7 for
            # 1e-3 (design Sec. 6.3) and cannot be expected to beat the
            # Anderson-accelerated forward. If this sits at 1, either lower
            # stability.rowsum_target (fewer iterations needed) or switch the
            # adjoint to a Krylov solver, which is the doc's own suggestion.
            module.stats["bwd_hit_cap"] = (
                1.0 if iters >= int(bw["max_iter"]) else 0.0)

            grads = torch.autograd.grad(
                (fz, fy), [x_in] + list(params), (lz, ly),
                retain_graph=False, allow_unused=True,
            )
        gx = grads[0]
        gparams = tuple(
            (torch.zeros_like(p) if g is None else g)
            for g, p in zip(grads[1:], params)
        )
        # inputs were: x, y0, module, *params
        return (gx, None, None) + gparams


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

class AEQMeshNet(nn.Module):
    def __init__(self, in_channels, n_classes, channels, aeq=None):
        super().__init__()
        aeq = aeq or {}
        C = int(channels)
        self.n_classes = int(n_classes)
        self.C = C
        self.C_y = int(_get(aeq, "C_y", 64))
        self.coupling = str(_get(aeq, "coupling", "mult"))       # mult|add|none
        self.gate_position = str(_get(aeq, "gate_position", "post_norm"))
        self.m_max = float(_get(aeq, "m_max", 0.9))
        self.alpha_outer = float(_get(aeq, "alpha_outer", 0.5))
        self.dilations = list(_get(aeq, "dilations", [1, 3, 9, 27]))
        self.n_groups = int(_get(aeq, "n_groups", 8))
        if C % self.n_groups != 0:
            self.n_groups = math.gcd(C, self.n_groups) or 1

        sol = dict(_get(aeq, "solver", {}))
        self.sol_cfg = {
            "window_m": int(_get(sol, "window_m", 3)),
            "beta": float(_get(sol, "beta", 0.8)),
            "lambda_ridge": float(_get(sol, "lambda_ridge", 1e-4)),
            "eps": float(_get(sol, "eps", 1e-3)),
            "max_iter": int(_get(sol, "max_iter", 12)),
            "history_dtype": str(_get(sol, "history_dtype", "bfloat16")),
            "compile_step": bool(_get(sol, "compile_step", True)),
            "check_every": int(_get(sol, "check_every", 1)),
        }
        # Set False by the trainer on non-logging steps to skip the handful of
        # scalar stats that each cost a host sync (m_mean, y_absmax, ...).
        self.collect_stats = True
        self._compiled_f = None   # lazy torch.compile twin of _f (solve only)
        bwd = dict(_get(aeq, "backward", {}))
        self.bw_cfg = {
            "max_iter": int(_get(bwd, "max_iter", 32)),
            "eps": float(_get(bwd, "eps", 1e-4)),
        }
        self.bw_amp = bool(_get(bwd, "bwd_amp", False))

        gat = dict(_get(aeq, "site_gating", {}))
        self.gating_cfg = {
            "enabled": bool(_get(gat, "enabled", False)),
            "tau_0": float(_get(gat, "tau_0", 5e-4)),
            "learned_threshold": bool(_get(gat, "learned_threshold", True)),
        }
        out = dict(_get(aeq, "outer", {}))
        self.read_residual = bool(_get(out, "read_residual", False))
        self.outer_grad = str(_get(out, "outer_grad", "implicit"))
        # What summary of z the outer contour sees. MEASURED PROBLEM with
        # "mean": the spatial mean of a feature map over a whole brain is
        # almost identical from subject to subject, so y cannot tell inputs
        # apart and m(y) collapses to a STATIC channel mask
        # (aeq/m_std_across_inputs == 0). "mean_std" concatenates the spatial
        # standard deviation, which carries far more between-subject
        # variance, at the cost of doubling g's input width (so g_mlp[0]
        # reinitializes when resuming a "mean" checkpoint). y stays
        # shape-free either way, so resolution transfer is preserved.
        self.pool_mode = str(_get(out, "pool", "mean"))
        # COMMON-MODE REMOVAL. Measured: the pooled summary varies only ~1-2%
        # across inputs (pooling over a whole brain destroys between-subject
        # variance -- adding std pooling does NOT help, tested). g therefore
        # sees a nearly constant absolute level, and the deviation that
        # actually carries information is then attenuated four more times
        # (rowsum_target_y cap, alpha damping, tanh, zero-init gate) until
        # m(y) is input-blind (aeq/m_std_across_inputs == 0). Centering the
        # summary against an EMA of its own history and dividing by its EMA
        # std hands g the DEVIATION at O(1) scale instead. Buffers, not
        # parameters, so checkpoints stay compatible.
        self.center_pool = bool(_get(out, "center_pool", False))
        self.center_momentum = float(_get(out, "center_momentum", 0.01))

        stab = dict(_get(aeq, "stability", {}))
        # ADAPTIVE SPECTRAL CONTROL -- the only mechanism that actually bounds
        # rho under gate_position=post_norm. Measured failure without it: a
        # trained checkpoint reached rho = 3.22 (expansive), the solve residual
        # sat at ~1.0 and DREW UPWARD over 40 sweeps, and the implicit gradient
        # was therefore meaningless. Cause: GroupNorm is scale-invariant, so the
        # row-sum cap on conv weights is erased; the only surviving Jacobian
        # factors are the ones applied AFTER each GN -- the gate m, and the GN
        # affine gamma. gamma is excluded from weight decay by the repo's
        # optimizer, so it grew unchecked and rho went with it (gamma^L across
        # L layers beats m_max, which only scales linearly).
        # Fix: measure rho by power iteration every rho_every steps and fold a
        # detached scalar into f's output so that s*rho -> rho_target. The
        # scalar is a non-persistent buffer (checkpoint-safe) and it multiplies
        # the map, so it cannot be normalized away by any downstream GN.
        self.rho_target = float(_get(stab, "rho_target", 0.7))
        self.rho_every = int(_get(stab, "rho_every", 20))
        # DEFAULT OFF. Measured harm: driving rho down by scaling f's output
        # does not reduce the recurrence's own gain -- GroupNorm's 1/std factor
        # cancels the scale exactly. rho only falls because the (unscaled)
        # input injection comes to dominate, i.e. the mechanism works by
        # AMPUTATING the recurrence. In a real run f_scale ratcheted to its
        # 1e-3 floor (it is clamped <= 1, so it can only shrink) while the
        # optimizer grew the GN gammas to compensate, and macro_dice collapsed
        # to the background-only value. Do not enable without the structural
        # fix: make f residual with the normalization OUTSIDE the recurrence,
        # so weight/step-size norms actually control the Jacobian.
        self.rho_control = bool(_get(stab, "rho_control", False))
        self.rowsum_target = float(_get(stab, "rowsum_target", 0.7))
        self.rowsum_target_y = float(_get(stab, "rowsum_target_y", 0.9))
        self.include_gn_gamma = bool(_get(stab, "include_gn_gamma", True))

        loss = dict(_get(aeq, "loss", {}))
        self.gamma_jac = float(_get(loss, "gamma_jac", 0.1))
        # The Hutchinson penalty is the single most expensive component:
        # measured at 128^3, gamma>0 costs +54% step time and +86% peak memory
        # (8.73s/6.74GiB vs 5.67s/3.61GiB) because it is a double-backward
        # through the whole inner map. It is a REGULARIZER, so paying it every
        # step is not required: apply it every n-th step with the weight
        # scaled by n, which preserves its expected contribution to the
        # gradient at ~1/n the cost. 1 = every step (old behavior).
        self.jac_every = max(1, int(_get(loss, "jac_every_n_steps", 1)))
        # Set False by the trainer on steps that skip the penalty.
        self.jac_this_step = True

        ph = dict(_get(aeq, "phases", {}))
        self.unroll_K = int(_get(ph, "unroll_K", 5))

        # runtime phase switches (set by the runner / phase schedule)
        self.mode = "solve"            # "unroll" | "solve"
        self.act_name = "softplus"     # "softplus" -> "elu" at phase C
        self.gating_on = False         # phase D, only after the random-mask gate
        self.eval_max_iter = int(_get(sol, "eval_max_iter", self.sol_cfg["max_iter"]))

        L = len(self.dilations)
        self.L = L
        # per-layer row-sum target so the composed product stays <= target
        self.rowsum_target_layer = self.rowsum_target ** (1.0 / L)

        # ---- explicit stem / head (outside the loop) ------------------------
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, C, 3, padding=1, bias=False),
            nn.GroupNorm(self.n_groups, C, affine=True),
        )
        self.head = nn.Conv3d(C, n_classes, 1)
        self.aux_head = nn.Linear(self.C_y, n_classes)   # Sec. 7.3 L_aux

        # ---- inner map f: weight-tied dilated conv micro-stack --------------
        self.convs = nn.ModuleList([
            nn.Conv3d(C, C, 3, padding=d, dilation=d, bias=False)
            for d in self.dilations
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(self.n_groups, C, affine=True) for _ in self.dilations
        ])

        # ---- coupling m(y) (Sec. 3.3) ---------------------------------------
        if self.coupling == "mult":
            self.gate = nn.Linear(self.C_y, L * C)
        elif self.coupling == "add":
            self.gate = nn.Linear(self.C_y, L * C)
        else:
            self.gate = None

        # ---- outer map g (Sec. 3.2) ------------------------------------------
        self.pool_dim = C * (2 if self.pool_mode == "mean_std" else 1)
        g_in = self.C_y + self.pool_dim + (2 if self.read_residual else 0)
        # persistent=False keeps these OUT of state_dict. Catalyst's checkpoint
        # callback loads with strict=True, so a persistent buffer added after a
        # checkpoint was written is a hard "Missing key(s)" failure on resume.
        # They are running statistics with a ~1/momentum step horizon (~100
        # steps at 0.01), so re-warming after a restart is cheap, and the
        # init (mean 0, var 1) makes _center a no-op until they warm up.
        self.register_buffer("f_scale", torch.ones(()), persistent=False)
        self._rho_step = 0
        self.register_buffer("pool_ema_mean", torch.zeros(self.pool_dim),
                             persistent=False)
        self.register_buffer("pool_ema_var", torch.ones(self.pool_dim),
                             persistent=False)
        self.g_mlp = nn.Sequential(
            nn.Linear(g_in, 2 * self.C_y), nn.GELU(),
            nn.Linear(2 * self.C_y, self.C_y),
        )
        # y_0 = tanh(Linear(pool(stem(x))))  (Sec. 4). NOTE: under outer_grad=
        # implicit this Linear gets NO gradient (the equilibrium is init-
        # independent); it is a deterministic, bounded seed, nothing more.
        self.y_init = nn.Linear(C, self.C_y)

        # learned per-site freezing threshold  tau = tau_0 * exp(-Linear(y))
        self.tau_net = nn.Linear(self.C_y, 1) if self.gating_cfg["learned_threshold"] else None

        # stashed per-forward extras for the criterion wrapper
        self._aux_logits = None
        self._jac_penalty = None
        self.stats = {}

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        for m in [self.stem[0], self.head] + list(self.convs):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.g_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)  # mult: m = m_max/2 everywhere at init
        nn.init.xavier_normal_(self.y_init.weight, gain=0.5)
        nn.init.zeros_(self.y_init.bias)
        if self.tau_net is not None:
            nn.init.zeros_(self.tau_net.weight)
            nn.init.zeros_(self.tau_net.bias)

    # -------------------------------------------------------- stability caps
    def _scaled_conv_weight(self, i):
        """Sec. 7.2 infinity-norm (row-sum) cap, per layer, gate- and
        gamma-aware. Detached scalar scale -> plain scaled-weight gradient."""
        W = self.convs[i].weight
        rowsum = W.abs().sum(dim=(1, 2, 3, 4)).max()
        gain = self.m_max if self.coupling == "mult" else 1.0
        if self.include_gn_gamma and self.norms[i].weight is not None:
            gain = gain * self.norms[i].weight.detach().abs().max().clamp(min=1e-6)
        scale = (self.rowsum_target_layer / (gain * rowsum.detach() + 1e-12)).clamp(max=1.0)
        return W * scale

    def _scaled_linear_weight(self, lin, target):
        W = lin.weight
        rowsum = W.abs().sum(dim=1).max()
        scale = (target / (rowsum.detach() + 1e-12)).clamp(max=1.0)
        return W * scale

    def rowsum_report(self):
        """Effective (post-cap) per-layer row sums, for logging."""
        rep = {}
        with torch.no_grad():
            for i in range(self.L):
                W = self._scaled_conv_weight(i)
                rep[f"rowsum_l{i}"] = float(W.abs().sum(dim=(1, 2, 3, 4)).max())
        return rep

    # ------------------------------------------------------------- the maps
    def _gates(self, y):
        """Returns list of per-layer (B, C, 1, 1, 1) modulation tensors."""
        if self.gate is None:
            return None
        u = _lin(self.gate, y).view(y.shape[0], self.L, self.C)
        if self.coupling == "mult":
            u = self.m_max * torch.sigmoid(u)
        return u.permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _f(self, z, y, x):
        """Inner map. x is the stem output (input injection)."""
        act = _act(self.act_name)
        gates = self._gates(y)
        h = z
        for i in range(self.L):
            u = F.conv3d(h, self._scaled_conv_weight(i), None,
                         padding=self.dilations[i], dilation=self.dilations[i])
            if self.gate_position == "pre_norm":
                # doc-literal Sec. 3.4: gate -> inject -> norm
                if gates is not None:
                    u = gates[i] * u if self.coupling == "mult" else u + gates[i]
                if i == 0:
                    u = u + x
                u = self.norms[i](u)
            else:
                # post_norm (default): norm -> gate -> inject
                u = self.norms[i](u)
                if gates is not None:
                    u = gates[i] * u if self.coupling == "mult" else u + gates[i]
                if i == 0:
                    u = u + x
            h = act(u)
        if self.rho_control:
            h = h * self.f_scale
        return h

    def _g_pooled(self, y, s, q=None):
        """Outer map on an already-pooled summary s = pool(z)."""
        feats = [y, s]
        if self.read_residual:
            if q is None:
                q = torch.zeros(y.shape[0], 2, device=y.device, dtype=y.dtype)
            feats.append(q)
        u = torch.cat(feats, dim=1)
        u = self.g_mlp[1](_lin(self.g_mlp[0], u))
        _w = self._scaled_linear_weight(self.g_mlp[2], self.rowsum_target_y)
        u = F.linear(u.to(_w.dtype), _w, self.g_mlp[2].bias)
        a = self.alpha_outer
        return (1.0 - a) * y + a * torch.tanh(u)

    def _pool(self, z):
        """Global summary of z that g is allowed to see (Sec. 3.2: g reads z
        ONLY through a pool, which is what keeps y shape-free)."""
        mu = z.mean(dim=(2, 3, 4))
        if self.pool_mode != "mean_std":
            return mu
        sd = z.float().var(dim=(2, 3, 4), unbiased=False).add(1e-8).sqrt().to(mu.dtype)
        return torch.cat([mu, sd], dim=1)

    def _center(self, s):
        """Hand g the DEVIATION of the pooled summary, not its level. The EMA
        is updated from detached values only (it is a statistic, not a path
        for gradients) and only while training the solver."""
        if not self.center_pool:
            return s
        sd = s.detach().float()
        if self.training:
            mom = self.center_momentum
            batch_mean = sd.mean(dim=0)
            self.pool_ema_mean.mul_(1 - mom).add_(batch_mean, alpha=mom)
            dev = (sd - self.pool_ema_mean).pow(2).mean(dim=0)
            self.pool_ema_var.mul_(1 - mom).add_(dev, alpha=mom)
        scale = self.pool_ema_var.clamp(min=1e-12).sqrt()
        return ((s - self.pool_ema_mean.to(s.dtype)) / scale.to(s.dtype)).clamp(-8, 8)

    def _g(self, y, z, q=None):
        """Outer map: damped tanh MLP on [y, pool(z), (residual features)]."""
        return self._g_pooled(y, self._center(self._pool(z)), q=q)

    def _H(self, z, y, x, q=None):
        """Joint Jacobi sweep from tick-k values (Sec. 4)."""
        return self._f(z, y, x), self._g(y, z, q=q)

    def _solver_f(self, x):
        """torch.compile twin of _f, used ONLY inside the no_grad forward
        solve. Implicit differentiation is untouched: the RBP backward builds
        its VJPs against a separate EAGER evaluation of H at the fixed point
        (Sec. 6.4), so the compiled graph only ever helps FIND z*, never
        differentiate through it. Eager GN + gates + residual chains are
        unfused bandwidth-bound kernels -- this is the main watts lever.

        Dynamo guards on self.act_name, so the phase B->C activation switch
        triggers exactly one recompile (expected, logged by inductor)."""
        if not (self.sol_cfg.get("compile_step") and x.is_cuda):
            return self._f
        if self._compiled_f is None:
            try:
                self._compiled_f = torch.compile(self._f, dynamic=False)
            except Exception:
                self._compiled_f = self._f
        return self._compiled_f

    def _tau(self, y):
        t0 = self.gating_cfg["tau_0"]
        if self.tau_net is None:
            return torch.full((y.shape[0],), t0, device=y.device, dtype=y.dtype)
        return t0 * torch.exp(-_lin(self.tau_net, y).squeeze(-1)).clamp(max=1e3)

    def _y0(self, x):
        return torch.tanh(_lin(self.y_init, x.mean(dim=(2, 3, 4))))

    def _eq_params(self):
        """Parameters reachable through one evaluation of H (grads via RBP)."""
        mods = [self.convs, self.norms, self.g_mlp]
        if self.gate is not None:
            mods.append(self.gate)
        ps = []
        for m in mods:
            ps += [p for p in m.parameters() if p.requires_grad]
        return ps

    # ---------------------------------------------------------------- solver
    def _solve(self, x, y0, force_random_mask=None, record_traj=False, z0=None):
        """Safeguarded Anderson (type-II, Sec. 5.2) with optional per-site
        gating (Sec. 5.3). Runs under no_grad (callers ensure it).

        force_random_mask: float in (0,1) -> at every sweep freeze that
        fraction of sites at random (the Sec. 7.2 empirical gate)."""
        cfg = self.sol_cfg
        m_win = cfg["window_m"]
        beta = cfg["beta"]
        hdt = (torch.bfloat16 if cfg["history_dtype"] == "bfloat16" and x.is_cuda
               else torch.float32)
        max_iter = cfg["max_iter"] if self.training else self.eval_max_iter

        B = x.shape[0]
        z = (torch.zeros(B, self.C, *x.shape[2:], device=x.device, dtype=x.dtype)
             if z0 is None else z0.detach().clone())
        y = y0.detach().clone()

        Fh, Rh = [], []                      # map outputs / residuals (history)
        prev_res = None
        last_was_anderson = False
        prev_rnorm = None                    # scalar ||r_z|| of previous sweep
        # Plain (non-extrapolated) map output of the IMMEDIATELY PRECEDING
        # sweep. The safeguard reverts here, NOT to Fh[-1]: with check_every>1
        # the history entry can be several sweeps stale, so reverting to it
        # discarded that many sweeps of progress on every rejection (observed
        # as anderson_rejects pinned at ~2 with the solve never converging).
        prev_plain = None
        q = None
        traj = [] if record_traj else None
        active_frac_curve = []
        nfe = 0
        rejects = 0
        final_res = float("nan")
        # The implicit function theorem applies AT a fixed point. If the solve
        # exits on the iteration cap instead of the residual test, the RBP
        # gradient is the exact gradient of an equilibrium we never reached --
        # a biased gradient whose error scales with the residual. Training
        # tolerates this (cf. JFB / one-step DEQ gradients), but it must be
        # VISIBLE: aeq/fwd_converged is the fraction of logged steps that
        # actually hit eps. If it drifts below 1, either raise solver.max_iter
        # or lower stability.rowsum_target so the map contracts harder.
        converged = False
        f_step = self._solver_f(x)
        # Convergence test + safeguard need HOST scalars, i.e. a CPU<->GPU sync
        # that drains the pipeline. check_every>1 evaluates them every k-th
        # sweep only (always on the last). Tradeoff: a bad Anderson
        # extrapolation survives up to k-1 sweeps before being reverted (safe,
        # because rho is bounded by the row-sum cap), and the solve can
        # overshoot convergence by up to k-1 sweeps -- so large k can COST
        # more compute than the syncs it saves. 1 == exact old behavior.
        check_every = max(1, int(cfg.get("check_every", 1)))

        for k in range(max_iter):
            # record (pool(z_k), q_k) BEFORE the sweep: this is exactly what
            # fy = g(y_k, z_k, q_k) reads, so the detached-BPTT replay
            # (Sec. 6.5) sees the same inputs the solve saw.
            if record_traj:
                q_rec = q if q is not None else torch.zeros(
                    y.shape[0], 2, device=y.device, dtype=y.dtype)
                traj.append((self._center(self._pool(z)).detach().clone(),
                             q_rec.detach().clone()))

            try:
                fz = f_step(z, y, x)
            except Exception:
                if f_step is not self._f:      # compiled twin failed at call
                    self._compiled_f = f_step = self._f
                    fz = f_step(z, y, x)
                else:
                    raise
            fy = self._g(y, z, q=q)
            nfe += 1
            rz, ry = fz - z, fy - y
            do_check = (k % check_every == 0) or (k == max_iter - 1)
            if do_check:
                # ONE sync per check: residual and state norm in a single
                # host transfer, so prev_res stays a float and the safeguard
                # comparison below needs no further device->host traffic.
                r_abs, s_abs = torch.stack(
                    [_block_norm(rz, ry), _state_norm(fz, fy)]).tolist()
                rel = r_abs / max(s_abs, 1e-12)
                final_res = rel

            # residual features for the closed loop (Sec. 3.2)
            if self.read_residual:
                rn = rz.detach().flatten(1).float().norm(dim=1) \
                    / math.sqrt(rz[0].numel())
                q1 = torch.log10(rn + 1e-9).clamp(-9, 2) / 9.0
                if prev_rnorm is None:
                    q2 = torch.zeros_like(q1)
                else:
                    q2 = torch.log((rn + 1e-9) / (prev_rnorm + 1e-9)).clamp(-2, 2)
                prev_rnorm = rn
                q = torch.stack([q1, q2], dim=1).to(y.dtype)

            if do_check:
                if rel < cfg["eps"]:
                    z, y = fz, fy
                    converged = True
                    break

                # retrospective safeguard: the Anderson candidate we accepted
                # at the last CHECKED sweep made things worse -> revert to the
                # plain step we stored then. Pure float comparison, no sync.
                if (last_was_anderson and prev_res is not None
                        and r_abs > prev_res and prev_plain is not None):
                    rejects += 1
                    z_prev_map, y_prev_map = prev_plain
                    z = z_prev_map.to(z.dtype)
                    y = y_prev_map.to(y.dtype)
                    Fh, Rh = Fh[-1:], Rh[-1:]
                    last_was_anderson = False
                    prev_res = None
                    continue
                prev_res = r_abs

            # remember this sweep's PLAIN map output as the safeguard's
            # one-sweep-back fallback (cheap: two tensors in history dtype).
            prev_plain = (fz.to(hdt), fy.to(hdt))

            # push MAP OUTPUT + normalized residual (Sec. 5.2)
            Fh.append((fz.to(hdt), fy.to(hdt)))
            Rh.append(torch.cat([
                (rz / math.sqrt(rz.numel())).flatten(),
                (ry / math.sqrt(ry.numel())).flatten(),
            ]).to(hdt))
            if len(Fh) > m_win:
                Fh.pop(0), Rh.pop(0)

            if len(Rh) > 1:
                R = torch.stack([r.float() for r in Rh])         # (n, N)
                G = R @ R.t()
                n = G.shape[0]
                G = G + cfg["lambda_ridge"] * (torch.diagonal(G).sum() / n) \
                    * torch.eye(n, device=G.device)
                try:
                    v = torch.linalg.solve(G, torch.ones(n, device=G.device))
                    alpha = (v / v.sum()).to(x.dtype)
                except Exception:
                    alpha = None
                if alpha is not None and torch.isfinite(alpha).all():
                    cz = sum(a * fz_i.to(x.dtype) for a, (fz_i, _) in zip(alpha, Fh))
                    cy = sum(a * fy_i.to(x.dtype) for a, (_, fy_i) in zip(alpha, Fh))
                    # damping beta mixes map outputs with iterates; we only kept
                    # map outputs, so damp toward the current iterate instead.
                    cand_z = beta * cz + (1 - beta) * z
                    cand_y = beta * cy + (1 - beta) * y
                    last_was_anderson = True
                else:
                    cand_z, cand_y = fz, fy
                    last_was_anderson = False
            else:
                cand_z, cand_y = fz, fy
                last_was_anderson = False

            # per-site residual gating (asynchronous relaxation, Sec. 5.3)
            use_gate = (self.gating_on and self.gating_cfg["enabled"]) \
                or force_random_mask is not None
            if use_gate:
                if force_random_mask is not None:
                    active = (torch.rand_like(z[:, :1]) > force_random_mask)
                else:
                    r_site = rz.detach().norm(dim=1, keepdim=True) / math.sqrt(self.C)
                    tau = self._tau(y).view(-1, 1, 1, 1, 1)
                    active = (r_site > tau)
                if do_check and self.collect_stats:   # else: one sync per sweep
                    active_frac_curve.append(float(active.float().mean()))
                z = torch.where(active, cand_z, z)
            else:
                z = cand_z
            y = cand_y

        self.stats.update({
            "nfe": nfe,
            "fwd_rel_res": final_res,        # already synced by the last check
            "fwd_converged": 1.0 if converged else 0.0,
            "fwd_hit_cap": 1.0 if nfe >= max_iter and not converged else 0.0,
            "anderson_rejects": rejects,
            "active_frac_last": active_frac_curve[-1] if active_frac_curve else 1.0,
            "active_frac_curve": active_frac_curve,
        })
        # Each of these is a separate device->host sync; only pay for them on
        # steps the trainer actually logs (see AEQRunner.handle_batch).
        if self.collect_stats:
            ya = y.detach().abs()
            self.stats["y_absmax"] = float(ya.max())
            # y = tanh(...), so |y| -> 1 means SATURATION: the local derivative
            # is 1-y^2 (0.04 at |y|=0.98), gradients to the outer contour
            # nearly vanish and y sticks at a corner regardless of input --
            # the persistent-excitation failure of design Sec. 13. Lower
            # stability.rowsum_target_y to shrink g's pre-activation.
            self.stats["y_sat_frac"] = float((ya > 0.9).float().mean())
            self.stats["y_absmean"] = float(ya.mean())
            if self.coupling == "mult" and self.gate is not None:
                g = self._gates(y)                      # (L, B, C, 1, 1, 1)
                self.stats["m_mean"] = float(g.mean())
                self.stats["m_absmax"] = float(g.max())
                self.stats["m_absmin"] = float(g.min())
                # Spread of m WITHIN one forward (across layers and channels).
                # NOTE: this is NOT input dependence -- with per-GPU batch 1
                # there is no batch axis to vary. The across-INPUT spread is
                # measured in the trainer by all-reducing m_mean over the DDP
                # ranks, which each hold a different volume (aeq/m_std_across
                # _inputs). That is the Sec. 8.2 / Sec. 13 quantity.
                self.stats["m_std_within"] = float(g.std())
        if record_traj:
            return z, y, traj
        return z, y

    # ------------------------------------------------------------ trajectories
    def _replay_y(self, y0, traj):
        """detached-BPTT (Sec. 6.5): re-run the tiny y-recurrence
        differentiably on recorded, detached pooled summaries."""
        y = y0
        for s_k, q_k in traj:
            y = self._g_pooled(y, s_k, q=q_k)
        return y

    # ---------------------------------------------------------------- forward
    def _forward_unrolled(self, x, y0):
        """Phase A: ordinary backprop through unroll_K Jacobi sweeps
        (per-sweep gradient checkpointing keeps memory at ~one sweep)."""
        z = torch.zeros(x.shape[0], self.C, *x.shape[2:],
                        device=x.device, dtype=x.dtype)
        y = y0

        def sweep(z_, y_, x_):
            return self._H(z_, y_, x_)

        for _ in range(self.unroll_K):
            if self.training and torch.is_grad_enabled():
                z, y = checkpoint(sweep, z, y, x, use_reentrant=False)
            else:
                z, y = sweep(z, y, x)
        self.stats.update({"nfe": self.unroll_K, "fwd_rel_res": float("nan")})
        return z, y

    def _ddp_zero_guard(self, ref):
        """Zero-valued scalar touching params that may otherwise sit outside
        the autograd graph (tau_net; y_init under implicit outer_grad;
        aux_head when lambda_aux=0), so DDP never sees unused parameters."""
        mods = [self.y_init, self.aux_head]
        if self.tau_net is not None:
            mods.append(self.tau_net)
        s = sum(p.sum() for m in mods for p in m.parameters())
        return (s * 0.0).to(ref.dtype)

    def forward(self, x):
        self.stats = {}

        if not self.training:
            # inference: plain no_grad solve, no autograd.Function involved
            with torch.no_grad():
                xin = self.stem(x)
                y0 = self._y0(xin)
                if self.mode == "unroll":
                    z_star, y_star = self._forward_unrolled(xin, y0)
                else:
                    z_star, y_star = self._solve(xin, y0)
                self._aux_logits = None
                self._jac_penalty = None
                return self.head(z_star)

        xin = self.stem(x)
        y0 = self._y0(xin)

        if self.mode == "unroll":
            z_star, y_star = self._forward_unrolled(xin, y0)
        else:
            use_bptt = (self.read_residual
                        and self.outer_grad == "detached_bptt")
            if use_bptt:
                with torch.no_grad():
                    _, _, traj = self._solve(xin.detach(), y0.detach(),
                                             record_traj=True)
            z_star, y_star = _AEQSolve.apply(
                xin, y0, self, *self._eq_params())
            if use_bptt and len(traj) > 0:
                y_traj = self._replay_y(y0, traj)
                # value = implicit y*, gradient = adjoint + trajectory paths
                y_star = y_star + (y_traj - y_traj.detach())

        logits = self.head(z_star)
        logits = logits + self._ddp_zero_guard(logits)

        self._aux_logits = _lin(self.aux_head, y_star)
        if self.mode == "solve":
            self._rho_step += 1
            # also fire on the FIRST step: f_scale is non-persistent, so a
            # resumed run starts at 1.0 and would otherwise spend rho_every
            # steps solving a possibly-divergent map before correcting.
            due = (self.rho_control and self.training
                   and (self._rho_step == 1
                        or self._rho_step % max(1, self.rho_every) == 0))
            if due or (self.collect_stats):
                try:
                    r = self._rho_estimate(z_star, y_star, xin)
                    self.stats["rho_est"] = r
                    self.stats["f_scale"] = float(self.f_scale)
                    if due and r > 0:
                        # s <- s * target/rho, clamped. rho was measured WITH
                        # the current s applied, so this is a feedback update
                        # and converges to rho == rho_target.
                        new = float(self.f_scale) * (self.rho_target / r)
                        self.f_scale.fill_(min(1.0, max(1e-3, new)))
                except Exception:
                    pass
        want_jac = (self.gamma_jac > 0 and self.mode == "solve"
                    and self.jac_this_step)
        # weight scaled by jac_every so the expected penalty is unchanged
        self._jac_penalty = (
            self._hutchinson_penalty(z_star, y_star, xin) * float(self.jac_every)
            if want_jac else None)
        return logits

    def _rho_estimate(self, z_star, y_star, xin, iters=3):
        """Power-iteration estimate of rho(dF/dz) at the fixed point.

        WHY THIS EXISTS: the Sec. 7.2 row-sum cap is INERT under
        gate_position=post_norm. GroupNorm is scale-invariant --
        GN(a*W*z) = GN(W*z) -- so scaling the conv weights (what the cap
        does) is cancelled by the GN that follows. Measured: sweeping
        rowsum_target 0.9 -> 0.1 moves rho only 0.369 -> 0.345. Under
        pre_norm the cap does bite, because the injected x breaks the scale
        invariance. So in the default mode rho is set by m_max and the GN/act
        gains, NOT by the stability target, and it must be measured rather
        than assumed. rho >= 1 means the solve can diverge and the
        asynchronous site-gating licence (rho(|J|) < 1) is void.
        """
        z_in = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            out = self._f(z_in, y_star.detach(), xin.detach())
            v = torch.randn_like(z_in)
            v = v / (v.norm() + 1e-12)
            r = float("nan")
            for _ in range(iters):
                (Jv,) = torch.autograd.grad(out, z_in, v, retain_graph=True)
                n = Jv.norm()
                r = float(n)
                v = Jv / (n + 1e-12)
        return r

    def _hutchinson_penalty(self, z_star, y_star, xin):
        """gamma * ||F_z||_F^2, one Rademacher probe (Sec. 7.3). Second-order
        through ONE evaluation of f."""
        z_in = z_star.detach().requires_grad_(True)
        fz = self._f(z_in, y_star.detach(), xin.detach())
        eps = torch.randint_like(fz, 0, 2) * 2.0 - 1.0
        (v,) = torch.autograd.grad(fz, z_in, eps, create_graph=True)
        return (v * v).sum() / v.numel()

    # convenience for the criterion wrapper
    def pop_extra_losses(self):
        aux, jac = self._aux_logits, self._jac_penalty
        self._aux_logits, self._jac_penalty = None, None
        return aux, jac

    # ---------------------------------------------------------------- gates
    @torch.no_grad()
    def random_mask_gate(self, x, mask_frac=0.5):
        """Sec. 7.2 mandatory empirical gate: does the asynchronously-frozen
        solve land on the synchronous fixed point (within 2*eps_f)?"""
        was = self.training
        self.eval()
        try:
            xin = self.stem(x)
            y0 = self._y0(xin)
            z_s, y_s = self._solve(xin, y0)
            z_a, y_a = self._solve(xin, y0, force_random_mask=mask_frac)
            gap = float(_block_norm(z_a - z_s, y_a - y_s)
                        / _state_norm(z_s, y_s))
        finally:
            self.train(was)
        passed = gap < 2 * self.sol_cfg["eps"]
        self.stats["async_gate_gap"] = gap
        self.stats["async_gate_pass"] = float(passed)
        return passed, gap


def build_aeq_model(in_channels, n_classes, channels, aeq_cfg):
    return AEQMeshNet(in_channels, n_classes, channels, aeq=aeq_cfg)
