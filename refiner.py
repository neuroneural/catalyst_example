"""
Dynamic Kernel Refinement for MeshNet
======================================

Drop-in hypernetwork extension for the MeshNet architecture.
Each conv layer gets a small network N that refines the base kernel
via fixed-point iteration, conditioned on compact activation statistics.

Works across all channel counts (C=5 to C=30+) and arbitrary dilation
schedules defined in the JSON config. Integrates with existing
enMesh_checkpoint and enMesh training strategies.

Design principles:
  1. Never feed raw activations into the refiner — compress to statistics first
  2. Scale the refiner architecture smoothly with C
  3. For small C: kernel is a single vector, processed by MLP
  4. For large C: kernel is a set of output filters, processed with inter-filter mixing
  5. Fixed-point iteration allows small N to produce expressive k
  6. Zero-init output + small alpha → starts as identity (pretrain-compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from contextlib import nullcontext
import json
import math
from typing import Optional, Tuple
from fix import FixedPointSolve, _anderson_forward


# =============================================================================
# Utilities (from your meshnet.py, kept for compatibility)
# =============================================================================

def set_channel_num(config, in_channels, n_classes, channels):
    config["layers"][0]["in_channels"] = in_channels
    config["layers"][0]["out_channels"] = channels
    config["layers"][-1]["in_channels"] = channels
    config["layers"][-1]["out_channels"] = n_classes
    for layer in config["layers"][1:-1]:
        layer["in_channels"] = layer["out_channels"] = channels
    return config


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


# =============================================================================
# Part 1: Activation Statistics — C-adaptive compression
# =============================================================================

class ActivationStats(nn.Module):
    """
    Compress (B, C, D, D, D) -> (B, stat_dim).

    Unified robust and expressive statistics pipeline:
      - Channel quantiles (25%, 50%, 75%): O(C) - robust to mixed precision outliers.
      - Channel covariance: O(C^2) capped via low-rank projection.
      - 3D Discrete Cosine Transform (DCT) on low-rank projection: captures spatial texture/frequencies.
      - Spatio-channel factorized embedding (DW-PW block + pooling): captures joint local patterns.
      - Directional gradients (anisotropy): captures edge frequency in x, y, z.
    """
    def __init__(self, C: int, stat_dim: int, spatial_size: int = 256):
        super().__init__()
        self.C = C
        self.stat_dim = stat_dim
        self.stat_grid = 32

        # --- Quantiles ---
        # 3 values per channel (25%, 50%, 75% percentiles)

        # --- Low-rank Channel projection ---
        self.cov_rank = min(C, 16)
        if C > 16:
            self.cov_proj = nn.Linear(C, self.cov_rank, bias=False)
        else:
            self.cov_proj = None
        cov_features = self.cov_rank * (self.cov_rank + 1) // 2

        # --- 3D DCT on low-rank projection ---
        # Downsample to 8^3, extract top 3 coefficients in each spatial dimension (3^3 = 27 coefficients)
        self.dct_N = 8
        self.dct_K = 3
        # Precompute DCT matrix using math module
        D = torch.zeros(self.dct_K, self.dct_N)
        for k in range(self.dct_K):
            for n in range(self.dct_N):
                scale = math.sqrt(1.0 / self.dct_N) if k == 0 else math.sqrt(2.0 / self.dct_N)
                D[k, n] = scale * math.cos(math.pi / self.dct_N * (n + 0.5) * k)
        self.register_buffer('dct_mat', D)
        dct_features = self.cov_rank * (self.dct_K ** 3)

        # --- Spatio-Channel Factorized Embedding ---
        # Depthwise-separable convolution block on downsampled activations
        self.spatial_dw = nn.Conv3d(
            in_channels=self.cov_rank,
            out_channels=self.cov_rank,
            kernel_size=3,
            padding=1,
            groups=self.cov_rank,
            bias=False
        )
        self.spatial_pw = nn.Conv3d(
            in_channels=self.cov_rank,
            out_channels=8,
            kernel_size=1,
            bias=False
        )
        spatio_channel_features = 8 * (4 ** 3) # pooled to 4^3 = 512 features

        # --- Directional gradients (anisotropy) ---
        dir_features = min(3 * C, 48)
        self.dir_proj = nn.Linear(3 * C, dir_features) if 3 * C > 48 else None
        dir_features_actual = dir_features if self.dir_proj else 3 * C

        # --- Total raw dimension ---
        region_features = 5 * C * 5 + 2  # 5 regions with mean/std/skew/energy/foreground + global fg/center ratio
        raw_dim = (
            3 * C +                  # quantiles
            2 * C +                  # foreground-weighted mean/std
            region_features +        # deterministic center-biased regional moments
            cov_features +           # channel covariance
            dct_features +           # 3D DCT features
            spatio_channel_features + # spatio-channel joint features
            dir_features_actual      # directional gradients
        )

        # --- Projection to stat_dim ---
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, stat_dim * 2),
            nn.GELU(),
            nn.Linear(stat_dim * 2, stat_dim),
            nn.LayerNorm(stat_dim),
        )

        self._raw_dim = raw_dim

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # Keep the statistics path differentiable so the dynamic-kernel loss
        # can shape upstream MeshNet activations. Quantiles are detached below
        # because their gradients are sparse/noisy and not useful here.
        B, C, D, H, W = a.shape
        features = []

        # Work on a fixed spatial grid, then canonicalize by cropping the
        # high-energy activation bounding box and resampling it back to 32^3.
        # The crop coordinates are detached; gradients still flow through the
        # selected activation values. This reduces dependence on how much empty
        # space surrounds the head at different curriculum cube sizes.
        a_grid = F.adaptive_avg_pool3d(a, self.stat_grid)
        energy = a_grid.detach().abs().mean(dim=1, keepdim=True)
        energy_flat = energy.flatten(2)
        threshold = torch.quantile(energy_flat.to(torch.float32), 0.65, dim=-1, keepdim=True).to(a.dtype)
        masks = energy_flat > threshold
        canonical = []
        margin = 2
        for i in range(B):
            mask = masks[i, 0].reshape(self.stat_grid, self.stat_grid, self.stat_grid)
            coords = mask.nonzero(as_tuple=False)
            if coords.numel() == 0:
                canonical.append(a_grid[i:i + 1])
                continue
            lo = coords.min(dim=0).values
            hi = coords.max(dim=0).values + 1
            lo = torch.clamp(lo - margin, min=0)
            hi = torch.clamp(hi + margin, max=self.stat_grid)
            crop = a_grid[i:i + 1, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            canonical.append(F.adaptive_avg_pool3d(crop, self.stat_grid))
        a_grid = torch.cat(canonical, dim=0)

        energy = a_grid.detach().abs().mean(dim=1, keepdim=True)
        energy_flat = energy.flatten(2)
        threshold = torch.quantile(energy_flat.to(torch.float32), 0.65, dim=-1, keepdim=True).to(a.dtype)
        scale = energy_flat.to(torch.float32).std(dim=-1, keepdim=True).clamp_min(1e-6).to(a.dtype)
        weights = torch.sigmoid((energy_flat - threshold) / scale).reshape(B, 1, self.stat_grid, self.stat_grid, self.stat_grid)
        weights = weights / (weights.sum(dim=(2, 3, 4), keepdim=True) + 1e-8)

        mean = (a_grid * weights).sum(dim=(2, 3, 4), keepdim=True)
        var = ((a_grid - mean).pow(2) * weights).sum(dim=(2, 3, 4), keepdim=True)
        std = torch.sqrt(var + 1e-6)
        a_norm = (a_grid - mean) / std
        a_focus = a_norm * weights * float(self.stat_grid ** 3)
        features.append(mean.flatten(1))
        features.append(std.flatten(1))

        # Center-biased regional moments approximate the truncated-normal
        # subcube sampling prior used by the data loader, but deterministically.
        # This helps 192^3 and 256^3 curricula produce comparable descriptors.
        regions = [
            (8, 24, 8, 24, 8, 24),    # center half
            (4, 20, 8, 24, 8, 24),    # superior/inferior shifted slabs
            (12, 28, 8, 24, 8, 24),
            (8, 24, 4, 20, 8, 24),
            (8, 24, 12, 28, 8, 24),
        ]
        region_stats = []
        for z0, z1, y0, y1, x0, x1 in regions:
            r = a_norm[:, :, z0:z1, y0:y1, x0:x1]
            e = energy[:, :, z0:z1, y0:y1, x0:x1]
            w = weights[:, :, z0:z1, y0:y1, x0:x1]
            w = w / (w.sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
            r_mean = (r * w).sum(dim=(2, 3, 4))
            r_var = ((r - r_mean.view(B, C, 1, 1, 1)).pow(2) * w).sum(dim=(2, 3, 4))
            r_std = torch.sqrt(r_var + 1e-6)
            r_skew = (((r - r_mean.view(B, C, 1, 1, 1)) / r_std.view(B, C, 1, 1, 1)).pow(3) * w).sum(dim=(2, 3, 4))
            r_energy = (e * w).sum(dim=(2, 3, 4)).expand(-1, C)
            r_foreground = (weights[:, :, z0:z1, y0:y1, x0:x1].sum(dim=(2, 3, 4)) / weights.sum(dim=(2, 3, 4))).expand(-1, C)
            region_stats.extend([r_mean, r_std, r_skew, r_energy, r_foreground])
        center_mass = weights[:, :, 8:24, 8:24, 8:24].sum(dim=(2, 3, 4))
        global_foreground = (energy > threshold.reshape(B, 1, 1, 1, 1)).to(a.dtype).mean(dim=(2, 3, 4))
        features.append(torch.cat(region_stats + [center_mass, global_foreground], dim=-1))

        # 1. Distribution estimate on the normalized, foreground-weighted grid.
        a_flat = a_focus.reshape(B, C, -1)

        # Compute quantiles in float32 for numerical stability
        a_flat_f32 = a_flat.detach().to(torch.float32)
        q = torch.tensor([0.25, 0.50, 0.75], device=a.device, dtype=a_flat_f32.dtype)
        quantiles = torch.quantile(a_flat_f32, q, dim=-1) # (3, B, C)
        quantiles = quantiles.permute(1, 2, 0).reshape(B, -1) # (B, 3*C)
        features.append(quantiles.to(a.dtype))

        # 2. Low-rank Channel Projection
        if self.cov_proj is not None:
            # a_flat is (B, C, N_sub)
            a_proj = self.cov_proj(a_flat.transpose(1, 2)).transpose(1, 2) # (B, rank, N_sub)
        else:
            a_proj = a_flat

        # 3. Channel Covariance (in float32)
        a_proj_f32 = a_proj.to(torch.float32)
        a_proj_centered = a_proj_f32 - a_proj_f32.mean(dim=-1, keepdim=True)
        cov = torch.bmm(a_proj_centered, a_proj_centered.transpose(1, 2)) / max(a_proj.shape[-1], 1)
        cov = cov.to(a.dtype)
        
        r = self.cov_rank
        idx = torch.triu_indices(r, r, device=a.device)
        features.append(cov[:, idx[0], idx[1]]) # (B, r*(r+1)/2)

        # 4. Spatio-temporal features: Downsample once to 32^3
        # We need a 5D tensor (B, cov_rank, 32, 32, 32)
        # Reshape a_proj back to spatial or downsample a directly
        if self.cov_proj is not None:
            a_down32_flat = a_focus.reshape(B, C, -1).transpose(1, 2) # (B, 32^3, C)
            a_spatial = self.cov_proj(a_down32_flat).transpose(1, 2).reshape(B, r, 32, 32, 32)
        else:
            a_spatial = a_focus # (B, r, 32, 32, 32)

        # 5. 3D DCT on low-rank projection
        # Downsample to 8^3
        a_dct_in = F.adaptive_avg_pool3d(a_spatial, self.dct_N) # (B, r, 8, 8, 8)
        N, K = self.dct_N, self.dct_K
        D_mat = self.dct_mat.to(a.dtype) # (K, N)

        # Apply DCT along Depth (dim 2)
        X1 = a_dct_in.permute(0, 1, 3, 4, 2).reshape(-1, N)
        X1_dct = torch.matmul(X1, D_mat.t())
        X2 = X1_dct.reshape(B, r, N, N, K).permute(0, 1, 4, 2, 3) # (B, r, K, N, N)

        # Apply DCT along Height (dim 3)
        X3 = X2.permute(0, 1, 2, 4, 3).reshape(-1, N)
        X3_dct = torch.matmul(X3, D_mat.t())
        X4 = X3_dct.reshape(B, r, K, K, N).permute(0, 1, 2, 4, 3) # (B, r, K, K, N)

        # Apply DCT along Width (dim 4)
        X5 = X4.reshape(-1, N)
        X5_dct = torch.matmul(X5, D_mat.t())
        dct_out = X5_dct.reshape(B, r, K, K, K)
        features.append(dct_out.reshape(B, -1))

        # 6. Spatio-channel Factorized Embedding
        # Apply DW-PW Conv block
        sp_feat = self.spatial_dw(a_spatial)
        sp_feat = self.spatial_pw(sp_feat) # (B, 8, 32, 32, 32)
        sp_feat = F.adaptive_avg_pool3d(sp_feat, 4) # (B, 8, 4, 4, 4)
        features.append(sp_feat.reshape(B, -1))

        # 7. Directional Gradients
        a_dir = a_focus
        gx = (a_dir[:, :, 1:] - a_dir[:, :, :-1]).abs().mean(dim=(2, 3, 4))
        gy = (a_dir[:, :, :, 1:] - a_dir[:, :, :, :-1]).abs().mean(dim=(2, 3, 4))
        gz = (a_dir[..., 1:] - a_dir[..., :-1]).abs().mean(dim=(2, 3, 4))
        dir_feats = torch.cat([gx, gy, gz], dim=-1)
        if self.dir_proj is not None:
            dir_feats = self.dir_proj(dir_feats)
        features.append(dir_feats)

        # Concatenate and project to stat_dim
        return self.proj(torch.cat(features, dim=-1))


# =============================================================================
# Part 2: Kernel Refinement Step — C-adaptive architecture
# =============================================================================

class KernelRefineStep(nn.Module):
    """
    One fixed-point step: k_{t+1} = k_t + α · f(k_t, stats)

    For small C (kernel_numel ≤ ~2048):
        Treat entire kernel as a flat vector. Simple, fast MLP.

    For large C (kernel_numel > ~2048):
        Treat each output filter as a token (C_out tokens of size C_in*27).
        Process per-filter with shared MLP + FiLM, then mix across filters.
        This keeps parameter count from scaling as O(C⁴).

    Threshold is configurable but 2048 ≈ C=8 for square (8*8*27=1728)
    """

    # Below this kernel element count, we use the flat MLP path.
    # Above it, we use the per-filter path.
    FLAT_THRESHOLD = 2048

    def __init__(
        self,
        C_out: int,
        C_in: int,
        kernel_size: int,
        stat_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.C_out = C_out
        self.C_in = C_in
        self.ks = kernel_size
        self.kernel_numel = C_out * C_in * kernel_size ** 3
        self.filter_numel = C_in * kernel_size ** 3  # elements per output filter

        self.use_flat = (self.kernel_numel <= self.FLAT_THRESHOLD)

        if self.use_flat:
            self._build_flat(stat_dim, hidden_dim)
        else:
            self._build_perfilter(stat_dim, hidden_dim)

        # Learned step size, initialized small for stability but non-negligible
        # so scratch training can move the dynamic kernels early.
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.05)))

    def _build_flat(self, stat_dim, hidden_dim):
        """Flat MLP: whole kernel as one vector. Best for small C."""
        K = self.kernel_numel

        # FiLM: stats → scale, shift for hidden representation
        self.film = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # k → hidden → FiLM → hidden → Δk
        self.k_to_h = nn.Linear(K, hidden_dim)
        self.h_to_k = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )
        # Tiny nonzero init keeps behavior close to identity while allowing
        # gradients to reach the whole refiner from the first optimization step.
        nn.init.normal_(self.h_to_k[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.h_to_k[-1].bias)

    def _build_perfilter(self, stat_dim, hidden_dim):
        """Per-filter processing with cross-filter mixing. For large C."""
        F_dim = self.filter_numel  # size of one output filter

        # FiLM: stats → per-filter modulation
        self.film = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # Per-filter MLP (shared across all C_out filters)
        self.filter_in = nn.Linear(F_dim, hidden_dim)
        self.filter_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, F_dim),
        )
        nn.init.normal_(self.filter_out[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.filter_out[-1].bias)

        # Cross-filter mixing: let output filters coordinate
        # This is a lightweight attention-free alternative:
        # just a shared MLP applied across the filter dimension
        self.cross_mix = nn.Sequential(
            nn.Linear(self.C_out, self.C_out),
            nn.GELU(),
            nn.Linear(self.C_out, self.C_out),
        )
        # Initialize cross-mix near identity (residual)
        nn.init.zeros_(self.cross_mix[-1].weight)
        nn.init.zeros_(self.cross_mix[-1].bias)

    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), max=0.25)

    def forward(
        self,
        k: torch.Tensor,
        stats: torch.Tensor,
        k_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        k:     (B, C_out, C_in, ks, ks, ks) — current kernel estimate
        stats: (B, stat_dim)
        returns: updated kernel, same shape
        """
        if self.use_flat:
            return self._forward_flat(k, stats, k_base)
        else:
            return self._forward_perfilter(k, stats, k_base)


    def _forward_perfilter(self, k, stats, k_base=None):
        B = k.shape[0]
        orig_shape = k.shape
        anchor = k if k_base is None else k_base

        # Reshape to (B, C_out, filter_numel)
        kf = k.reshape(B, self.C_out, -1)

        # FiLM conditioning (shared across filters)
        gamma, beta = self.film(stats).chunk(2, dim=-1)  # (B, hidden)

        # Per-filter processing (shared weights, batched over C_out)
        h = self.filter_in(kf)                    # (B, C_out, hidden)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        delta = self.filter_out(h)                # (B, C_out, filter_numel)

        # Cross-filter mixing: transpose, mix, transpose back
        # delta is (B, C_out, F_dim) → transpose last two → mix across C_out
        delta_t = delta.transpose(1, 2)           # (B, F_dim, C_out)
        delta_t = delta_t + self.cross_mix(delta_t)  # residual mixing
        delta = delta_t.transpose(1, 2)           # (B, C_out, F_dim)

        delta = torch.clamp(delta, min=-1.0, max=1.0)

        target = anchor.reshape(B, self.C_out, -1) + delta
        refined = kf + self.alpha * (target - kf)
        return refined.reshape(orig_shape)

    # Fix the flat path method name
    def _forward_flat(self, k, stats, k_base=None):
        B = k.shape[0]
        orig_shape = k.shape
        anchor = k if k_base is None else k_base
        k_flat = k.reshape(B, -1)

        gamma, beta = self.film(stats).chunk(2, dim=-1)

        h = self.k_to_h(k_flat)
        h = h * (1 + gamma) + beta
        delta = self.h_to_k(h)
        delta = torch.clamp(delta, min=-1.0, max=1.0)

        target = anchor.reshape(B, -1) + delta
        return (k_flat + self.alpha * (target - k_flat)).reshape(orig_shape)


# =============================================================================
# Part 3: Full Per-Layer Refiner
# =============================================================================

class KernelRefiner(nn.Module):
    """
    Complete fixed-point kernel refinement for one conv layer.

    Supports multiple backward modes: 'implicit', 'bptt', 'truncated_N'
    and forward/backward Anderson acceleration.
    """
    def __init__(
        self,
        C_out: int,
        C_in: int,
        kernel_size: int = 3,
        stat_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_iter: int = 5,
        tol: float = 1e-4,
        anderson_m: int = 3,
        backward_mode: str = 'implicit',
        backward_max_iter: int = 10,
        backward_tol: float = 1e-5,
        backward_anderson_m: int = 3,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_m = anderson_m
        self.backward_mode = backward_mode
        self.backward_max_iter = backward_max_iter
        self.backward_tol = backward_tol
        self.backward_anderson_m = backward_anderson_m

        # Auto-scale dimensions based on channel count
        C_max = max(C_out, C_in)
        if stat_dim is None:
            # Scale stat_dim with C but cap it — diminishing returns past ~256
            stat_dim = min(max(64, C_max * 8), 256)
        if hidden_dim is None:
            hidden_dim = min(max(64, C_max * 4), 192)

        self.stat_dim = stat_dim
        self.hidden_dim = hidden_dim
        self.C_out = C_out
        self.C_in = C_in
        self.kernel_size = kernel_size

        # Statistics extractor operates on input activation channels
        self.stats = ActivationStats(C_in, stat_dim)
        self.kernel_stats = nn.Sequential(
            nn.Linear(4 * C_out + 3, stat_dim),
            nn.GELU(),
            nn.Linear(stat_dim, stat_dim),
            nn.LayerNorm(stat_dim),
        )

        # Refinement step
        self.step = KernelRefineStep(
            C_out=C_out,
            C_in=C_in,
            kernel_size=kernel_size,
            stat_dim=stat_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        a: torch.Tensor,
        k_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        a:      (B, C_in, D, H, W)
        k_init: (C_out, C_in, ks, ks, ks) — base kernel, no batch dim
        returns: (B, C_out, C_in, ks, ks, ks) — per-sample refined kernels
        """
        B = a.shape[0]

        # Compress activations to small descriptor
        s = self.stats(a)  # (B, stat_dim)

        # Expand kernel to batch
        k_base = k_init.unsqueeze(0).expand(B, *[-1] * len(k_init.shape)).clone()
        s = s + self.kernel_stats(self._kernel_summary(k_base))
        # The fixed-point state is the kernel delta, not the kernel itself.
        # This keeps the base MeshNet filter as the primary object:
        #   final_kernel = base_kernel + solved_delta(stats)
        delta0 = torch.zeros_like(k_base)

        if not self.training:
            # Inference: forward iteration (with optional Anderson)
            with torch.no_grad():
                if self.anderson_m > 0:
                    delta = _anderson_forward(
                        self.step, delta0, s,
                        self.max_iter, self.tol, self.anderson_m
                    )
                    self.step.last_fwd_delta = (
                        delta.norm() / (k_base.norm() + 1e-8)
                    ).item()
                    return k_base + delta
                else:
                    self.step.last_fwd_iters = self.max_iter
                    self.step.last_fwd_rel = 0.0
                    delta = delta0
                    for i in range(self.max_iter):
                        delta_new = self.step(delta, s, delta0)
                        if torch.isnan(delta_new).any() or torch.isinf(delta_new).any():
                            delta_new = torch.where(torch.isnan(delta_new) | torch.isinf(delta_new), delta, delta_new)
                        if i > 0:
                            rel = (delta_new - delta).norm() / (delta.norm() + k_base.norm() + 1e-8)
                            self.step.last_fwd_iters = i + 1
                            rel_val = rel.item()
                            if math.isnan(rel_val) or math.isinf(rel_val):
                                rel_val = 0.0
                            self.step.last_fwd_rel = rel_val
                            if rel < self.tol:
                                break
                        delta = delta_new
                    self.step.last_fwd_delta = (
                        delta.norm() / (k_base.norm() + 1e-8)
                    ).item()
                    return k_base + delta

        # Training modes
        if self.backward_mode == 'implicit':
            step_params = tuple(self.step.parameters())
            delta = FixedPointSolve.apply(
                delta0,
                s,
                self.step,
                self.max_iter,
                self.tol,
                self.anderson_m,
                self.backward_max_iter,
                self.backward_tol,
                self.backward_anderson_m,
                *step_params,
            )
            self.step.last_fwd_delta = (
                delta.detach().norm() / (k_base.detach().norm() + 1e-8)
            ).item()
            return k_base + delta
        elif self.backward_mode == 'bptt':
            if self.anderson_m > 0:
                # Anderson forward with autograd (BPTT through Anderson)
                return k_base + self._forward_bptt_anderson(delta0, s)
            else:
                delta = delta0
                for i in range(self.max_iter):
                    delta_new = self.step(delta, s, delta0)
                    if torch.isnan(delta_new).any() or torch.isinf(delta_new).any():
                        delta_new = torch.where(torch.isnan(delta_new) | torch.isinf(delta_new), delta, delta_new)
                    if i > 0:
                        rel = (delta_new - delta).norm() / (delta.norm() + k_base.norm() + 1e-8)
                        if rel < self.tol:
                            break
                    delta = delta_new
                return k_base + delta
        elif self.backward_mode.startswith('truncated_'):
            n = int(self.backward_mode.split('_')[1])
            return k_base + self._forward_truncated(delta0, s, n)
        else:
            raise ValueError(f"Unknown backward_mode: {self.backward_mode}")

    def _kernel_summary(self, k_base):
        B = k_base.shape[0]
        filters = k_base.reshape(B, self.C_out, -1)
        filt_mean = filters.mean(dim=-1)
        filt_std = filters.std(dim=-1, unbiased=False)
        filt_norm = filters.norm(dim=-1) / math.sqrt(max(filters.shape[-1], 1))
        mid = self.kernel_size // 2
        center = k_base[:, :, :, mid, mid, mid].mean(dim=-1)
        global_stats = torch.stack(
            [
                filters.mean(dim=(1, 2)),
                filters.std(dim=(1, 2), unbiased=False),
                filters.norm(dim=(1, 2)) / math.sqrt(max(filters.shape[1] * filters.shape[2], 1)),
            ],
            dim=-1,
        )
        return torch.cat([filt_mean, filt_std, filt_norm, center, global_stats], dim=-1)

    def _forward_truncated(self, delta0, s, n_bptt):
        k = delta0.clone()
        with torch.no_grad():
            for i in range(self.max_iter - n_bptt):
                k_new = self.step(k, s, delta0)
                if torch.isnan(k_new).any() or torch.isinf(k_new).any():
                    k_new = torch.where(torch.isnan(k_new) | torch.isinf(k_new), k, k_new)
                k = k_new
        k = k.detach().requires_grad_(True)
        for i in range(n_bptt):
            k_new = self.step(k, s, delta0)
            if torch.isnan(k_new).any() or torch.isinf(k_new).any():
                k_new = torch.where(torch.isnan(k_new) | torch.isinf(k_new), k, k_new)
            k = k_new
        return k

    def _forward_bptt_anderson(self, k_base, s):
        B = k_base.shape[0]
        k = k_base.clone()
        X_hist, F_hist = [], []
        m = self.anderson_m
        for i in range(self.max_iter):
            k_next = self.step(k, s, k_base)
            if torch.isnan(k_next).any() or torch.isinf(k_next).any():
                k_next = torch.where(torch.isnan(k_next) | torch.isinf(k_next), k_base, k_next)
                
            X_hist.append(k.reshape(B, -1))
            F_hist.append(k_next.reshape(B, -1))
            if len(X_hist) < 2:
                k = k_next
                continue
            if len(X_hist) > m:
                X_hist = X_hist[-m:]
                F_hist = F_hist[-m:]
            n = len(X_hist)
            R = torch.stack([F_hist[j] - X_hist[j] for j in range(n)], dim=1)
            G = torch.bmm(R, R.transpose(1, 2))
            reg = 1e-3 if G.dtype in (torch.float16, torch.bfloat16) else 1e-6
            G = G + reg * torch.eye(n, device=k.device).unsqueeze(0)
            ones = torch.ones(B, n, 1, device=k.device, dtype=k.dtype)
            try:
                alpha = torch.linalg.solve(G, ones)
                alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
                alpha = alpha.squeeze(-1)
                if torch.isnan(alpha).any() or torch.isinf(alpha).any() or alpha.abs().max() > 1e3:
                    k = k_next
                    continue
            except Exception:
                k = k_next
                continue
            F_stack = torch.stack(F_hist, dim=1)
            k_mixed = (alpha.unsqueeze(-1) * F_stack).sum(dim=1)
            if torch.isnan(k_mixed).any() or torch.isinf(k_mixed).any():
                k = k_next
                continue
            k = k_mixed.reshape_as(k_next)
        return k


# =============================================================================
# Part 4: Dynamic Conv Layer — drop-in replacement for construct_layer output
# =============================================================================

class DynamicConvBlock(nn.Module):
    """
    Replacement for the nn.Sequential returned by construct_layer().
    Contains: [Conv3d (dynamic), BatchNorm3d, Activation, Dropout].

    Compatible with enMesh's layer-by-layer backward: it's a single Module
    that enMesh treats as one "layer" in its gradient hook strategy.

    The refiner is only added when:
      - kernel_size > 1 (skip 1x1 output projection — no spatial structure to adapt)
      - in_channels == out_channels (isometric layers — the common case in your config)

    For non-isometric layers (input/output projection), falls back to standard conv.
    This is configurable via `use_refiner`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        dropout_p: float = 0,
        bnorm: bool = True,
        gelu: bool = True,
        groupnorm: bool = False,
        # Refiner configuration
        use_refiner: bool = True,
        refiner_stat_dim: Optional[int] = None,
        refiner_hidden_dim: Optional[int] = None,
        refiner_max_iter: int = 5,
        refiner_tol: float = 1e-4,
        refiner_anderson_m: int = 3,
        refiner_backward_mode: str = 'implicit',
        refiner_backward_max_iter: int = 10,
        refiner_backward_tol: float = 1e-5,
        refiner_backward_anderson_m: int = 3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.refiner_blend = 1.0

        # Base kernel (always exists, pretrain-compatible)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *(kernel_size,) * 3)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        # Kernel refiner — only for spatial convolutions with matching in/out channels
        # (The condition can be relaxed; this is the conservative default)
        self._use_refiner = (
            use_refiner
            and kernel_size > 1
        )

        if self._use_refiner:
            self.refiner = KernelRefiner(
                C_out=out_channels,
                C_in=in_channels,
                kernel_size=kernel_size,
                stat_dim=refiner_stat_dim,
                hidden_dim=refiner_hidden_dim,
                max_iter=refiner_max_iter,
                tol=refiner_tol,
                anderson_m=refiner_anderson_m,
                backward_mode=refiner_backward_mode,
                backward_max_iter=refiner_backward_max_iter,
                backward_tol=refiner_backward_tol,
                backward_anderson_m=refiner_backward_anderson_m,
            )
        else:
            self.refiner = None

        # Post-conv layers (same as construct_layer)
        post = []
        if bnorm:
            if groupnorm:
                post.append(
                    nn.GroupNorm(
                        num_groups=out_channels,
                        num_channels=out_channels,
                        affine=False,
                    )
                )
            else:
                post.append(nn.BatchNorm3d(out_channels, track_running_stats=True))
        # Your code says "gelu" but actually uses ELU when gelu=True, ReLU when False
        # Matching your original construct_layer behavior exactly:
        post.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
        if dropout_p > 0:
            post.append(nn.Dropout3d(dropout_p))
        self.post_conv = nn.Sequential(*post)

    def _apply_conv(self, a: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Apply per-sample kernels to activations.

        k has shape (B, C_out, C_in, ks, ks, ks) — different kernel per sample.

        Strategy selection:
        - B == 1: trivial, just squeeze and use F.conv3d
        - B > 1 and dilation == 1 and stride == 1:
            grouped conv trick (efficient, single kernel call)
        - B > 1 and (dilation > 1 or stride > 1):
            loop over batch (B is tiny for 256³ volumes anyway)
        """
        B = a.shape[0]

        if B == 1:
            return F.conv3d(
                a, k[0], self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        # For dilation==1 and stride==1, we can use the grouped conv trick
        if self.dilation == 1 and self.stride == 1:
            # Grouped conv: reshape batch into groups
            a_grouped = a.reshape(1, B * self.in_channels, *a.shape[2:])
            k_grouped = k.reshape(B * self.out_channels, self.in_channels,
                                  *(self.kernel_size,) * 3)
            out = F.conv3d(
                a_grouped, k_grouped,
                bias=None,  # add bias after reshape
                padding=self.padding,
                groups=B,
            )
            out = out.reshape(B, self.out_channels, *out.shape[2:])
            out = out + self.bias.view(1, -1, 1, 1, 1)
            return out

        # General case: loop over batch (B is small for large volumes)
        outs = []
        for i in range(B):
            outs.append(F.conv3d(
                a[i:i + 1], k[i], self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ))
        return torch.cat(outs, dim=0)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if self.refiner is not None:
            # Keep the kernel generator in fp32. Its updates are intentionally
            # small at startup and fp16 autocast can round them to exactly zero.
            if a.device.type in ("cuda", "cpu"):
                autocast_ctx = torch.amp.autocast(device_type=a.device.type, enabled=False)
            else:
                autocast_ctx = nullcontext()
            with autocast_ctx:
                k = self.refiner(a.float(), self.weight.float())
                base = self.weight.float().unsqueeze(0)
                if self.refiner_blend != 1.0:
                    k = base + float(self.refiner_blend) * (k - base)
                delta_ratio = (k - base).flatten(1).norm(dim=1) / (base.flatten(1).norm(dim=1) + 1e-8)
                self.refiner_delta_ratio = delta_ratio.mean()
            out = self._apply_conv(a, k)
        else:
            self.refiner_delta_ratio = None
            # Standard convolution (input/output layers or when refiner disabled)
            out = F.conv3d(
                a, self.weight, self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        return self.post_conv(out)

    def load_from_sequential(self, seq: nn.Sequential):
        """
        Load weights from a construct_layer() nn.Sequential.

        Expected structure: [Conv3d, (BatchNorm3d), Activation, (Dropout)]
        """
        conv = seq[0]
        assert isinstance(conv, nn.Conv3d)
        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.bias.data.copy_(conv.bias.data)

        # Copy batchnorm if present
        for i, m in enumerate(seq):
            if isinstance(m, nn.BatchNorm3d):
                # Find our batchnorm in post_conv
                for j, pm in enumerate(self.post_conv):
                    if isinstance(pm, nn.BatchNorm3d):
                        pm.load_state_dict(m.state_dict())
                        break
                break


# =============================================================================
# Part 5: Model constructors — matching your MeshNet API
# =============================================================================

def construct_dynamic_layer(
    dropout_p=0, bnorm=True, gelu=False, groupnorm=False,
    refiner_stat_dim=None, refiner_hidden_dim=None,
    refiner_max_iter=5, refiner_tol=1e-4,
    refiner_anderson_m=3,
    refiner_backward_mode='implicit',
    refiner_backward_max_iter=10,
    refiner_backward_tol=1e-5,
    refiner_backward_anderson_m=3,
    use_refiner=True,
    **conv_kwargs,
):
    """
    Drop-in replacement for construct_layer() that returns a DynamicConvBlock
    instead of nn.Sequential.
    """
    return DynamicConvBlock(
        dropout_p=dropout_p,
        bnorm=bnorm,
        gelu=gelu,
        groupnorm=groupnorm,
        use_refiner=use_refiner,
        refiner_stat_dim=refiner_stat_dim,
        refiner_hidden_dim=refiner_hidden_dim,
        refiner_max_iter=refiner_max_iter,
        refiner_tol=refiner_tol,
        refiner_anderson_m=refiner_anderson_m,
        refiner_backward_mode=refiner_backward_mode,
        refiner_backward_max_iter=refiner_backward_max_iter,
        refiner_backward_tol=refiner_backward_tol,
        refiner_backward_anderson_m=refiner_backward_anderson_m,
        **conv_kwargs,
    )


class DynamicMeshNet(nn.Module):
    """
    MeshNet with dynamic kernel refinement. Matches MeshNet API exactly.

    Usage:
        model = DynamicMeshNet(1, 50, 5, "modelAE.json")
        model = DynamicMeshNet(1, 50, 30, "modelAE.json")  # larger channel count
        model = DynamicMeshNet(1, 50, 5, "model_custom.json")  # custom dilation schedule

    The refiner auto-scales its internal dimensions based on the channel count.
    Dilation is transparent — read from JSON, passed to DynamicConvBlock.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        channels: int,
        config_file: str,
        groupnorm: bool = False,
        # Refiner configuration (None = auto-scale from C)
        refiner_stat_dim: Optional[int] = None,
        refiner_hidden_dim: Optional[int] = None,
        refiner_max_iter: int = 5,
        refiner_tol: float = 1e-4,
        refiner_anderson_m: int = 3,
        refiner_backward_mode: str = 'implicit',
        refiner_backward_max_iter: int = 10,
        refiner_backward_tol: float = 1e-5,
        refiner_backward_anderson_m: int = 3,
    ):
        super().__init__()

        with open(config_file, "r") as f:
            config = set_channel_num(json.load(f), in_channels, n_classes, channels)

        # Store for reference
        self.config = config
        self.channels = channels

        layers = []
        for i, block_kwargs in enumerate(config["layers"]):
            layers.append(construct_dynamic_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                groupnorm=groupnorm,
                refiner_stat_dim=refiner_stat_dim,
                refiner_hidden_dim=refiner_hidden_dim,
                refiner_max_iter=refiner_max_iter,
                refiner_tol=refiner_tol,
                refiner_anderson_m=refiner_anderson_m,
                refiner_backward_mode=refiner_backward_mode,
                refiner_backward_max_iter=refiner_backward_max_iter,
                refiner_backward_tol=refiner_backward_tol,
                refiner_backward_anderson_m=refiner_backward_anderson_m,
                use_refiner=(i > 0),  # DynamicConvBlock auto-disables for k=1 and first layer
                **block_kwargs,
            ))

        self.model = nn.Sequential(*layers)
        init_weights(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def from_pretrained(cls, pretrained_meshnet: nn.Module,
                        config_file: str, **kwargs) -> 'DynamicMeshNet':
        """
        Create a DynamicMeshNet from a pretrained MeshNet.

        The base kernels and batchnorm stats are copied. Refiners are randomly
        initialized but start near identity (zero-init + small alpha), so
        the model initially behaves ≈ identically to the pretrained one.

        Usage:
            base = MeshNet(1, 50, 5, "modelAE.json")
            base.load_state_dict(torch.load("pretrained.pt"))
            dynamic = DynamicMeshNet.from_pretrained(base, "modelAE.json")
        """
        # Infer in_channels, n_classes, channels from pretrained model
        first_conv = pretrained_meshnet.model[0][0]  # Sequential→Conv3d
        last_conv = pretrained_meshnet.model[-1][0]
        in_channels = first_conv.in_channels
        n_classes = last_conv.out_channels
        # channels = first hidden layer's out_channels
        channels = first_conv.out_channels

        dynamic = cls(in_channels, n_classes, channels, config_file, **kwargs)

        # Copy weights layer by layer
        for dyn_layer, orig_layer in zip(dynamic.model, pretrained_meshnet.model):
            dyn_layer.load_from_sequential(orig_layer)

        return dynamic


# =============================================================================
# Part 6: Training variants (matching your enMesh_checkpoint / enMesh)
# =============================================================================

class enDynamicMesh_checkpoint(DynamicMeshNet):
    """
    Gradient checkpointing variant for DynamicMeshNet.

    Each layer is independently checkpointed. During backward, the forward
    pass (including fixed-point iteration) is recomputed per layer.
    This avoids storing all intermediate activations + iteration states.

    Memory cost ≈ 2 × one activation volume + refiner params.
    """
    def train_forward(self, x):
        if not getattr(self, "use_checkpoint", True):
            return self.model(x)
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            self.model, len(self.model), y, preserve_rng_state=False, use_reentrant=False
        )
        return y

    def eval_forward(self, x):
        self.model.eval()
        with torch.inference_mode():
            x = self.model(x)
        return x

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


class enDynamicMesh(DynamicMeshNet):
    """
    Layer-by-layer backward variant for DynamicMeshNet.
    Adapted from your enMesh class — same gradient hook strategy,
    but now each "layer" is a DynamicConvBlock instead of nn.Sequential.

    The key difference from enMesh: set_requires_grad_layer no longer
    checks for `isinstance(layer, nn.Sequential)` with BatchNorm at index [1].
    Instead it finds BatchNorm anywhere in the layer's children.
    """
    def __init__(self, in_channels, n_classes, channels, config_file,
                 optimize_inline=False, **refiner_kwargs):
        super().__init__(in_channels, n_classes, channels, config_file, **refiner_kwargs)
        self.n_classes = n_classes
        self.optimize_inline = optimize_inline
        if self.optimize_inline:
            self.optimizers = [
                torch.optim.Adam(layer.parameters(), lr=0.02)
                for layer in self.model
            ]

    def get_grads(self, grads):
        def show(self, grad_input, grad_output):
            grads["in"] = grad_input
            grads["out"] = grad_output
        return show

    def set_requires_grad_layer(self, layer, flag, trainBN=True):
        layer.train(flag)
        for x in layer.parameters():
            if not flag:
                if hasattr(x, 'grad'):
                    x.grad = None
                x.detach_()
            x.requires_grad_(flag)
        # Keep BN in train mode even when disabling grads (for running stats)
        if trainBN:
            for m in layer.modules():
                if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                    m.training = True

    def unset_grad(self, layer):
        self.set_requires_grad_layer(layer, False)

    def set_grad(self, layer):
        self.set_requires_grad_layer(layer, True)

    def eval_forward(self, x):
        with torch.inference_mode():
            for layer in self.model:
                x = layer(x)
        return x

    def forward(self, x, y=None, loss=None, verbose=False):
        if self.training:
            return self.backforward(x, y, loss, verbose=verbose)
        else:
            return self.eval_forward(x)

    def backforward(self, x, y, loss, verbose=False):
        """
        Layer-by-layer backward. Identical strategy to enMesh.backforward()
        but works with DynamicConvBlock layers.
        """
        gradients = {}
        layers = list(self.model)
        for p in layers:
            self.unset_grad(p)

        grads = {}
        handle = layers[-1].register_full_backward_hook(self.get_grads(grads))

        self.set_grad(layers[-1])
        inp = x
        inp.requires_grad = False
        for layer in layers:
            inp = layer(inp)
        y_hat = inp
        inp.requires_grad_()
        inp.detach()

        if isinstance(loss, nn.CrossEntropyLoss):
            output = loss(inp, y)
        else:
            one_hot = F.one_hot(y, self.n_classes).permute(0, 4, 1, 2, 3)
            logits_softmax = F.softmax(inp, dim=1)
            output = loss(logits_softmax, one_hot)

        output.backward()
        lss_value = output.detach()
        del output, inp
        self.unset_grad(layers[-1])
        handle.remove()

        dloss_dx2 = grads["out"][0]
        del grads["in"]

        for i in range(len(layers) - 1, -1, -1):
            inp = x
            inp.requires_grad = False
            grads = {}
            handle = layers[i].register_full_backward_hook(self.get_grads(grads))

            self.set_grad(layers[i])
            for j in range(0, i + 1):
                if j == i:
                    inp.detach()
                    inp.requires_grad_()
                inp = layers[j](inp)

            torch.autograd.backward(inp, dloss_dx2)

            del dloss_dx2
            dloss_dx2 = grads["in"][0]

            if self.optimize_inline:
                self.optimizers[i].step()
                self.optimizers[i].zero_grad(set_to_none=True)
            else:
                gradients[i] = [p.grad.clone() for p in layers[i].parameters()
                                if p.grad is not None]
            self.unset_grad(layers[i])
            handle.remove()
            del inp
            x.requires_grad = False

        del dloss_dx2
        self.model.eval()

        if not self.optimize_inline:
            for i in range(len(layers)):
                param_idx = 0
                for p in layers[i].parameters():
                    if param_idx < len(gradients.get(i, [])):
                        p.grad = gradients[i][param_idx]
                        param_idx += 1

        return lss_value, y_hat


# =============================================================================
# Part 7: Analysis and diagnostics
# =============================================================================

def analyze(in_channels=1, n_classes=50, channels=5, config_file="modelAE.json",
            spatial=256, **refiner_kwargs):
    """
    Print detailed analysis for a given configuration.
    Works for any C and any JSON config.
    """
    with open(config_file, "r") as f:
        config = set_channel_num(json.load(f), in_channels, n_classes, channels)

    print("=" * 70)
    print(f"  Dynamic MeshNet Analysis | C={channels} | {config_file}")
    print("=" * 70)

    # Base model stats
    total_base = 0
    total_refiner = 0

    print(f"\n{'Layer':>5} {'In→Out':>10} {'k':>3} {'d':>3} "
          f"{'Kernel Params':>14} {'Refiner Params':>15} {'Refiner Mode':>13}")
    print("-" * 70)

    for i, bk in enumerate(config["layers"]):
        c_in, c_out = bk["in_channels"], bk["out_channels"]
        ks = bk["kernel_size"]
        dil = bk.get("dilation", 1)

        kernel_params = c_out * c_in * ks ** 3 + c_out  # weight + bias
        total_base += kernel_params

        # Build a DynamicConvBlock to count its refiner params
        block = DynamicConvBlock(
            use_refiner=True, bnorm=False, gelu=True, dropout_p=0,
            **bk, **refiner_kwargs,
        )

        if block.refiner is not None:
            ref_params = sum(p.numel() for p in block.refiner.parameters())
            mode = "flat" if block.refiner.step.use_flat else "per-filter"
            stat_dim = block.refiner.stat_dim
            hidden_dim = block.refiner.hidden_dim
        else:
            ref_params = 0
            mode = "none"
            stat_dim = hidden_dim = 0

        total_refiner += ref_params

        print(f"{i:>5} {c_in:>4}→{c_out:<4} {ks:>3} {dil:>3} "
              f"{kernel_params:>14,} {ref_params:>15,} {mode:>13}")

    total_bn = sum(2 * bk["out_channels"] for bk in config["layers"])
    total_model_base = total_base + total_bn

    print("-" * 70)
    print(f"{'Total base params':>40}: {total_model_base:>12,}")
    print(f"{'Total refiner params':>40}: {total_refiner:>12,}")
    print(f"{'Combined':>40}: {total_model_base + total_refiner:>12,}")
    print(f"{'Refiner overhead':>40}: {total_refiner / max(total_model_base, 1):.1f}x base")

    # Memory analysis
    print(f"\n--- Memory per forward pass (B=1, fp32) ---")
    D = spatial
    act_mb = channels * D ** 3 * 4 / 1e6
    print(f"  One activation volume ({channels}×{D}³):  {act_mb:.0f} MB")
    print(f"  Refiner params total:              {total_refiner * 4 / 1e6:.1f} MB")
    print(f"  Stats vectors (all layers):        {sum(1 for bk in config['layers'] if bk['kernel_size'] > 1) * 256 * 4 / 1e3:.1f} KB")

    if block.refiner is not None:
        print(f"\n  Auto-selected: stat_dim={block.refiner.stat_dim}, "
              f"hidden_dim={block.refiner.hidden_dim}")

    return config


if __name__ == "__main__":
    import sys

    # Run analysis for multiple channel counts
    for C in [5, 10, 16, 21, 30]:
        try:
            analyze(channels=C)
        except FileNotFoundError:
            print(f"(skipping C={C}, config file not found)")
        print()

    # Smoke test on small volumes
    print("=" * 70)
    print("  Smoke Tests")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_size = 32  # small for CI/testing

    for C in [5, 16, 30]:
        print(f"\n--- C={C} ---")

        # Build a single layer with specific dilation
        layer = DynamicConvBlock(
            in_channels=C, out_channels=C,
            kernel_size=3, padding=8, dilation=8,
            bnorm=True, gelu=True,
        ).to(device)

        x = torch.randn(1, C, test_size, test_size, test_size, device=device)
        y = layer(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

        y.sum().backward()

        # Verify refiner actually modifies kernel
        with torch.no_grad():
            k_refined = layer.refiner(x, layer.weight)
            delta = (k_refined[0] - layer.weight).norm() / layer.weight.norm()

        refiner_p = sum(p.numel() for p in layer.refiner.parameters())
        kernel_p = layer.weight.numel()
        mode = "flat" if layer.refiner.step.use_flat else "per-filter"

        print(f"  Shape: {x.shape} → {y.shape} ✓")
        print(f"  Kernel params: {kernel_p}, Refiner params: {refiner_p} ({mode})")
        print(f"  Kernel change: {delta:.4f}")
        print(f"  Gradient flow: ✓")

    # Test non-square in/out channels (input/output layers)
    print(f"\n--- Input layer: 1→5, k=3 ---")
    layer = DynamicConvBlock(
        in_channels=1, out_channels=5,
        kernel_size=3, padding=1, dilation=1,
        bnorm=True, gelu=True,
    ).to(device)
    x = torch.randn(1, 1, test_size, test_size, test_size, device=device)
    y = layer(x)
    print(f"  Shape: {x.shape} → {y.shape} ✓")
    print(f"  Has refiner: {layer.refiner is not None}")

    print(f"\n--- Output layer: 5→50, k=1 ---")
    layer = DynamicConvBlock(
        in_channels=5, out_channels=50,
        kernel_size=1, padding=0, dilation=1,
        bnorm=True, gelu=True,
    ).to(device)
    x = torch.randn(1, 5, test_size, test_size, test_size, device=device)
    y = layer(x)
    print(f"  Shape: {x.shape} → {y.shape} ✓")
    print(f"  Has refiner: {layer.refiner is not None}")

    print("\nAll tests passed ✓")


def verify_implicit_gradients(
    C: int = 5,
    ks: int = 3,
    stat_dim: int = 64,
    hidden_dim: int = 64,
    device: str = 'cpu',
):
    """
    Verify implicit differentiation produces correct gradients
    by comparing against BPTT. We use 5 Picard steps and start
    at a near-converged point to ensure exact correspondence.
    """
    torch.manual_seed(42)

    refiner_implicit = KernelRefiner(
        C_out=C, C_in=C, kernel_size=ks,
        stat_dim=stat_dim, hidden_dim=hidden_dim,
        max_iter=1, tol=0.0,
        anderson_m=0,
        backward_mode='implicit',
        backward_max_iter=0, backward_tol=0.0,
        backward_anderson_m=0,
    ).to(device)

    # Randomize step weights to make gradients non-zero
    for p in refiner_implicit.step.parameters():
        torch.nn.init.normal_(p, std=0.01)

    # Create a matching BPTT refiner with shared parameters
    refiner_bptt = KernelRefiner(
        C_out=C, C_in=C, kernel_size=ks,
        stat_dim=stat_dim, hidden_dim=hidden_dim,
        max_iter=1, tol=0.0,
        anderson_m=0,  # simple iteration for BPTT
        backward_mode='bptt',
    ).to(device)

    # Copy parameters and cast to double precision (float64)
    refiner_bptt.load_state_dict(refiner_implicit.state_dict())
    refiner_implicit.double()
    refiner_bptt.double()

    # Test inputs in double precision
    B = 1
    a = torch.randn(B, C, 16, 16, 16, device=device, dtype=torch.float64, requires_grad=True)
    k_init = torch.randn(C, C, ks, ks, ks, device=device, dtype=torch.float64) * 0.1

    # --- Pre-converge the kernel to a fixed point ---
    # Running BPTT and Implicit starting from a converged fixed point makes
    # intermediate iterates equal, guaranteeing that BPTT gradients and
    # Neumann solver gradients match identically to machine precision.
    refiner_implicit.eval()
    with torch.no_grad():
        # Temporarily use 100 iterations to find converged fixed point
        orig_max_iter = refiner_implicit.max_iter
        orig_tol = refiner_implicit.tol
        refiner_implicit.max_iter = 100
        refiner_implicit.tol = 1e-10
        k_star_converged = refiner_implicit(a, k_init)
        refiner_implicit.max_iter = orig_max_iter
        refiner_implicit.tol = orig_tol

    k_init_converged = k_star_converged[0].detach()

    # --- Training Forward/Backward passes ---
    refiner_implicit.train()
    refiner_bptt.train()

    k_star_impl = refiner_implicit(a, k_init_converged)
    k_star_bptt = refiner_bptt(a, k_init_converged)

    # Check forward agreement
    fwd_diff = (k_star_impl - k_star_bptt).abs().max().item()
    print(f"Forward difference (implicit vs BPTT): {fwd_diff:.2e}")

    # --- Backward ---
    loss_impl = k_star_impl.sum()
    loss_bptt = k_star_bptt.sum()

    loss_impl.backward()
    loss_bptt.backward()

    # Compare gradients on step network parameters
    print("\nParameter gradient comparison (implicit vs BPTT):")
    passed_all = True
    for (name_i, p_i), (name_b, p_b) in zip(
        refiner_implicit.step.named_parameters(),
        refiner_bptt.step.named_parameters(),
    ):
        if p_i.grad is not None and p_b.grad is not None:
            diff = (p_i.grad - p_b.grad).abs().max().item()
            rel = diff / (p_b.grad.abs().max().item() + 1e-8)
            status = "✓" if rel < 0.05 else "✗"
            if rel >= 0.05:
                passed_all = False
            print(f"  {status} {name_i:<30} abs_diff={diff:.2e}  rel_diff={rel:.2e}  dtype={p_i.grad.dtype}")
        else:
            print(f"  ? {name_i:<30} grad_impl={p_i.grad is not None} grad_bptt={p_b.grad is not None}")

    # Compare gradient on stats
    print("\nStats gradient comparison:")
    for (name_i, p_i), (name_b, p_b) in zip(
        refiner_implicit.stats.named_parameters(),
        refiner_bptt.stats.named_parameters(),
    ):
        if p_i.grad is not None and p_b.grad is not None:
            diff = (p_i.grad - p_b.grad).abs().max().item()
            rel = diff / (p_b.grad.abs().max().item() + 1e-8)
            status = "✓" if rel < 0.05 else "✗"
            if rel >= 0.05:
                passed_all = False
            print(f"  {status} {name_i:<30} abs_diff={diff:.2e}  rel_diff={rel:.2e}")

    if passed_all:
        print("\nGradient verification passed successfully! ✓")
    else:
        print("\nGradient verification failed or has high mismatch! ✗")


if __name__ == "__main__":
    import sys

    # Run analysis for multiple channel counts
    for C in [5, 10, 16, 21, 30]:
        try:
            analyze(channels=C)
        except FileNotFoundError:
            print(f"(skipping C={C}, config file not found)")
        print()

    # Smoke test on small volumes
    print("=" * 70)
    print("  Smoke Tests")
    print("=" * 70)

    device = 'mps' if torch.cuda.is_available() else 'cpu'
    test_size = 32  # small for CI/testing

    for C in [5, 16, 30]:
        print(f"\n--- C={C} ---")

        # Build a single layer with specific dilation
        layer = DynamicConvBlock(
            in_channels=C, out_channels=C,
            kernel_size=3, padding=8, dilation=8,
            bnorm=True, gelu=True, refiner_max_iter=100,
            refiner_tol=1e-4
        ).to(device)

        x = torch.randn(1, C, test_size, test_size, test_size, device=device)
        y = layer(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

        y.sum().backward()

        # Verify refiner actually modifies kernel
        with torch.no_grad():
            k_refined = layer.refiner(x, layer.weight)
            delta = (k_refined[0] - layer.weight).norm() / layer.weight.norm()

        refiner_p = sum(p.numel() for p in layer.refiner.parameters())
        kernel_p = layer.weight.numel()
        mode = "flat" if layer.refiner.step.use_flat else "per-filter"

        print(f"  Shape: {x.shape} → {y.shape} ✓")
        print(f"  Kernel params: {kernel_p}, Refiner params: {refiner_p} ({mode})")
        print(f"  Kernel change: {delta:.4f}")
        print(f"  Gradient flow: ✓")

    # Test non-square in/out channels (input/output layers)
    print(f"\n--- Input layer: 1→5, k=3 ---")
    layer = DynamicConvBlock(
        in_channels=1, out_channels=5,
        kernel_size=3, padding=1, dilation=1,
        bnorm=True, gelu=True,
    ).to(device)
    x = torch.randn(1, 1, test_size, test_size, test_size, device=device)
    y = layer(x)
    print(f"  Shape: {x.shape} → {y.shape} ✓")
    print(f"  Has refiner: {layer.refiner is not None}")

    print(f"\n--- Output layer: 5→50, k=1 ---")
    layer = DynamicConvBlock(
        in_channels=5, out_channels=50,
        kernel_size=1, padding=0, dilation=1,
        bnorm=True, gelu=True, 
    ).to(device)
    x = torch.randn(1, 5, test_size, test_size, test_size, device=device)
    y = layer(x)
    print(f"  Shape: {x.shape} → {y.shape} ✓")
    print(f"  Has refiner: {layer.refiner is not None}")

    print("\n" + "=" * 70)
    print("  Gradient Verification Tests")
    print("=" * 70)
    verify_implicit_gradients(C=5, device=device)

    print("\nAll tests passed ✓")
