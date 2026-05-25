"""Basis-kernel adapter for MeshNet.

Each adapted layer keeps a strong base kernel and learns a small bank of basis
kernels. Input statistics choose bounded coefficients:

    k_eff = k_base + sum_i coeff_i(stats, k_base) * basis_i

This adds parameter capacity without increasing activation width or requiring a
fixed-point solve.
"""

import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from refiner import ActivationStats, DynamicConvBlock, DynamicMeshNet, enDynamicMesh, enDynamicMesh_checkpoint


class BasisKernelAdapter(nn.Module):
    def __init__(
        self,
        C_out: int,
        C_in: int,
        kernel_size: int = 3,
        stat_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        basis_rank: int = 16,
        max_delta_ratio: float = 0.35,
        **_,
    ):
        super().__init__()
        self.C_out = C_out
        self.C_in = C_in
        self.kernel_size = kernel_size
        self.basis_rank = basis_rank
        self.max_delta_ratio = max_delta_ratio
        C_max = max(C_out, C_in)
        if stat_dim is None:
            stat_dim = min(max(64, C_max * 8), 256)
        if hidden_dim is None:
            hidden_dim = min(max(64, C_max * 4), 192)
        self.stat_dim = stat_dim
        self.hidden_dim = hidden_dim

        self.stats = ActivationStats(C_in, stat_dim)
        self.kernel_stats = nn.Sequential(
            nn.Linear(4 * C_out + 3, stat_dim),
            nn.GELU(),
            nn.Linear(stat_dim, stat_dim),
            nn.LayerNorm(stat_dim),
        )
        self.coeff_net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, basis_rank),
        )
        nn.init.normal_(self.coeff_net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.coeff_net[-1].bias)

        self.basis = nn.Parameter(torch.empty(basis_rank, C_out, C_in, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.basis, mode="fan_out", nonlinearity="relu")
        self.basis.data.mul_(0.05)

        object.__setattr__(self, "step", SimpleNamespace())

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

    def forward(self, a: torch.Tensor, k_init: torch.Tensor) -> torch.Tensor:
        B = a.shape[0]
        k_base = k_init.unsqueeze(0).expand(B, *[-1] * len(k_init.shape)).clone()
        s = self.stats(a) + self.kernel_stats(self._kernel_summary(k_base))
        coeff = torch.tanh(self.coeff_net(s))
        delta = torch.einsum("br,r...->b...", coeff, self.basis)

        base_norm = k_base.flatten(1).norm(dim=1).view(B, 1, 1, 1, 1, 1)
        delta_norm = delta.flatten(1).norm(dim=1).view(B, 1, 1, 1, 1, 1)
        max_norm = self.max_delta_ratio * base_norm
        delta = delta * torch.clamp(max_norm / (delta_norm + 1e-8), max=1.0)

        self.step.last_fwd_iters = 1
        self.step.last_fwd_rel = 0.0
        self.step.last_fwd_delta = (delta.detach().norm() / (k_base.detach().norm() + 1e-8)).item()
        self.step.last_fwd_alpha = 1.0
        self.step.last_coeff_abs_mean = coeff.detach().abs().mean().item()
        self.step.last_coeff_abs_max = coeff.detach().abs().max().item()
        return k_base + delta


class BasisDynamicConvBlock(DynamicConvBlock):
    def __init__(self, *args, refiner_basis_rank=16, refiner_max_delta_ratio=0.35, **kwargs):
        super().__init__(*args, **kwargs)
        if self.refiner is not None:
            self.refiner = BasisKernelAdapter(
                C_out=self.out_channels,
                C_in=self.in_channels,
                kernel_size=self.kernel_size,
                stat_dim=getattr(self.refiner, "stat_dim", None),
                hidden_dim=getattr(self.refiner, "hidden_dim", None),
                basis_rank=refiner_basis_rank,
                max_delta_ratio=refiner_max_delta_ratio,
            )


class BasisDynamicMeshNet(DynamicMeshNet):
    def __init__(self, *args, refiner_basis_rank=16, refiner_max_delta_ratio=0.35, **kwargs):
        super().__init__(*args, **kwargs)
        for i, layer in enumerate(self.model):
            if isinstance(layer, DynamicConvBlock):
                replacement = BasisDynamicConvBlock(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                    stride=layer.stride,
                    dilation=layer.dilation,
                    bnorm=any(isinstance(m, nn.BatchNorm3d) for m in layer.post_conv),
                    gelu=any(isinstance(m, nn.ELU) for m in layer.post_conv),
                    groupnorm=any(isinstance(m, nn.GroupNorm) for m in layer.post_conv),
                    use_refiner=layer.refiner is not None,
                    refiner_basis_rank=refiner_basis_rank,
                    refiner_max_delta_ratio=refiner_max_delta_ratio,
                )
                replacement.weight.data.copy_(layer.weight.data)
                replacement.bias.data.copy_(layer.bias.data)
                replacement.post_conv.load_state_dict(layer.post_conv.state_dict())
                self.model[i] = replacement


class enBasisDynamicMesh_checkpoint(BasisDynamicMeshNet, enDynamicMesh_checkpoint):
    pass


class enBasisDynamicMesh(BasisDynamicMeshNet, enDynamicMesh):
    pass
