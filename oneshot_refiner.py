"""One-shot residual kernel adapter for MeshNet.

This keeps the existing fixed-point refiner untouched. The model here uses the
same dynamic-conv wrapper API, but predicts a bounded residual delta in one pass:

    k_effective = k_base + delta(stats, kernel_summary)
"""

import math
from typing import Optional
from types import SimpleNamespace

import torch
import torch.nn as nn

from refiner import (
    ActivationStats,
    DynamicConvBlock,
    DynamicMeshNet,
    enDynamicMesh,
    enDynamicMesh_checkpoint,
)


class OneShotKernelRefiner(nn.Module):
    def __init__(
        self,
        C_out: int,
        C_in: int,
        kernel_size: int = 3,
        stat_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_delta_ratio: float = 0.35,
        **_,
    ):
        super().__init__()
        self.C_out = C_out
        self.C_in = C_in
        self.kernel_size = kernel_size
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
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, C_out * C_in * kernel_size ** 3),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

        # Compatibility with existing diagnostics that look for module.step.*.
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
        raw_delta = self.net(s).reshape_as(k_base)

        base_norm = k_base.flatten(1).norm(dim=1).view(B, 1, 1, 1, 1, 1)
        raw_norm = raw_delta.flatten(1).norm(dim=1).view(B, 1, 1, 1, 1, 1)
        max_norm = self.max_delta_ratio * base_norm
        delta = raw_delta * torch.clamp(max_norm / (raw_norm + 1e-8), max=1.0)

        self.step.last_fwd_iters = 1
        self.step.last_fwd_rel = 0.0
        self.step.last_fwd_delta = (delta.detach().norm() / (k_base.detach().norm() + 1e-8)).item()
        self.step.last_fwd_alpha = 1.0
        return k_base + delta


class OneShotDynamicConvBlock(DynamicConvBlock):
    def __init__(self, *args, refiner_max_delta_ratio: float = 0.35, **kwargs):
        super().__init__(*args, **kwargs)
        if self.refiner is not None:
            self.refiner = OneShotKernelRefiner(
                C_out=self.out_channels,
                C_in=self.in_channels,
                kernel_size=self.kernel_size,
                stat_dim=getattr(self.refiner, "stat_dim", None),
                hidden_dim=getattr(self.refiner, "hidden_dim", None),
                max_delta_ratio=refiner_max_delta_ratio,
            )


def construct_oneshot_dynamic_layer(
    dropout_p=0, bnorm=True, gelu=False, groupnorm=False,
    refiner_stat_dim=None, refiner_hidden_dim=None,
    refiner_max_delta_ratio=0.35,
    use_refiner=True,
    **conv_kwargs,
):
    return OneShotDynamicConvBlock(
        dropout_p=dropout_p,
        bnorm=bnorm,
        gelu=gelu,
        groupnorm=groupnorm,
        use_refiner=use_refiner,
        refiner_stat_dim=refiner_stat_dim,
        refiner_hidden_dim=refiner_hidden_dim,
        refiner_max_delta_ratio=refiner_max_delta_ratio,
        **conv_kwargs,
    )


class OneShotDynamicMeshNet(DynamicMeshNet):
    def __init__(self, *args, refiner_max_delta_ratio=0.35, **kwargs):
        super().__init__(*args, **kwargs)
        for i, layer in enumerate(self.model):
            if isinstance(layer, DynamicConvBlock):
                replacement = OneShotDynamicConvBlock(
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
                    refiner_max_delta_ratio=refiner_max_delta_ratio,
                )
                replacement.weight.data.copy_(layer.weight.data)
                replacement.bias.data.copy_(layer.bias.data)
                replacement.post_conv.load_state_dict(layer.post_conv.state_dict())
                self.model[i] = replacement


class enOneShotDynamicMesh_checkpoint(OneShotDynamicMeshNet, enDynamicMesh_checkpoint):
    pass


class enOneShotDynamicMesh(OneShotDynamicMeshNet, enDynamicMesh):
    pass
