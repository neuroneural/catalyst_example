"""Two-head student for teacher-matched distillation.

The 6-channel student keeps its deployment head (n_classes, e.g. 3) and gains a
parallel AUXILIARY head with the teacher's class count (e.g. 18). Both heads
read the SAME trunk features. The aux head is trained only by KL to the teacher's
full 18-class distribution (see distill.Distiller.kd18_loss); it forces the
shared trunk toward the teacher's more discriminative representation -- which is
where the teacher's robustness (e.g. dura rejection on real scans) lives, and
which marginalizing 18->3 before the KL destroys.

Export / inference: the aux head is dropped. `export_base()` returns the
original single-head model with the trained trunk + deployment head, so the
WebGPU model and its peak memory are byte-identical to the current model.

Resuming: load your last 3-class checkpoint straight into `.base` (keys match);
the aux head initializes fresh.

NOTE on checkpointing: this wrapper runs the trunk as a plain module sequence,
matching enMesh_checkpoint_gn with use_checkpoint=False (the turbo config). If
you re-enable gradient checkpointing, the segment logic would need to be
mirrored here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoHeadMeshNet(nn.Module):
    def __init__(self, base: nn.Module, aux_classes: int = 18):
        super().__init__()
        self.base = base                      # enMesh_checkpoint_gn (its .model ends in Conv3d->n_classes)
        final = base.model[-1]
        assert isinstance(final, nn.Conv3d), (
            f"expected final layer to be Conv3d, got {type(final)}")
        ch = final.in_channels
        self.n_classes = final.out_channels
        self.aux_classes = aux_classes
        self.head_aux = nn.Conv3d(ch, aux_classes, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.head_aux.weight, nonlinearity="linear")
        nn.init.zeros_(self.head_aux.bias)
        # keep the flag the training loop checks; trunk runs plain (no checkpoint)
        self.use_checkpoint = bool(getattr(base, "use_checkpoint", False))

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        # everything except the final (deployment) conv
        for m in list(self.base.model)[:-1]:
            x = m(x)
        return x

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """Default path returns ONLY the deployment logits, so every existing
        caller (loss, dice metric, inference, export) is unchanged. Pass
        return_aux=True during training to also get the 18-class aux logits."""
        feat = self._trunk(x)
        out = self.base.model[-1](feat)
        if return_aux:
            return out, self.head_aux(feat)
        return out

    @torch.no_grad()
    def export_base(self) -> nn.Module:
        """The original single-head model (trunk + deployment head), aux dropped.
        Serialize base.state_dict() for the WebGPU export path."""
        return self.base
