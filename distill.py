"""Knowledge distillation from an 18-class / 16-channel teacher into the 3-class
6-channel student (gn_hdc_deep *siam3).

Why this exists
---------------
The student trains on a HARD SynthSeg distribution where per-voxel labels near
tissue boundaries are ambiguous. A tiny 6ch model cannot discover clean
boundaries from hard one-hot labels alone. A stronger teacher that has seen
equally-hard data emits *soft* targets that are far easier to fit exactly where
the student currently fails.

Two correctness points that are easy to get wrong
-------------------------------------------------
1. 18 -> 3 must be done in PROBABILITY space, not logit space. Logits are not
   additive across classes; adding them is meaningless. The proper reduction is
   a marginalization:  P_student(c) = sum_{k : map(k)=c} P_teacher(k).
   We softmax the teacher over its 18 classes, then scatter-sum the
   probabilities into the 3 student bins.
2. Temperature is applied to the teacher's *18-class* logits BEFORE softmax and
   BEFORE marginalizing. This carries the teacher's intra-superclass
   uncertainty (e.g. which WM sub-structure it is torn between) into the soft
   target, which is extra signal for free.

Peak memory
-----------
The teacher runs only during TRAINING, under no_grad, in eval mode. Inference
and the exported WebGPU model are completely untouched.
"""

from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

from meshnet_gn import enMesh_checkpoint as enMesh_checkpoint_gn


# ---------------------------------------------------------------------------
# 18 -> 3 label map, derived by composing _lut104_to_18 and _lut104_to_3 through
# their shared 104-label space (see conf comment / user's LUTs).
#
#   background (0): 0 CSF, 3 lat-vent, 4 inf-lat-vent, 11 3rd-vent, 12 4th-vent
#   white      (1): 1 cerebral-WM+CC, 5 cerebellum-WM, 13 brain-stem
#   gray       (2): 2 cortex, 6 cerebellum-cortex, 7 thalamus, 8 caudate,
#                   9 putamen, 10 pallidum, 14 hippocampus, 15 amygdala,
#                   16 accumbens, 17 ventralDC
#
# This exactly matches the student's _lut104_to_3 judgment calls
# (ventralDC -> gray, brain-stem -> white), so teacher and student agree on the
# tissue boundaries.
# ---------------------------------------------------------------------------
LUT_18_TO_3 = torch.tensor(
    #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
    [ 0, 1, 2, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 1, 2, 2, 2, 2],
    dtype=torch.long,
)


def build_marginalizer(lut: torch.Tensor, n_student_classes: int) -> torch.Tensor:
    """Return a [n_teacher_classes, n_student_classes] 0/1 matrix M such that
    p_student = einsum('bk...,kc->bc...', p_teacher, M) marginalizes teacher
    probabilities into student super-classes."""
    n_teacher = lut.numel()
    M = torch.zeros(n_teacher, n_student_classes, dtype=torch.float32)
    M[torch.arange(n_teacher), lut] = 1.0
    # sanity: each teacher class maps to exactly one student class
    assert torch.all(M.sum(dim=1) == 1), "LUT is not a clean partition"
    return M


def marginalize_18_to_3(
    teacher_logits: torch.Tensor,
    marg_matrix: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """teacher_logits: [B, 18, *spatial] -> soft target probs [B, 3, *spatial].

    Softmax over the 18 classes at the given temperature, then contract the
    class axis with the [18, 3] marginalization matrix.
    """
    p18 = F.softmax(teacher_logits / temperature, dim=1)  # [B,18,...]
    # move class axis last, matmul with [18,3], move back
    p3 = torch.einsum("bk...,kc->bc...", p18, marg_matrix.to(p18.dtype))
    return p3.clamp_min(1e-8)


class Distiller:
    """Holds the frozen teacher and computes the KD loss term.

    Usage inside the training step (student already forwarded):
        loss_kd = distiller.kd_loss(student_logits, sample)
        loss = (1 - alpha) * loss_sup + alpha * loss_kd
    """

    def __init__(
        self,
        checkpoint_path: str,
        device,
        *,
        teacher_channels: int = 16,
        teacher_classes: int = 18,
        student_classes: int = 3,
        config_file: str = "./modelAE_hdc_deep.json",
        affine: bool = True,
        temperature: float = 2.0,
        lut: torch.Tensor = LUT_18_TO_3,
        amp_dtype: torch.dtype = torch.bfloat16,
        channels_last: bool = True,
    ):
        self.temperature = float(temperature)
        self.amp_dtype = amp_dtype
        self.device = device
        self.marg = build_marginalizer(lut, student_classes).to(device)

        model = enMesh_checkpoint_gn(
            in_channels=1,
            n_classes=teacher_classes,
            channels=teacher_channels,
            config_file=config_file,
            affine=affine,
        )
        model.use_checkpoint = False  # teacher is no_grad; checkpointing pointless
        state = torch.load(checkpoint_path, map_location="cpu")
        # accept raw state_dict or catalyst-style {"model_state_dict": ...}
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[distill] teacher load: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys (strict=False)",
                file=sys.stderr, flush=True,
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last_3d)
        self.teacher = model
        print(
            f"[distill] teacher ready: {teacher_channels}ch/{teacher_classes}cls "
            f"T={self.temperature} from {checkpoint_path}",
            file=sys.stderr, flush=True,
        )

    @torch.no_grad()
    def _teacher_soft(self, sample: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
            tlogits = self.teacher(sample)
        # marginalize in fp32 for a stable, well-normalized target
        return marginalize_18_to_3(
            tlogits.float(), self.marg, self.temperature
        )

    @torch.no_grad()
    def _teacher_logits(self, sample: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
            return self.teacher(sample).float()

    def kd18_loss(self, student_aux_logits: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Full-granularity KD for the two-head student: KL between the teacher's
        18-class distribution and the student's 18-class AUX head, at the same
        temperature. No marginalization -- this preserves the intra-tissue and
        boundary structure that carries the teacher's robustness, and gives the
        shared trunk a strong gradient (unlike the near-degenerate 3-class KL).

        student_aux_logits: [B, 18, *spatial] from TwoHeadMeshNet's aux head.
        """
        p_teacher = F.softmax(self._teacher_logits(sample) / self.temperature, dim=1).clamp_min(1e-8)
        logp_student = F.log_softmax(student_aux_logits.float() / self.temperature, dim=1)
        kl = F.kl_div(logp_student, p_teacher, reduction="none").sum(dim=1)
        return kl.mean() * (self.temperature ** 2)

    def marginal_consistency_loss(self, deploy_logits: torch.Tensor,
                                  aux_logits: torch.Tensor) -> torch.Tensor:
        """Self-distillation: pull the cheap 3-class deploy head toward the
        student's OWN marginalized 18-class aux head.

        KL( marginalize(aux_18)  ||  deploy_3 ), with the aux/marginal target
        DETACHED so gradient flows only into the deploy head + trunk (the aux
        head is not dragged down toward the coarser 3-class view). Because the
        aux head tracks the teacher, this hands the deploy head the teacher's
        sharp GM/WM/background boundaries -- brainstem, dura -- at 3-channel
        deployment cost. Marginalized at T=1 (a sharp, valid target).

        deploy_logits: [B, 3, *spatial]   aux_logits: [B, 18, *spatial]
        """
        with torch.no_grad():
            p3_target = marginalize_18_to_3(aux_logits.float(), self.marg, temperature=1.0)
        logp_deploy = F.log_softmax(deploy_logits.float(), dim=1)
        kl = F.kl_div(logp_deploy, p3_target, reduction="none").sum(dim=1)
        return kl.mean()

    def kd_loss(self, student_logits: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """KL(teacher_p3 || student) * T^2, averaged over voxels & batch.

        student_logits: [B, 3, *spatial] (raw logits from the student).
        """
        p3_teacher = self._teacher_soft(sample)              # [B,3,...] fp32
        logp_student = F.log_softmax(student_logits.float() / self.temperature, dim=1)
        # F.kl_div(input=logQ, target=P) = sum P*(logP - logQ); reduction over all
        # elements. Divide by (B * spatial) to get a per-voxel mean, then T^2.
        kl = F.kl_div(logp_student, p3_teacher, reduction="none").sum(dim=1)  # per-voxel
        return kl.mean() * (self.temperature ** 2)
