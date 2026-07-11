"""JDX / JDRX channel-decorrelation pressure for tiny MeshNet trunks.

Motivation
----------
A 5/6-channel MeshNet is capacity-starved *only if* it uses its channels well.
The failure mode we suspect is not too-few parameters but wasted ones: several
channels drifting toward encoding the *same* thing, so the effective width is
smaller than the nominal width. That directly costs robustness at fixed peak
memory (peak memory is set by ``channels x volume`` at inference, one reused
buffer per layer -- it does not change if the channels are redundant or not).

This module adds a training-only auxiliary loss that pushes the per-layer
feature channels to encode *different* sources, drawing on the JDX blind
source-separation framework (JDX.pdf). JDX's central empirical claim is that
even when latent sources look globally dependent, they are locally (within a
small spatial patch / subcube) uncorrelated, and that jointly diagonalizing the
covariances of *structured* sub-batches recovers a de-mixing that separates
them. Barlow-Twins-style objectives use the same second-order, off-diagonal
covariance penalty to remove redundancy between representation dimensions.

Rather than *solving* a joint-diagonalization (estimating a de-mixing matrix W),
we treat the conv kernels themselves as the learnable mixing/de-mixing and add
the joint-diagonalization *objective* as a differentiable pressure on each
layer's activations. Concretely, for a layer activation A of shape
[N, C, D, H, W] we sample L sub-batches, compute the C x C channel covariance
of each, and minimize the JDX energy ratio (JDX.pdf eq. 3):

        rho = sum_batches sum_{j!=k} Cov_jk^2  /  sum_batches sum_j Cov_jj^2

i.e. off-diagonal (cross-channel) covariance energy relative to diagonal
(per-channel variance) energy. Driving rho down makes the channels mutually
uncorrelated *within* each sub-batch, so the kernels are encouraged to emit
distinct sources instead of duplicating one.

Two sampling regimes (JDX.pdf sec. III-B / III-C), selected by ``randomize``:

  * JDX  (randomize=False, default): each sub-batch is a contiguous spatial
    SUBCUBE (the 3-D analogue of the paper's 2-D square patches, Fig. 1).
    Exploits the local-uncorrelatedness of spatially structured data -- the
    most promising regime for volumetric brains.

  * JDRX (randomize=True): each sub-batch is a set of voxels sampled i.i.d.
    from the whole volume. Drops the spatial-structure assumption and instead
    leans on the global distributional statistics (this is the regime the paper
    links to minimizing cross-cumulants / maximizing marginal kurtosis, i.e. a
    more classical ICA-like pressure).

Cost / memory
-------------
The activations are already materialized in the training graph (the turbo
config runs with ``use_checkpoint: false``), so capturing references adds no
peak memory. The covariance work is C x C per sub-batch (C <= ~6), i.e.
negligible. Inference and the WebGPU export are entirely untouched: this loss
exists only while ``model.training`` is True and is never part of the exported
graph.
"""

from __future__ import annotations

import torch


def _sub_batch_cov(y: torch.Tensor):
    """C x C channel covariance of a feature matrix ``y`` [C, n] (unbiased,
    centered). Returns None for degenerate n < 2."""
    C, n = y.shape
    if n < 2:
        return None
    y = y - y.mean(dim=1, keepdim=True)
    return (y @ y.transpose(0, 1)) / (n - 1)              # [C, C]


def _cov_terms_ratio(cov: torch.Tensor):
    """(off_diagonal_energy, diagonal_energy) of a covariance matrix -- the raw
    JDX energy-ratio numerator/denominator (eq. 3)."""
    diag = torch.diagonal(cov)
    diag_energy = (diag * diag).sum()
    off_energy = (cov * cov).sum() - diag_energy
    return off_energy, diag_energy


def _cov_mean_offdiag_corr(cov: torch.Tensor, eps: float = 1e-6):
    """Mean squared OFF-DIAGONAL CORRELATION of a covariance matrix.

    Normalizes to a correlation matrix first (R_jk = Cov_jk / (std_j std_k)),
    so the diagonal is fixed at ~1 by construction. This removes the
    variance-inflation loophole of the raw ratio: the model can no longer shrink
    the penalty by scaling a channel's variance (bigger diagonal denominator) --
    only by genuinely reducing cross-channel *correlation*. This is the
    Barlow-Twins redundancy term (off-diagonal of the normalized cross-corr)."""
    C = cov.shape[0]
    d = torch.diagonal(cov)
    std = torch.sqrt(torch.clamp(d, min=eps))
    R = cov / (std[:, None] * std[None, :] + eps)         # [C, C] correlation
    r2 = R * R
    off = r2.sum() - torch.diagonal(r2).sum()             # zero the diagonal
    n_off = max(1, C * (C - 1))
    return off / n_off                                    # mean over off-diag pairs


def _clamp_region(region, D, H, W):
    """Clamp a (z0,z1,y0,y1,x0,x1) box to an activation's spatial size, or
    return the full box if region is None/degenerate."""
    if region is None:
        return 0, D, 0, H, 0, W
    z0, z1, y0, y1, x0, x1 = region
    z0, z1 = max(0, min(int(z0), D - 1)), max(1, min(int(z1), D))
    y0, y1 = max(0, min(int(y0), H - 1)), max(1, min(int(y1), H))
    x0, x1 = max(0, min(int(x0), W - 1)), max(1, min(int(x1), W))
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return 0, D, 0, H, 0, W
    return z0, z1, y0, y1, x0, x1


def _iter_sub_batches(act: torch.Tensor, num_batches: int, subcube: int,
                      randomize: bool, region=None):
    """Yield [C, n] feature matrices for L structured (JDX) or randomized
    (JDRX) sub-batches drawn from a [N, C, D, H, W] activation.

    ``region`` = (z0,z1,y0,y1,x0,x1) restricts sampling to a spatial box (e.g.
    the brain bounding box), so subcubes land on tissue instead of background.
    None => whole volume."""
    N, C, D, H, W = act.shape
    L = max(1, int(num_batches))
    z0, z1, y0, y1, x0, x1 = _clamp_region(region, D, H, W)
    if randomize:
        box = act[:, :, z0:z1, y0:y1, x0:x1]
        flat = box.permute(1, 0, 2, 3, 4).reshape(C, -1)  # [C, N*box]
        M = flat.shape[1]
        n = min(int(subcube) ** 3, M)
        for _ in range(L):
            idx = torch.randint(0, M, (n,), device=act.device)
            yield flat[:, idx]
    else:
        sd = min(int(subcube), z1 - z0)
        sh = min(int(subcube), y1 - y0)
        sw = min(int(subcube), x1 - x0)
        for _ in range(L):
            z = z0 + int(torch.randint(0, (z1 - z0) - sd + 1, (1,)).item())
            y = y0 + int(torch.randint(0, (y1 - y0) - sh + 1, (1,)).item())
            x = x0 + int(torch.randint(0, (x1 - x0) - sw + 1, (1,)).item())
            sub = act[:, :, z:z + sd, y:y + sh, x:x + sw]
            yield sub.permute(1, 0, 2, 3, 4).reshape(C, -1)


def jdx_layer_penalty(
    act: torch.Tensor,
    num_batches: int = 4,
    subcube: int = 9,
    randomize: bool = False,
    mode: str = "ratio",
    region=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-layer channel-decorrelation penalty for one activation.

    Args:
        act: activation tensor [N, C, D, H, W] (any float dtype; cast to fp32).
        num_batches: number of structured/random sub-batches (L in the paper).
        subcube: sub-batch size. Structured: edge length of the D/H/W subcube
            (n = N * subcube^3 samples). Randomized: n = subcube^3 voxels drawn
            i.i.d. across all volumes and positions.
        randomize: False -> JDX (contiguous subcubes); True -> JDRX (global
            i.i.d. voxel sampling).
        mode: "corr" (recommended) -> mean squared off-diagonal CORRELATION
            (variance-normalized; cannot be gamed by scaling channel variance).
            "ratio" -> the raw JDX off/diag covariance energy ratio (eq. 3);
            kept for backward-compat but note it is scale-exploitable.

    Returns:
        Scalar tensor; lower == more decorrelated channels. 0 for C < 2.
    """
    if act.dim() != 5:
        return act.new_zeros(())
    act = act.float()
    C = act.shape[1]
    if C < 2:
        return act.new_zeros(())

    if mode == "corr":
        vals = []
        for y in _iter_sub_batches(act, num_batches, subcube, randomize, region):
            cov = _sub_batch_cov(y)
            if cov is not None:
                vals.append(_cov_mean_offdiag_corr(cov, eps))
        if not vals:
            return act.new_zeros(())
        return torch.stack(vals).mean()

    # mode == "ratio" (legacy)
    off_sum = act.new_zeros(())
    diag_sum = act.new_zeros(())
    for y in _iter_sub_batches(act, num_batches, subcube, randomize, region):
        cov = _sub_batch_cov(y)
        if cov is not None:
            o, d = _cov_terms_ratio(cov)
            off_sum = off_sum + o
            diag_sum = diag_sum + d
    return off_sum / (diag_sum + eps)


def jdx_penalty(
    activations,
    num_batches: int = 4,
    subcube=9,
    randomize: bool = False,
    mode: str = "ratio",
    region=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean per-layer JDX penalty over a list of activation tensors.

    ``subcube`` may be a single int (same size every layer) or a per-layer list
    aligned with ``activations`` (e.g. larger subcubes for the middle layers).
    ``region`` restricts sampling to a spatial box (brain bbox) for all layers.

    Returns a scalar tensor (0 if the list is empty)."""
    if not activations:
        return torch.zeros(())
    if isinstance(subcube, (list, tuple)):
        sizes = [int(subcube[min(i, len(subcube) - 1)])
                 for i in range(len(activations))]
    else:
        sizes = [int(subcube)] * len(activations)
    ratios = [
        jdx_layer_penalty(a, num_batches=num_batches, subcube=s,
                          randomize=randomize, mode=mode, region=region, eps=eps)
        for a, s in zip(activations, sizes)
    ]
    ratios = [r for r in ratios if r is not None]
    if not ratios:
        return activations[0].new_zeros(())
    return torch.stack(ratios).mean()
