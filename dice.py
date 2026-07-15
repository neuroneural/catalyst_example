# Inspired by
# https://github.com/BBillot/SynthSeg/blob/492453421020d66ebf0e11bf0cc266754d21b895/SynthSeg/evaluate.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@torch.no_grad()
def signed_boundary_field(targets, class_id, radius):
    """Kervadec level-set field phi for ONE class, GPU-native + torch.compile-safe.

    phi(v) = out_dist(v) - in_dist(v)   (truncated L_inf / Chebyshev distance):
        < 0  strictly inside the class region  (= -dist to nearest non-class voxel)
        > 0  strictly outside                  (= +dist to nearest class voxel)
       ~ 0  on the boundary surface
    both distances are truncated at `radius`.

    We use the L_inf distance (one 3x3x3 max-pool == one dilation ring) as a
    monotone level-set surrogate for the exact Euclidean SDM used in Kervadec et
    al. 2019 (MedIA "Boundary loss for highly unbalanced segmentation"). The SIGN
    is exact; only the magnitude differs (L_inf <= L2). For a narrow boundary-band
    term this is the standard, fully-differentiable-downstream, compile-friendly
    choice, and -- crucially -- it needs NO CPU distance transform in the data
    pipeline (which at 256^3 x 18 classes would make the run data-bound).

    Distance accumulation: with coverage c_r = dilate^r(mask) (c_0 = mask),
    a voxel at true distance d has c_r = 0 exactly for r = 0..d-1, so
    dist = sum_{r=0}^{radius-1} (1 - c_r), capped at `radius`.

    targets:  [B, D, H, W] int class indices.
    returns:  [B, 1, D, H, W] float phi (targets' float dtype).
    """
    m = (targets == class_id).unsqueeze(1).to(torch.float32)   # [B,1,D,H,W]
    inv = 1.0 - m
    out_dist = inv.clone()          # r = 0 term for the OUTSIDE distance
    in_dist = m.clone()             # r = 0 term for the INSIDE distance (= 1 - inv)
    cov_out, cov_in = m, inv
    for _ in range(int(radius) - 1):                            # r = 1 .. radius-1
        cov_out = F.max_pool3d(cov_out, kernel_size=3, stride=1, padding=1)
        cov_in = F.max_pool3d(cov_in, kernel_size=3, stride=1, padding=1)
        out_dist = out_dist + (1.0 - cov_out)
        in_dist = in_dist + (1.0 - cov_in)
    return out_dist - in_dist


@torch.no_grad()
def _signed_distance_batched(masks, radius):
    """Vectorized signed_boundary_field over the CHANNEL dim (all classes at once).

    masks:  [B, K, *spatial] float 0/1 (one mask per class).
    returns phi [B, K, *spatial], same truncated-L_inf convention as
    signed_boundary_field. Fixed (K, radius) => unrolls under torch.compile.
    """
    inv = 1.0 - masks
    out_dist = inv.clone()          # r=0 term (outside distance)
    in_dist = masks.clone()         # r=0 term (inside distance = 1 - inv)
    cov_out, cov_in = masks, inv
    for _ in range(int(radius) - 1):
        cov_out = F.max_pool3d(cov_out, kernel_size=3, stride=1, padding=1)
        cov_in = F.max_pool3d(cov_in, kernel_size=3, stride=1, padding=1)
        out_dist = out_dist + (1.0 - cov_out)
        in_dist = in_dist + (1.0 - cov_in)
    return out_dist - in_dist


def boundary_loss_from_probs(probs, targets, radius=8, include_bg=False,
                             downsample=2):
    """Kervadec boundary term: mean_voxels( sum_c phi_c(v) * p_c(v) ).

    Minimizing it pulls softmax mass toward the interior of each GT region and
    penalizes probability that leaks across the boundary -- directly attacking
    "floppy" / leaky boundaries, unlike region Dice which tolerates boundary jitter.

    Speed: all foreground classes are processed at once (channel-batched), and by
    default phi is built at HALF resolution (downsample=2) -- the dominant cost is
    the max-pool dilation rings, which scale with voxel count, so half-res is ~8x
    cheaper per ring plus half the rings. Measured ~313 ms -> ~60 ms/step at 256^3.
    The coarsening uses a PRESENCE-PRESERVING max-pool (a coarse cell is 1 if ANY
    fine voxel is set), so thin 1-voxel structures survive and still get a boundary
    field. phi is then upsampled back to full res, so the FULL-res probabilities
    are supervised. Set downsample=1 for the exact full-res field (slower).

    Training memory is not a concern (80 GB A100; inference is untouched since this
    is train-only), so we favor the batched/full-res-probs path over a memory-lean
    per-class loop. phi carries no grad; gradient flows only through probs.

    probs:   [B, C, D, H, W] softmax probabilities.
    targets: [B, D, H, W] int class indices.
    """
    B, C = probs.shape[0], probs.shape[1]
    start = 0 if include_bg else 1          # class 0 is background/off-brain
    K = C - start
    ids = torch.arange(start, C, device=targets.device).view(1, K, 1, 1, 1)
    ds = int(downsample)
    with torch.no_grad():
        m = (targets.unsqueeze(1) == ids).to(probs.dtype)          # [B,K,D,H,W]
        if ds > 1:
            # presence-preserving coarsening -> thin structures are not erased
            m = F.max_pool3d(m, kernel_size=ds, stride=ds)
            r_eff = max(1, round(radius / ds))
        else:
            r_eff = radius
        phi = _signed_distance_batched(m, r_eff)                   # [B,K,d,h,w]
        if ds > 1:
            phi = F.interpolate(phi, scale_factor=ds, mode="trilinear",
                                align_corners=False) * ds          # full-res units
            phi = phi[..., :probs.shape[-3], :probs.shape[-2], :probs.shape[-1]]
    # sum over classes per voxel, then mean over batch+voxels (grad via probs)
    return (phi * probs[:, start:]).sum(dim=1).mean()


def _soft_erode(x):
    """Morphological erosion of a soft mask via 3x3x3 min-pool (= -maxpool(-x))."""
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x, iters):
    """Differentiable soft-skeleton (Shit et al. 2021, clDice). x in [0,1],
    [B,K,D,H,W]. Iterated erosion minus opening accumulates the medial axis."""
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(int(iters)):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss_from_probs(probs, targets, iters=5, include_bg=False,
                           downsample=1, smooth=1.0):
    """clDice topology loss: 1 - harmonic mean of topology precision/sensitivity.

    Rewards the predicted foreground's SKELETON lying inside the GT (and vice
    versa), i.e. preserving connectivity of thin structures (sulci, thin
    cerebellar WM) -- exactly what volumetric Dice is blind to. Train-only,
    differentiable through the predicted probabilities. Channel-batched;
    memory is fine on the training GPU. downsample>1 builds the skeletons at
    1/ds resolution (presence-preserving for GT) to cut cost.

    probs:   [B, C, D, H, W] softmax probabilities (grad).
    targets: [B, D, H, W] int class indices.
    """
    C = probs.shape[1]
    start = 0 if include_bg else 1
    K = C - start
    ids = torch.arange(start, C, device=targets.device).view(1, K, 1, 1, 1)
    V_pred = probs[:, start:]                                   # soft, grad
    with torch.no_grad():
        V_gt = (targets.unsqueeze(1) == ids).to(probs.dtype)    # hard one-hot
    ds = int(downsample)
    if ds > 1:
        V_pred = F.avg_pool3d(V_pred, kernel_size=ds, stride=ds)
        V_gt = F.max_pool3d(V_gt, kernel_size=ds, stride=ds)    # presence-preserving
    dims = (2, 3, 4)

    # The predicted skeleton sits in the grad path: ~2*iters+ pooling ops, each
    # retaining a full activation for backward. Batched over K classes at full
    # res that tape is tens of GB (OOM). So process ONE class per gradient-
    # checkpoint segment: forward stores nothing, and in backward only a single
    # class's skeleton tape is rebuilt at a time -> peak ~ one class, not K.
    def _class_cldice(vp, vg, Sg):
        Sp = soft_skeleton(vp, iters)
        tp = ((Sp * vg).sum(dims) + smooth) / (Sp.sum(dims) + smooth)   # pred skel in GT
        ts = ((Sg * vp).sum(dims) + smooth) / (Sg.sum(dims) + smooth)   # GT skel in pred
        return (1.0 - 2.0 * tp * ts / (tp + ts)).mean()                 # scalar over B

    total = probs.new_zeros(())
    for j in range(K):
        vp = V_pred[:, j:j + 1]
        vg = V_gt[:, j:j + 1]
        with torch.no_grad():
            Sg = soft_skeleton(vg, iters)          # GT skeleton: constant, grad-free
        if vp.requires_grad:
            total = total + checkpoint(_class_cldice, vp, vg, Sg, use_reentrant=False)
        else:
            total = total + _class_cldice(vp, vg, Sg)
    return total / K


def faster_dice(x, y, labels, fudge_factor=1e-8):
    """Faster PyTorch implementation of Dice scores.
    :param x: input label map as torch.Tensor
    :param y: input label map as torch.Tensor of the same size as x
    :param labels: list of labels to evaluate on
    :param fudge_factor: an epsilon value to avoid division by zero
    :return: pytorch Tensor with Dice scores in the same order as labels.
    """

    assert (
        x.shape == y.shape
    ), "both inputs should have same size, had {} and {}".format(
        x.shape, y.shape
    )

    if len(labels) > 1:
        dice_score = torch.zeros(len(labels))
        for label in labels:
            x_label = x == label
            y_label = y == label
            xy_label = (x_label & y_label).sum()
            dice_score[label] = (
                2 * xy_label / (x_label.sum() + y_label.sum() + fudge_factor)
            )

    else:
        dice_score = dice(
            x == labels[0], y == labels[0], fudge_factor=fudge_factor
        )

    return dice_score


def dice(x, y, fudge_factor=1e-8):
    """Implementation of dice scores ofr 0/1 numy array"""
    return 2 * torch.sum(x * y) / (torch.sum(x) + torch.sum(y) + fudge_factor)


def soft_dice_from_probs(probs, targets, smooth=1, generalized=False, gdl_eps=1e-6):
    """Vectorized soft-Dice from softmax probabilities.

    Memory/speed: no Python class loop (far fewer kernel launches) and no
    one-hot / per-class float volumes retained for backward. The gather+scatter
    trick keeps every intermediate at [B, C] except `probs` itself, so peak
    memory is ~ a single softmax volume.

    generalized=False -> plain multi-class soft-Dice: per (sample, class)
        1 - (2*inter + smooth)/(pred_sum + true_count + smooth), summed over
        classes, averaged over batch. Identical math to the original loop.
    generalized=True  -> Generalized Dice Loss (Sudre et al. 2017): each class
        weighted by 1/volume^2 and aggregated into ONE ratio per sample before
        averaging, which stops large structures from dominating small ones.

    Args:
        probs:   [B, C, *spatial] softmax probabilities (any float dtype).
        targets: [B, *spatial] integer class indices.
    """
    B, C = probs.shape[0], probs.shape[1]
    probs = probs.reshape(B, C, -1)                            # [B, C, V]
    t = targets.reshape(B, -1)                                 # [B, V]

    # prob assigned to the *true* class at each voxel -> [B, V] (1 channel)
    p_true = probs.gather(1, t.unsqueeze(1)).squeeze(1)

    intersection = torch.zeros(B, C, device=probs.device, dtype=probs.dtype)
    intersection.scatter_add_(1, t, p_true)                    # [B, C]

    true_count = torch.zeros(B, C, device=probs.device, dtype=probs.dtype)
    true_count.scatter_add_(1, t, torch.ones_like(p_true))     # [B, C]

    pred_sum = probs.sum(dim=2)                                # [B, C]

    if generalized:
        # inverse squared volume weighting; classes absent from a sample get
        # zero weight so empty channels neither help nor hurt.
        w = torch.where(
            true_count > 0,
            1.0 / (true_count * true_count + gdl_eps),
            torch.zeros_like(true_count),
        )
        num = 2.0 * (w * intersection).sum(dim=1) + smooth     # [B]
        den = (w * (pred_sum + true_count)).sum(dim=1) + smooth
        return (1 - num / den).mean()

    total = pred_sum + true_count                              # [B, C]
    dice = 1 - (2.0 * intersection + smooth) / (total + smooth)
    return dice.sum(dim=1).mean()


class DiceLoss(torch.nn.Module):
    """Vectorized soft-Dice (see `soft_dice_from_probs`). Takes raw logits and
    runs an fp32 softmax (AMP-safe: fp16 sums over 256^3 lose precision).

    generalized=True switches to inverse-volume-weighted Generalized Dice."""

    def __init__(self, generalized=False, gdl_eps=1e-6):
        super(DiceLoss, self).__init__()
        self.generalized = generalized
        self.gdl_eps = gdl_eps

    def forward(self, inputs, targets, smooth=1):
        probs = inputs.float().softmax(dim=1)
        return soft_dice_from_probs(
            probs, targets, smooth, self.generalized, self.gdl_eps
        )


class CEDiceLoss(torch.nn.Module):
    """Combined cross-entropy + soft-Dice that shares ONE log_softmax.

    At 104 classes / 256^3 a second full softmax volume is multiple GB; here
    log_softmax is computed once and the Dice term reuses exp(log_probs), so the
    combined loss costs ~one softmax volume instead of two. The CE term
    reproduces torch.nn.CrossEntropyLoss(weight, label_smoothing) (validate with
    _verify_dice.py before trusting on a long run -- it is opt-in via config).
    """

    def __init__(self, loss_weight=(0.5, 0.5), class_weight=None,
                 label_smoothing=0.0, generalized=False, gdl_eps=1e-6,
                 dice_smooth=1, boundary_weight=0.0, boundary_radius=8,
                 boundary_include_bg=False, boundary_downsample=2,
                 cldice_weight=0.0, cldice_iters=5, cldice_downsample=1,
                 cldice_include_bg=False):
        super(CEDiceLoss, self).__init__()
        self.w_ce, self.w_dice = float(loss_weight[0]), float(loss_weight[1])
        if class_weight is not None and not torch.is_tensor(class_weight):
            class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight", class_weight)
        self.label_smoothing = float(label_smoothing)
        self.generalized = generalized
        self.gdl_eps = gdl_eps
        self.dice_smooth = dice_smooth
        # Kervadec boundary term (default 0.0 => byte-identical to CE+Dice).
        self.boundary_weight = float(boundary_weight)
        self.boundary_radius = int(boundary_radius)
        self.boundary_include_bg = bool(boundary_include_bg)
        self.boundary_downsample = int(boundary_downsample)
        # clDice topology term (default 0.0 => byte-identical to CE+Dice).
        self.cldice_weight = float(cldice_weight)
        self.cldice_iters = int(cldice_iters)
        self.cldice_downsample = int(cldice_downsample)
        self.cldice_include_bg = bool(cldice_include_bg)

    def _ce(self, log_probs, targets, C):
        eps = self.label_smoothing
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # [B, *]
        w = self.class_weight
        if w is not None:
            w = w.to(log_probs.dtype)
            wt = w[targets]                                           # [B, *]
            nll = nll * wt
            denom = wt.sum().clamp_min(1e-8)
            view = (1, C) + (1,) * (log_probs.dim() - 2)
            smooth = -(log_probs * w.view(view)).sum(dim=1)          # [B, *]
            return ((1 - eps) * nll + (eps / C) * smooth).sum() / denom
        smooth = -log_probs.mean(dim=1)                              # [B, *]
        return ((1 - eps) * nll.sum() + eps * smooth.sum()) / nll.numel()

    def forward(self, inputs, targets):
        C = inputs.shape[1]
        log_probs = torch.log_softmax(inputs.float(), dim=1)   # single big tensor
        loss = inputs.new_zeros(())
        if self.w_ce != 0.0:
            loss = loss + self.w_ce * self._ce(log_probs, targets, C)
        # probs reused by Dice, boundary and clDice terms -> one exp() at most.
        _need_probs = (self.w_dice != 0.0 or self.boundary_weight != 0.0
                       or self.cldice_weight != 0.0)
        probs = log_probs.exp() if _need_probs else None
        if self.w_dice != 0.0:
            loss = loss + self.w_dice * soft_dice_from_probs(
                probs, targets, self.dice_smooth,
                self.generalized, self.gdl_eps,
            )
        if self.boundary_weight != 0.0:
            loss = loss + self.boundary_weight * boundary_loss_from_probs(
                probs, targets, self.boundary_radius, self.boundary_include_bg,
                self.boundary_downsample,
            )
        if self.cldice_weight != 0.0:
            loss = loss + self.cldice_weight * cldice_loss_from_probs(
                probs, targets, self.cldice_iters, self.cldice_include_bg,
                self.cldice_downsample,
            )
        return loss


class DiceLossInt(torch.nn.Module):
    def __init__(self):
        super(DiceLossInt, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Getting the number of classes from inputs
        num_classes = torch.max(inputs) + 1

        dice_loss = 0.0
        for c in range(num_classes.long()):
            true_flat = (targets == c).float()
            pred_flat = (inputs == c).float()
            intersection = (pred_flat * true_flat).sum(dim=(1, 2, 3))
            total = (pred_flat + true_flat).sum(dim=(1, 2, 3))

            # Adding the smooth term in the denominator
            dice_loss += 1 - (2.0 * intersection + smooth) / (total + smooth)

        return dice_loss.mean()
