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
                           downsample=1, smooth=1.0, classes=None):
    """clDice topology loss: 1 - harmonic mean of topology precision/sensitivity.

    Rewards the predicted foreground's SKELETON lying inside the GT (and vice
    versa), i.e. preserving connectivity of thin structures (ventricle horns,
    foliate WM) -- what volumetric Dice is blind to. Train-only, differentiable
    through the predicted probabilities.

    `classes`: list of class indices to apply clDice to. None/empty => all
    foreground. Restricting to thin/tubular structures keeps clDice's win while
    sparing cortical/cerebellar SHEETS, which it over-thickens (resisting
    thinning merges sulci). downsample>1 builds skeletons at 1/ds resolution
    (presence-preserving for GT) to cut cost.

    Processed ONE class per gradient-checkpoint segment: forward stores nothing,
    and backward rebuilds only a single class's skeleton tape at a time -> peak
    memory ~ one class, not all (full-res, all-class batched OOMs an 80GB GPU).
    """
    C = probs.shape[1]
    if classes:
        class_list = [int(c) for c in classes
                      if 0 <= int(c) < C and (include_bg or int(c) != 0)]
    else:
        class_list = list(range(0 if include_bg else 1, C))
    if not class_list:
        return probs.new_zeros(())
    ds = int(downsample)
    dims = (2, 3, 4)

    def _class_cldice(vp, vg, Sg):
        Sp = soft_skeleton(vp, iters)
        tp = ((Sp * vg).sum(dims) + smooth) / (Sp.sum(dims) + smooth)   # pred skel in GT
        ts = ((Sg * vp).sum(dims) + smooth) / (Sg.sum(dims) + smooth)   # GT skel in pred
        return (1.0 - 2.0 * tp * ts / (tp + ts)).mean()                 # scalar over B

    total = probs.new_zeros(())
    for c in class_list:
        vp = probs[:, c:c + 1]                                  # soft, grad
        with torch.no_grad():
            vg = (targets == c).unsqueeze(1).to(probs.dtype)    # hard one-hot
        if ds > 1:
            vp = F.avg_pool3d(vp, kernel_size=ds, stride=ds)
            vg = F.max_pool3d(vg, kernel_size=ds, stride=ds)    # presence-preserving
        with torch.no_grad():
            Sg = soft_skeleton(vg, iters)          # GT skeleton: constant, grad-free
        if vp.requires_grad:
            total = total + checkpoint(_class_cldice, vp, vg, Sg, use_reentrant=False)
        else:
            total = total + _class_cldice(vp, vg, Sg)
    return total / len(class_list)


def tversky_loss_from_probs(probs, targets, classes=None, alpha=0.7, beta=0.3,
                            include_bg=False, smooth=1.0):
    """Soft Tversky loss on selected classes (Salehi et al. 2017).

    Tversky index = TP / (TP + alpha*FP + beta*FN); loss = 1 - index, averaged
    over classes. alpha > beta penalizes FALSE POSITIVES more than false
    negatives -> PRECISION-favoring: discourages a class bleeding outward. On
    cortex that means less spilling into sulcal CSF -> sulci stay open (crisper
    folds). Cheap/differentiable (no skeleton), train-only.

    `classes`: which class indices to apply it to (e.g. [2,6] = cortex + cereb
    cortex). None/empty => all foreground.
    """
    C = probs.shape[1]
    if classes:
        class_list = [int(c) for c in classes
                      if 0 <= int(c) < C and (include_bg or int(c) != 0)]
    else:
        class_list = list(range(0 if include_bg else 1, C))
    if not class_list:
        return probs.new_zeros(())
    total = probs.new_zeros(())
    for c in class_list:
        p = probs[:, c]
        with torch.no_grad():
            g = (targets == c).to(probs.dtype)
        tp = (p * g).sum()
        fp = (p * (1.0 - g)).sum()
        fn = ((1.0 - p) * g).sum()
        tv = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        total = total + (1.0 - tv)
    return total / len(class_list)


# ---------------------------------------------------------------------------
# Group marginalization for the auxiliary shape / topology terms.
#
# Why: at 104 classes the cortex is 68 PARCELS whose mutual boundaries are atlas
# conventions with no image evidence, so a per-parcel Tversky fights an arbitrary
# boundary and a per-parcel clDice skeletonizes a patch of a sheet. Marginalizing
# to the 18 tissue/structure groups puts the same shape pressure exactly where
# image evidence exists (cortex/WM and cortex/exterior), costs an 18-class loop
# instead of a 103-class one on a tensor 18/104 the size, and is TRAIN-ONLY: the
# deployed 104-class head and its peak memory are untouched.
#
# Measured 2026-09-08 on real MRN with GT (error_attrib.py): collapsed cortex
# Dice 0.866 vs per-parcel 0.724, and the predicted ribbon is FAT on both sides
# (7.6% of predicted cortex is truly background, 7.0% is truly cerebral WM)
# against a 12.0% miss -- an over-call/miss ratio of only ~1.28, which is why
# alpha/beta ~ 1.5 is the right starting push and 2.33 (0.7/0.3) likely
# overshoots into under-call.
# ---------------------------------------------------------------------------

# lut[104-class id] -> 0..17 group. MUST stay identical to
# predict_samples_siam.lut104_to_18(); test_group_marginalize.py asserts it.
GROUP_LUT_104_TO_18 = (
    [0]                      # 0   background
    + [2] * 68               # 1-68   ctx-lh-* (1-34) + ctx-rh-* (35-68) -> Cortex
    + [7, 7]                 # 69,70  thalamus
    + [8, 8]                 # 71,72  caudate
    + [9, 9]                 # 73,74  putamen
    + [10, 10]               # 75,76  pallidum
    + [14, 14]               # 77,78  hippocampus
    + [15, 15]               # 79,80  amygdala
    + [16, 16]               # 81,82  accumbens
    + [17, 17]               # 83,84  ventralDC
    + [1, 1]                 # 85,86  cerebral white matter
    + [3, 4, 3, 4]           # 87-90  lateral / inf-lat ventricle (L,R)
    + [11]                   # 91     3rd ventricle
    + [12]                   # 92     4th ventricle
    + [0]                    # 93     CSF -> background in 18-space
    + [13]                   # 94     brain stem
    + [5, 5]                 # 95,96  cerebellum white matter
    + [6, 6]                 # 97,98  cerebellum cortex
    + [1] * 5                # 99-103 corpus callosum -> white matter
)

GROUP_LUTS = {"lut104_to_18": GROUP_LUT_104_TO_18}


def marginalize_probs(probs, lut, n_groups):
    """Sum class probabilities into super-class probabilities.

    probs : [B, C, *spatial] softmax probabilities
    lut   : [C] long, lut[c] = group index of class c
    ->      [B, n_groups, *spatial]

    index_add over the class axis rather than an einsum with a [C, G] matrix:
    same result, one kernel, and no [C, G] operand. Differentiable through
    `probs` (the backward is a gather).
    """
    out = probs.new_zeros((probs.shape[0], int(n_groups)) + tuple(probs.shape[2:]))
    return out.index_add(1, lut, probs)


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
                 cldice_include_bg=False, cldice_classes=None,
                 tversky_weight=0.0, tversky_alpha=0.7, tversky_beta=0.3,
                 tversky_classes=None,
                 group_lut=None, group_n_classes=None,
                 group_cldice_weight=0.0, group_cldice_iters=5,
                 group_cldice_downsample=1, group_cldice_include_bg=False,
                 group_cldice_classes=None,
                 group_tversky_weight=0.0, group_tversky_alpha=0.6,
                 group_tversky_beta=0.4, group_tversky_classes=None,
                 log_terms=False):
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
        self.cldice_classes = list(cldice_classes) if cldice_classes else None
        # Tversky (precision-favoring on selected classes; default 0 => off).
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.tversky_classes = list(tversky_classes) if tversky_classes else None
        # --- MARGINALIZED (group-space) aux terms. All default 0.0 / None, so
        # a CEDiceLoss built without them is byte-identical to before. ---
        if group_lut is not None and not torch.is_tensor(group_lut):
            group_lut = torch.as_tensor(list(group_lut), dtype=torch.long)
        if group_lut is not None:
            group_lut = group_lut.to(torch.long)
        self.register_buffer("group_lut", group_lut)
        self.group_n_classes = (
            int(group_n_classes) if group_n_classes is not None
            else (int(group_lut.max().item()) + 1 if group_lut is not None else 0)
        )
        self.group_cldice_weight = float(group_cldice_weight)
        self.group_cldice_iters = int(group_cldice_iters)
        self.group_cldice_downsample = int(group_cldice_downsample)
        self.group_cldice_include_bg = bool(group_cldice_include_bg)
        self.group_cldice_classes = (list(group_cldice_classes)
                                     if group_cldice_classes else None)
        self.group_tversky_weight = float(group_tversky_weight)
        self.group_tversky_alpha = float(group_tversky_alpha)
        self.group_tversky_beta = float(group_tversky_beta)
        self.group_tversky_classes = (list(group_tversky_classes)
                                      if group_tversky_classes else None)
        if (self.group_cldice_weight != 0.0 or self.group_tversky_weight != 0.0) \
                and self.group_lut is None:
            raise ValueError(
                "group_cldice_weight/group_tversky_weight > 0 need group_lut "
                "(e.g. model.group_lut: lut104_to_18)"
            )
        # Per-term values of the LAST forward, detached, for logging only. Never
        # read by the training math. Off by default: writing to self inside
        # forward is a side effect that breaks a torch.compile graph, so runs
        # with log_terms=True should set FAST_COMPILE_LOSS=0 (the aux terms
        # already graph-break on their per-class checkpoint loop anyway).
        self.log_terms = bool(log_terms)
        self._terms = {}

    def _group_on(self):
        return (self.group_lut is not None
                and (self.group_cldice_weight != 0.0
                     or self.group_tversky_weight != 0.0))

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
        terms = {} if self.log_terms else None
        if self.w_ce != 0.0:
            _t = self._ce(log_probs, targets, C)
            loss = loss + self.w_ce * _t
            if terms is not None:
                terms["ce"] = _t.detach()
        # probs reused by Dice, boundary and clDice terms -> one exp() at most.
        _need_probs = (self.w_dice != 0.0 or self.boundary_weight != 0.0
                       or self.cldice_weight != 0.0 or self.tversky_weight != 0.0
                       or self._group_on())
        probs = log_probs.exp() if _need_probs else None
        if self.w_dice != 0.0:
            _t = soft_dice_from_probs(
                probs, targets, self.dice_smooth,
                self.generalized, self.gdl_eps,
            )
            loss = loss + self.w_dice * _t
            if terms is not None:
                terms["dice"] = _t.detach()
        if self.boundary_weight != 0.0:
            _t = boundary_loss_from_probs(
                probs, targets, self.boundary_radius, self.boundary_include_bg,
                self.boundary_downsample,
            )
            loss = loss + self.boundary_weight * _t
            if terms is not None:
                terms["boundary"] = _t.detach()
        if self.cldice_weight != 0.0:
            _t = cldice_loss_from_probs(
                probs, targets, self.cldice_iters, self.cldice_include_bg,
                self.cldice_downsample, classes=self.cldice_classes,
            )
            loss = loss + self.cldice_weight * _t
            if terms is not None:
                terms["cldice"] = _t.detach()
        if self.tversky_weight != 0.0:
            _t = tversky_loss_from_probs(
                probs, targets, classes=self.tversky_classes,
                alpha=self.tversky_alpha, beta=self.tversky_beta,
            )
            loss = loss + self.tversky_weight * _t
            if terms is not None:
                terms["tversky"] = _t.detach()
        # ---- MARGINALIZED aux terms, in 18-group space ----------------------
        # One [B, G, *spatial] tensor (18/104 the size of `probs`) plus a group
        # target volume; both freed before returning. The same clDice / Tversky
        # implementations are reused, so `group_*_classes` are indices in
        # GROUP space (2 = Cortex, 6 = CerebCortex, 3/4/11/12 = ventricles...).
        if self._group_on():
            g_probs = marginalize_probs(probs, self.group_lut,
                                        self.group_n_classes)
            g_targets = self.group_lut[targets.clamp_max(
                self.group_lut.numel() - 1)]
            if self.group_tversky_weight != 0.0:
                _t = tversky_loss_from_probs(
                    g_probs, g_targets, classes=self.group_tversky_classes,
                    alpha=self.group_tversky_alpha,
                    beta=self.group_tversky_beta,
                )
                loss = loss + self.group_tversky_weight * _t
                if terms is not None:
                    terms["group_tversky"] = _t.detach()
            if self.group_cldice_weight != 0.0:
                _t = cldice_loss_from_probs(
                    g_probs, g_targets, self.group_cldice_iters,
                    self.group_cldice_include_bg, self.group_cldice_downsample,
                    classes=self.group_cldice_classes,
                )
                loss = loss + self.group_cldice_weight * _t
                if terms is not None:
                    terms["group_cldice"] = _t.detach()
            del g_probs, g_targets
        if terms is not None:
            terms["total"] = loss.detach()
            self._terms = terms
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
