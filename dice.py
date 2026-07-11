# Inspired by
# https://github.com/BBillot/SynthSeg/blob/492453421020d66ebf0e11bf0cc266754d21b895/SynthSeg/evaluate.py
import numpy as np
import torch


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
                 dice_smooth=1):
        super(CEDiceLoss, self).__init__()
        self.w_ce, self.w_dice = float(loss_weight[0]), float(loss_weight[1])
        if class_weight is not None and not torch.is_tensor(class_weight):
            class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight", class_weight)
        self.label_smoothing = float(label_smoothing)
        self.generalized = generalized
        self.gdl_eps = gdl_eps
        self.dice_smooth = dice_smooth

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
        if self.w_dice != 0.0:
            loss = loss + self.w_dice * soft_dice_from_probs(
                log_probs.exp(), targets, self.dice_smooth,
                self.generalized, self.gdl_eps,
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
