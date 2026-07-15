"""Boundary-aware segmentation metrics (per-class), pure numpy + scipy.

Kept torch-free so it can be unit-tested standalone and reused by any eval.
All distances are in millimetres via the `spacing` (sz, sy, sx) argument.

Metrics per class:
  * Dice           - volumetric overlap (bulk; boundary-insensitive)
  * NSD@tau        - Normalized Surface Dice: fraction of BOTH surfaces lying
                     within `tau` mm of the other surface. The metric that best
                     tracks "is this boundary acceptable" (Nikolov et al. 2018).
  * HD95           - 95th-percentile symmetric Hausdorff surface distance (mm);
                     robust worst-case boundary error / leakage catcher.
  * ASSD           - average symmetric surface distance (mm); mean boundary error.

Design notes:
  * Each class is cropped to the union bounding box (+margin) of its GT and pred
    masks before the Euclidean distance transform, so EDT runs on a small volume
    (structures are tiny vs 256^3) -> ~2 orders of magnitude faster, exact result.
  * Surface voxels = mask XOR erosion(mask) (6-connectivity by default).
  * Edge cases return NaN with a status string so the aggregator can decide how
    to treat "class absent in GT" vs "present in GT but nothing predicted".
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _bbox_union(a, b, margin, shape):
    """Union bounding box of two boolean volumes, padded by `margin`, clamped."""
    idx = np.argwhere(a | b)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def _surface(mask, struct):
    """Boundary voxels: mask AND NOT eroded(mask). Erosion pads with 0, so the
    volume border is treated as outside (a mask touching the border has surface
    there) -- correct for cropped sub-volumes with a background margin."""
    return mask & ~ndimage.binary_erosion(mask, structure=struct, border_value=0)


def class_surface_metrics(gt_mask, pred_mask, spacing=(1.0, 1.0, 1.0),
                          tau=1.0, margin=4, connectivity=1):
    """Return dict of Dice, NSD@tau, HD95, ASSD for one binary class.

    gt_mask, pred_mask: bool ndarrays, same shape.
    status: 'ok' | 'absent_gt' | 'empty_pred' | 'empty_gt_and_pred'.
    """
    gt_sum = int(gt_mask.sum())
    pred_sum = int(pred_mask.sum())
    out = {"dice": np.nan, "nsd": np.nan, "hd95": np.nan, "assd": np.nan,
           "gt_vox": gt_sum, "pred_vox": pred_sum, "status": "ok"}

    if gt_sum == 0 and pred_sum == 0:
        out["status"] = "empty_gt_and_pred"
        return out

    # Dice is always well-defined when at least one is non-empty.
    inter = int(np.logical_and(gt_mask, pred_mask).sum())
    out["dice"] = 2.0 * inter / (gt_sum + pred_sum)

    if gt_sum == 0:
        # nothing in GT to compare a boundary against
        out["status"] = "absent_gt"
        return out
    if pred_sum == 0:
        # GT present, model predicted none -> worst boundary agreement
        out["status"] = "empty_pred"
        out["nsd"] = 0.0
        out["hd95"] = np.inf
        out["assd"] = np.inf
        return out

    sl = _bbox_union(gt_mask, pred_mask, margin, gt_mask.shape)
    g = gt_mask[sl]
    p = pred_mask[sl]
    struct = ndimage.generate_binary_structure(3, connectivity)

    g_surf = _surface(g, struct)
    p_surf = _surface(p, struct)
    if g_surf.sum() == 0 or p_surf.sum() == 0:
        # fully solid tiny blob with no interior to erode; fall back to the mask
        g_surf = g_surf if g_surf.sum() else g
        p_surf = p_surf if p_surf.sum() else p

    # distance (mm) from every voxel to the nearest surface of the other mask
    dt_to_p = ndimage.distance_transform_edt(~p_surf, sampling=spacing)
    dt_to_g = ndimage.distance_transform_edt(~g_surf, sampling=spacing)
    d_g = dt_to_p[g_surf]   # gt surface -> pred surface
    d_p = dt_to_g[p_surf]   # pred surface -> gt surface

    out["hd95"] = float(max(np.percentile(d_g, 95), np.percentile(d_p, 95)))
    out["assd"] = float((d_g.sum() + d_p.sum()) / (len(d_g) + len(d_p)))
    within = (np.count_nonzero(d_g <= tau) + np.count_nonzero(d_p <= tau))
    out["nsd"] = float(within / (len(d_g) + len(d_p)))
    return out


def all_class_metrics(gt, pred, n_classes, spacing=(1.0, 1.0, 1.0), tau=1.0,
                      include_bg=False):
    """Per-class metrics for an integer label volume pair.

    gt, pred: int ndarrays [D,H,W] with values in 0..n_classes-1.
    returns: dict {class_index: metrics_dict}.
    """
    start = 0 if include_bg else 1
    return {c: class_surface_metrics(gt == c, pred == c, spacing=spacing, tau=tau)
            for c in range(start, n_classes)}
