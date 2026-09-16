"""Downstream task evaluation: does super-resolution improve actual analysis?

The problem statement asks for improved "interpretability and analytical
utility" -- a different claim from improved PSNR. PSNR says the pixels are
closer to a reference; this module asks whether an *analysis* run on the
imagery gets closer to the analysis run on the reference. That is the claim an
operational user cares about.

Two tasks, chosen because they are the PS's own examples:

  field boundaries (crop monitoring)  -- NDVI edge delineation
  built/edge structure (urban)        -- image edge agreement

Protocol per task: run the analysis on HR truth (the reference result), then on
bicubic LR-upsampled and on each SR branch, and score each against the truth
result. SR "improves analytical utility" iff its score beats bicubic's.
"""

from __future__ import annotations

import numpy as np

from ..trust.layer import ndvi


def _edges(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Canny edges on a single-band float image, robustly normalised."""
    from skimage import feature

    lo, hi = np.percentile(img, [2, 98])
    x = np.clip((img - lo) / max(hi - lo, 1e-8), 0, 1)
    return feature.canny(x, sigma=sigma)


def boundary_f1(pred_edges: np.ndarray, true_edges: np.ndarray, tol: int = 2) -> dict:
    """F1 for edge maps with a spatial tolerance.

    Exact pixel-matching would punish sub-pixel shifts that no analyst cares
    about, so a predicted edge counts as correct if a true edge lies within
    `tol` pixels (and vice versa for recall) -- the standard BSDS-style
    boundary matching, implemented via distance transforms.
    """
    from scipy import ndimage

    if not true_edges.any() or not pred_edges.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    dist_to_true = ndimage.distance_transform_edt(~true_edges)
    dist_to_pred = ndimage.distance_transform_edt(~pred_edges)
    precision = float((dist_to_true[pred_edges] <= tol).mean())
    recall = float((dist_to_pred[true_edges] <= tol).mean())
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": float(f1)}


def ndvi_field_boundaries(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Field-boundary proxy: edges of the NDVI surface.

    Adjacent fields in different growth stages differ in NDVI, so NDVI edges
    trace parcel boundaries without needing labelled cadastral data.
    """
    return _edges(ndvi(arr), sigma=sigma)


def image_edges(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Structural edges on mean reflectance -- roads, buildings, water lines."""
    return _edges(arr.mean(axis=0), sigma=sigma)


TASKS = {
    "field_boundaries_ndvi": ndvi_field_boundaries,
    "structural_edges": image_edges,
}


def evaluate_tasks(
    truth_hr: np.ndarray,
    candidates: dict[str, np.ndarray],
    tol: int = 2,
) -> dict:
    """Score every candidate against the truth analysis, per task.

    `candidates` maps branch name -> HR-sized array (bicubic upsample of the
    LR input must be among them to serve as the no-SR baseline).
    """
    out: dict = {}
    for task, fn in TASKS.items():
        true_edges = fn(truth_hr)
        rows = {}
        for name, arr in candidates.items():
            a = arr[:, : truth_hr.shape[1], : truth_hr.shape[2]]
            rows[name] = boundary_f1(fn(a), true_edges, tol=tol)
        out[task] = {"n_true_edge_px": int(true_edges.sum()), "scores": rows}
    return out
