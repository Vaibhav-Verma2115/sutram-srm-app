"""Reference validation via Wald's protocol and synthetic degradation.

We cannot validate a 10m -> 2.5m product directly without 2.5m ground truth, so
we validate the *model* at a scale where truth exists (Wald et al., 1997):

    degrade 10m -> 40m, super-resolve back to 10m, compare against the real 10m.

Degradation uses a Gaussian approximation of the Sentinel-2 MTF plus optional
sensor noise, not plain bicubic. This matters: a model trained and tested on
bicubic degradation learns to invert bicubic, then fails on real imagery whose
blur comes from the instrument's optics.
"""

from __future__ import annotations

import numpy as np

from ..metrics.core import evaluate
from ..trust.layer import psf_downsample


def degrade(
    hr: np.ndarray,
    scale: int = 4,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate observing `hr` with a sensor `scale` times coarser."""
    lr = psf_downsample(hr, scale=scale)
    if noise_std > 0:
        rng = rng or np.random.default_rng(0)
        lr = lr + rng.normal(0.0, noise_std, size=lr.shape).astype(np.float32)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)


def wald_protocol(
    branch,
    image: np.ndarray,
    scale: int = 4,
    noise_std: float = 0.0,
) -> dict:
    """Run Wald's protocol for one branch on one image.

    `image` plays the role of ground truth at its native resolution.
    """
    lr = degrade(image, scale=scale, noise_std=noise_std)
    pred = branch.predict(lr)
    sr = pred.sr[:, : image.shape[1], : image.shape[2]]
    metrics = evaluate(sr, image, scale=scale)
    metrics["seconds"] = pred.seconds
    metrics["branch"] = branch.name
    return metrics


def consistency_check(sr: np.ndarray, lr: np.ndarray, scale: int = 4) -> dict:
    """Scene-level LR-consistency: does the SR product re-observe as its input?

    This needs no ground truth at all, so it runs on every real product we ship.
    """
    back = psf_downsample(sr, scale=scale)
    h = min(back.shape[1], lr.shape[1])
    w = min(back.shape[2], lr.shape[2])
    back, ref = back[:, :h, :w], lr[:, :h, :w]
    from ..metrics.core import sam as sam_metric

    diff = back - ref
    return {
        "consistency_mae": float(np.abs(diff).mean()),
        "consistency_rmse": float(np.sqrt((diff ** 2).mean())),
        "consistency_sam_deg": float(sam_metric(back, ref)),
        "consistency_bias": float(diff.mean()),
    }


def calibration_curve(confidence: np.ndarray, error: np.ndarray, n_bins: int = 10) -> dict:
    """Does the confidence map actually predict error?

    Bins pixels by confidence and reports mean error per bin. A trustworthy
    trust layer shows error decreasing monotonically as confidence rises; if it
    is flat, the confidence map is decoration and we should say so.
    """
    conf = confidence.ravel()
    err = error.ravel()
    n = min(conf.size, err.size)
    conf, err = conf[:n], err[:n]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if sel.sum() == 0:
            continue
        bins.append({
            "conf_lo": float(lo),
            "conf_hi": float(hi),
            "n_pixels": int(sel.sum()),
            "mean_error": float(err[sel].mean()),
        })

    # Spearman-style monotonicity: correlation of confidence with error.
    # Should be negative -- higher confidence, lower error.
    corr = float(np.corrcoef(conf, err)[0, 1]) if conf.size > 1 else float("nan")
    return {"bins": bins, "confidence_error_corr": corr}
