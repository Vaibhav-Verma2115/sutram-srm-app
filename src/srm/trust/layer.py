"""Trust layer: quantify how much of the super-resolved detail is trustworthy.

The problem statement is explicit that reconstructed detail is *inferred*, not
observed, and must be reported as such. This module turns that requirement into
four per-pixel signals, fused into a single confidence map:

1. LR-consistency  -- downsample the SR result through the sensor PSF and
                      compare against the true input. Disagreement means the
                      model invented radiometry that the sensor never saw.
                      This is a physical check, not a learned one.
2. Spectral angle  -- SAM between input and downsampled SR, per pixel. Catches
                      colour drift that L1 error hides.
3. Branch spread   -- disagreement between the fidelity branch and the
                      generative branch. Where a conservative regressor and a
                      diffusion model disagree, detail is model-invented.
4. Sampling sigma  -- per-pixel std-dev over N diffusion samples, when the
                      branch provides it (epistemic uncertainty).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psf_downsample(sr: np.ndarray, scale: int = 4, sigma: float | None = None) -> np.ndarray:
    """Downsample an SR image the way the sensor would see it.

    Applies a Gaussian approximation of the Sentinel-2 MTF before decimating,
    rather than naive averaging -- this is what makes the consistency check
    physically meaningful instead of a tautology.
    """
    if sigma is None:
        # Gaussian sigma that approximates an ideal box PSF of width `scale`.
        sigma = scale / 2.355
    x = torch.from_numpy(np.ascontiguousarray(sr)).float()[None]

    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()

    c = x.shape[1]
    kx = k1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), ky, groups=c)

    x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    return x[0].numpy().astype(np.float32)


def lr_consistency(sr: np.ndarray, lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """Per-LR-pixel absolute reflectance error after PSF downsampling."""
    back = psf_downsample(sr, scale=scale)
    h = min(back.shape[1], lr.shape[1])
    w = min(back.shape[2], lr.shape[2])
    return np.abs(back[:, :h, :w] - lr[:, :h, :w]).mean(axis=0)


def spectral_angle_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel spectral angle (degrees) between two multiband images."""
    dot = np.sum(a * b, axis=0)
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    denom = np.maximum(na * nb, 1e-8)
    return np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0))).astype(np.float32)


def sam_consistency(sr: np.ndarray, lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """Per-LR-pixel spectral angle between input and PSF-downsampled SR."""
    back = psf_downsample(sr, scale=scale)
    h = min(back.shape[1], lr.shape[1])
    w = min(back.shape[2], lr.shape[2])
    return spectral_angle_map(back[:, :h, :w], lr[:, :h, :w])


def ndvi(arr: np.ndarray, red_idx: int = 0, nir_idx: int = 3) -> np.ndarray:
    """NDVI from a B04/B03/B02/B08-ordered stack.

    Inputs are clamped at 0 first: SR products can carry slightly negative
    reflectance (the hard constraint acts after the model's clamp), and a
    negative nir+red falls below the epsilon denominator, exploding the ratio
    to ~1e5 instead of staying in [-1, 1]. Found via a change-detection run
    whose mean |dNDVI| came out at 1.15 -- physically impossible.
    """
    red = np.clip(arr[red_idx].astype(np.float32), 0.0, None)
    nir = np.clip(arr[nir_idx].astype(np.float32), 0.0, None)
    out = (nir - red) / np.maximum(nir + red, 1e-6)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def delta_ndvi(sr: np.ndarray, lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """Change in NDVI introduced by super-resolution, at LR scale.

    A well-behaved SR model should leave aggregate vegetation signal untouched;
    large |dNDVI| means the model altered the physical quantity analysts use.
    """
    back = psf_downsample(sr, scale=scale)
    h = min(back.shape[1], lr.shape[1])
    w = min(back.shape[2], lr.shape[2])
    return (ndvi(back[:, :h, :w]) - ndvi(lr[:, :h, :w])).astype(np.float32)


def branch_disagreement(sr_a: np.ndarray, sr_b: np.ndarray) -> np.ndarray:
    """Normalised difference between two branches' outputs.

    Uses normalised difference rather than L1 because, as opensr-test argues, ND
    is insensitive to absolute reflectance magnitude and so compares bright and
    dark targets fairly.
    """
    a = sr_a.mean(axis=0)
    b = sr_b.mean(axis=0)
    return (np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)).astype(np.float32)


def _upscale(x: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale of an LR-scale map to SR scale."""
    return np.repeat(np.repeat(x, scale, axis=0), scale, axis=1)


def _norm(x: np.ndarray, hi: float) -> np.ndarray:
    """Map [0, hi] onto [0, 1], clipped."""
    return np.clip(x / max(hi, 1e-8), 0.0, 1.0).astype(np.float32)


def confidence_map(
    sr: np.ndarray,
    lr: np.ndarray,
    scale: int = 4,
    sigma: np.ndarray | None = None,
    other_branch: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Fuse the trust signals into a per-pixel confidence map in [0, 1].

    Returns every component alongside the fused map so the demo can show *why*
    a region is flagged, not just that it is. 1.0 = fully supported by the
    observed data; 0.0 = model-invented, do not use for decisions.
    """
    weights = weights or {"consistency": 0.4, "sam": 0.2, "disagreement": 0.2, "sigma": 0.2}
    # Thresholds at which a signal counts as fully untrustworthy.
    thresholds = thresholds or {"consistency": 0.02, "sam": 3.0, "disagreement": 0.10, "sigma": 0.05}

    out: dict[str, np.ndarray] = {}
    risk = np.zeros(sr.shape[1:], dtype=np.float32)
    used = 0.0

    cons = lr_consistency(sr, lr, scale)
    out["lr_consistency"] = cons
    risk += weights["consistency"] * _upscale(_norm(cons, thresholds["consistency"]), scale)[: sr.shape[1], : sr.shape[2]]
    used += weights["consistency"]

    sam_map = sam_consistency(sr, lr, scale)
    out["sam"] = sam_map
    risk += weights["sam"] * _upscale(_norm(sam_map, thresholds["sam"]), scale)[: sr.shape[1], : sr.shape[2]]
    used += weights["sam"]

    out["delta_ndvi"] = delta_ndvi(sr, lr, scale)

    if other_branch is not None:
        dis = branch_disagreement(sr, other_branch)
        out["disagreement"] = dis
        risk += weights["disagreement"] * _norm(dis, thresholds["disagreement"])
        used += weights["disagreement"]

    if sigma is not None:
        out["sigma"] = sigma.astype(np.float32)
        risk += weights["sigma"] * _norm(sigma, thresholds["sigma"])
        used += weights["sigma"]

    out["confidence"] = np.clip(1.0 - risk / max(used, 1e-8), 0.0, 1.0).astype(np.float32)
    return out
