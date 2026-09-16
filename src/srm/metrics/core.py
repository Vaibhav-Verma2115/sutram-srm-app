"""Reference metrics for satellite super-resolution.

Reflectance-aware: all metrics assume float arrays in reflectance units (0-1),
not 8-bit 0-255. This follows Puri & Kotze (2022, AGILE), who show that using
0-255-oriented losses/metrics on 16-bit satellite data distorts results.

Arrays are (C, H, W) float32 unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

# Sentinel-2 reflectance is scaled 0-1 after dividing by 10000.
DATA_RANGE = 1.0


def psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = DATA_RANGE) -> float:
    """Peak signal-to-noise ratio, computed with max reflectance = 1.0."""
    mse = float(np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = DATA_RANGE) -> float:
    """Mean structural similarity across bands."""
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            hr.transpose(1, 2, 0),
            sr.transpose(1, 2, 0),
            data_range=data_range,
            channel_axis=2,
        )
    )


def sam(sr: np.ndarray, hr: np.ndarray) -> float:
    """Spectral Angle Mapper, in degrees. Lower is better; 0 = identical spectra.

    This is the key spectral-consistency metric the problem statement asks for --
    it is insensitive to brightness scaling and measures only spectral shape.
    """
    a = sr.reshape(sr.shape[0], -1).astype(np.float64)
    b = hr.reshape(hr.shape[0], -1).astype(np.float64)
    dot = np.sum(a * b, axis=0)
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    valid = (na > 1e-8) & (nb > 1e-8)
    if not np.any(valid):
        return float("nan")
    cos = np.clip(dot[valid] / (na[valid] * nb[valid]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def ergas(sr: np.ndarray, hr: np.ndarray, scale: int = 4) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese. Lower is better.

    Standard pan-sharpening metric: band-wise RMSE normalised by band mean,
    scaled by the resolution ratio.
    """
    total = 0.0
    for c in range(hr.shape[0]):
        mu = float(hr[c].mean())
        if abs(mu) < 1e-8:
            continue
        rmse = float(np.sqrt(np.mean((sr[c] - hr[c]) ** 2)))
        total += (rmse / mu) ** 2
    return float(100.0 / scale * np.sqrt(total / hr.shape[0]))


def rmse(sr: np.ndarray, hr: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2)))


def evaluate(sr: np.ndarray, hr: np.ndarray, scale: int = 4) -> dict[str, float]:
    """Full reference-metric suite for one image pair."""
    return {
        "psnr": psnr(sr, hr),
        "ssim": ssim(sr, hr),
        "sam_deg": sam(sr, hr),
        "ergas": ergas(sr, hr, scale=scale),
        "rmse": rmse(sr, hr),
    }
