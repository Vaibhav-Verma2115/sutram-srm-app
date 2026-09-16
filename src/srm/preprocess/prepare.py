"""Sentinel-2 L2A preprocessing: cloud masking, scaling, tiling.

Implements the "Preprocessing" stage of the pipeline: SCL mask, scale,
tile with overlap.
"""

from __future__ import annotations

import numpy as np

# Scene Classification Layer classes to treat as invalid.
# 0 no-data, 1 saturated, 3 cloud shadow, 8 cloud medium prob,
# 9 cloud high prob, 10 thin cirrus.
SCL_INVALID = (0, 1, 3, 8, 9, 10)

# Model input band order for SEN2SR / LDSR-S2 (verified from mlm.json).
BAND_ORDER = ("B04", "B03", "B02", "B08")


def scl_mask(scl: np.ndarray, invalid: tuple[int, ...] = SCL_INVALID) -> np.ndarray:
    """Boolean mask, True where the pixel is valid (clear land/water)."""
    return ~np.isin(scl, invalid)


def cloud_fraction(scl: np.ndarray) -> float:
    """Fraction of the scene flagged as cloud/shadow/cirrus."""
    return float(1.0 - scl_mask(scl).mean())


def apply_cloud_mask(
    arr: np.ndarray,
    scl: np.ndarray,
    fill: float = 0.0,
    invalid: tuple[int, ...] = SCL_INVALID,
) -> tuple[np.ndarray, dict]:
    """Zero out cloud/shadow/cirrus pixels before super-resolution.

    Masking must happen *before* the model runs, not after. A super-resolution
    model handed a cloud will happily synthesise convincing high-frequency
    texture inside it -- structure that looks like ground but corresponds to
    nothing observable. Masking afterwards would leave that texture in every
    intermediate the trust layer measures.

    The SCL band is resampled by nearest-neighbour if it is at 20 m while the
    imagery is at 10 m, which is the usual L2A layout.

    Returns the masked array and a report suitable for the metrics JSON.
    """
    if scl.ndim == 3:
        scl = scl[0]

    if scl.shape != arr.shape[1:]:
        ys = np.linspace(0, scl.shape[0] - 1, arr.shape[1]).round().astype(int)
        xs = np.linspace(0, scl.shape[1] - 1, arr.shape[2]).round().astype(int)
        scl = scl[np.ix_(ys, xs)]

    valid = scl_mask(scl, invalid)
    out = arr.copy()
    out[:, ~valid] = fill
    return out, {
        "cloud_masked": True,
        "cloud_fraction": float(1.0 - valid.mean()),
        "valid_fraction": float(valid.mean()),
        "scl_classes_masked": list(invalid),
    }


def tile_positions(height: int, width: int, tile: int = 128, overlap: int = 32):
    """Top-left (row, col) positions covering the image with `overlap` px stride.

    The last row/column is clamped to the image edge so the whole raster is
    covered without padding.
    """
    step = tile - overlap
    rows = list(range(0, max(height - tile, 0) + 1, step))
    cols = list(range(0, max(width - tile, 0) + 1, step))
    if rows[-1] != height - tile and height > tile:
        rows.append(height - tile)
    if cols[-1] != width - tile and width > tile:
        cols.append(width - tile)
    return [(r, c) for r in rows for c in cols]


def hann_window(size: int, overlap: int) -> np.ndarray:
    """2-D feathering weight for seamless tile blending.

    Ramps up over `overlap` pixels at each edge and stays 1.0 in the middle,
    so overlapping predictions cross-fade instead of producing seam lines.

    The ramp is strictly positive. `np.hanning(2*overlap)` starts at exactly
    0.0, and that zero was silently destroying data: tiled_predict normalises
    by the accumulated weight, so a pixel whose only contribution carries
    weight 0 came out as 0 rather than as the prediction. Every product this
    pipeline wrote had a blacked-out first row and first column, and every
    benchmark that ran through tiling paid for it -- 4.5 dB on a 128 px output,
    where that ring is 1.6% of the pixels at full reflectance magnitude.

    Dropping the endpoint keeps the cross-fade shape and guarantees that a
    pixel covered by exactly one tile reconstructs that tile exactly.
    """
    # A floor as well as the dropped endpoint: tiled_predict divides by the
    # accumulated weight with a 1e-8 guard, and the squared corner weight of a
    # bare Hann ramp lands within an order of magnitude of that guard, so the
    # division loses precision exactly where the taper is thinnest.
    ramp = np.maximum(np.hanning(overlap * 2 + 2)[1 : overlap + 1], 1e-3)
    w = np.ones(size, dtype=np.float32)
    w[:overlap] = ramp
    w[-overlap:] = ramp[::-1]
    return np.outer(w, w).astype(np.float32)
