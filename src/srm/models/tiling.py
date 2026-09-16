"""Feathered tiled inference for fixed-input-size SR models.

SEN2SR's HardConstraint carries a low-pass mask baked to a 128x128 input, so
the model only accepts exactly that size. Real scenes are far bigger, so we
tile. Tiles are blended with a Hann window rather than hard-cropped, which is
what keeps seams from appearing along tile boundaries.

We do not use sen2sr.predict_large: it derives the output height from the input
*width* (utils.py:90), so it silently corrupts non-square rasters.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..preprocess.prepare import hann_window, tile_positions


def tiled_predict(
    lr: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    tile: int = 128,
    overlap: int = 32,
    scale: int = 4,
) -> np.ndarray:
    """Apply `fn` (a fixed `tile`-sized SR model) across an arbitrary raster.

    `fn` maps (C, tile, tile) -> (C, tile*scale, tile*scale).
    """
    c, h, w = lr.shape

    # Pad up to at least one tile so small inputs still work.
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h or pad_w:
        lr = np.pad(lr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = lr.shape[1], lr.shape[2]

    out = np.zeros((c, ph * scale, pw * scale), dtype=np.float32)
    acc = np.zeros((ph * scale, pw * scale), dtype=np.float32)
    weight = hann_window(tile * scale, overlap * scale)

    for r, col in tile_positions(ph, pw, tile=tile, overlap=overlap):
        patch = lr[:, r : r + tile, col : col + tile]
        sr_patch = fn(np.ascontiguousarray(patch))
        R, C = r * scale, col * scale
        out[:, R : R + tile * scale, C : C + tile * scale] += sr_patch * weight
        acc[R : R + tile * scale, C : C + tile * scale] += weight

    out /= np.maximum(acc, 1e-8)
    return out[:, : h * scale, : w * scale].astype(np.float32)
