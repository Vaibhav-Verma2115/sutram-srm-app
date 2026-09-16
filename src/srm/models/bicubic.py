"""Bicubic baseline branch. Always reported alongside learned models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bicubic_upsample(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Upsample (B, C, H, W) or (C, H, W) by `scale` using bicubic interpolation.

    Clamped at 0 because negative reflectance is unphysical.
    """
    squeeze = x.ndim == 3
    if squeeze:
        x = x[None]
    out = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)
    out = out.clamp(min=0.0)
    return out[0] if squeeze else out
