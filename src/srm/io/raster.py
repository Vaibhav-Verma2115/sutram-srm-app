"""Geospatial raster I/O with strict CRS / transform preservation.

The single most important invariant in this project: a super-resolved product
must cover *exactly* the same ground footprint as its input. Pixel size shrinks
by `scale`; the bounding box does not move. Every write goes through
`sr_profile()` so this cannot be violated by accident.
"""

from __future__ import annotations

import pathlib

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy

# Sentinel-2 L2A reflectance scaling (L2A DN -> surface reflectance).
S2_SCALE = 10_000.0


def read_bands(path: str | pathlib.Path, bands: list[int] | None = None):
    """Read a raster as (C, H, W) float32 reflectance plus its rasterio profile."""
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        arr = src.read(idx).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def to_reflectance(arr: np.ndarray, scale: float = S2_SCALE) -> np.ndarray:
    """Convert L2A integer DN to 0-1 reflectance, clipped to physical range."""
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def sr_profile(src_profile: dict, scale: int, count: int) -> dict:
    """Build the output profile for a super-resolved raster.

    The affine transform is rescaled so pixel size divides by `scale` while the
    upper-left corner stays put -- this is what keeps the footprint identical.
    """
    t = src_profile["transform"]
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=count,
        height=src_profile["height"] * scale,
        width=src_profile["width"] * scale,
        transform=rasterio.Affine(
            t.a / scale, t.b, t.c,
            t.d, t.e / scale, t.f,
        ),
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=None,
    )
    return profile


def write_cog(
    path: str | pathlib.Path,
    arr: np.ndarray,
    profile: dict,
    band_names: list[str] | None = None,
) -> pathlib.Path:
    """Write (C, H, W) float32 as a Cloud-Optimised GeoTIFF with overviews."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {**profile, "count": arr.shape[0], "dtype": "float32"}

    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr.astype(np.float32))
            if band_names:
                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
        rio_copy(
            mem.name, str(path), driver="COG",
            compress="deflate", overview_resampling="average",
        )
    return path


def check_footprint(src_path, dst_path, tol: float = 1e-6) -> dict:
    """Verify an SR product covers the same ground footprint as its source.

    Returns a dict of the comparison; `ok` is True when CRS matches and all
    bounds agree within `tol` (in CRS units, i.e. metres for UTM).
    """
    with rasterio.open(src_path) as a, rasterio.open(dst_path) as b:
        drift = [abs(x - y) for x, y in zip(a.bounds, b.bounds)]
        return {
            "src_crs": str(a.crs),
            "dst_crs": str(b.crs),
            "crs_match": a.crs == b.crs,
            "src_bounds": tuple(a.bounds),
            "dst_bounds": tuple(b.bounds),
            "max_bound_drift_m": max(drift),
            "src_res": a.res,
            "dst_res": b.res,
            "ok": bool(a.crs == b.crs and max(drift) <= tol),
        }
