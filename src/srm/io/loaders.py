"""Flexible scene loading: GeoTIFF, or Sentinel-2 SAFE JPEG-2000 bands.

A SAFE product does not contain one multi-band file -- it ships a separate
.jp2 per band under GRANULE/.../IMG_DATA/. So "upload your Sentinel-2 scene"
realistically means handing us four files named like

    T44QKE_20260310T052651_B04_10m.jp2

and expecting them assembled in the right order. Doing that assembly here,
rather than making the user build a stack in QGIS first, is the difference
between a demo someone can try and one they cannot.
"""

from __future__ import annotations

import io
import pathlib
import re

import numpy as np
import rasterio

# Order the models expect (verified from the model metadata).
MODEL_ORDER = ("B04", "B03", "B02", "B08")
# Accept the common aliases people use for the same four bands.
ALIASES = {
    "B04": ("B04", "B4", "RED"),
    "B03": ("B03", "B3", "GREEN"),
    "B02": ("B02", "B2", "BLUE"),
    "B08": ("B08", "B8", "NIR"),
}
_BAND_RE = re.compile(r"(?<![0-9A-Za-z])(B0?[2348A]|RED|GREEN|BLUE|NIR)(?![0-9A-Za-z])", re.I)


def band_from_name(name: str) -> str | None:
    """Infer which band a file holds from its filename.

    Sentinel-2 names carry the band as a token (…_B04_10m.jp2). We scan for the
    last match, because the tile id can itself contain something band-like.
    """
    stem = pathlib.Path(name).stem.upper()
    hits = _BAND_RE.findall(stem)
    if not hits:
        return None
    token = hits[-1].upper()
    if token == "B8A":          # 20 m red-edge band, not the 10 m NIR
        return None
    for canonical, alts in ALIASES.items():
        if token in alts:
            return canonical
    return None


def _open(payload) -> rasterio.DatasetReader:
    """Open bytes or a path with rasterio."""
    if isinstance(payload, (str, pathlib.Path)):
        return rasterio.open(payload)
    return rasterio.open(io.BytesIO(payload))


def probe(payload) -> dict:
    """Read a raster's metadata without decoding any pixels.

    Essential for JPEG-2000: a real Sentinel-2 10 m band is 10980x10980, which
    takes tens of seconds to decompress and ~0.5 GB as float32. Probing first
    lets the caller choose a window before paying that cost.
    """
    with _open(payload) as src:
        return {"width": src.width, "height": src.height, "count": src.count,
                "crs": src.crs, "transform": src.transform, "dtype": src.dtypes[0]}


def overview(payload, size: int = 320) -> np.ndarray:
    """Decode a heavily decimated thumbnail of a whole raster.

    JPEG-2000 stores reduced-resolution levels, so rasterio can serve this from
    a low level rather than decompressing 10980x10980 -- cheap enough to show
    the user where the data actually is before they choose a window. Without
    it the position sliders are blind, and a granule's nodata border (which can
    be most of the tile) looks identical to a bug.
    """
    with _open(payload) as src:
        return src.read(1, out_shape=(size, size)).astype(np.float32)


def _read(src, window=None) -> np.ndarray:
    """Read band 1, optionally only a window, as float32."""
    if window is None:
        return src.read(1).astype(np.float32)
    from rasterio.windows import Window
    row, col, size = window
    row = max(0, min(row, src.height - 1))
    col = max(0, min(col, src.width - 1))
    h = min(size, src.height - row)
    w = min(size, src.width - col)
    return src.read(1, window=Window(col, row, w, h)).astype(np.float32)


def load_single(payload) -> tuple[np.ndarray, dict]:
    """Load one multi-band raster (GeoTIFF or multi-band JP2)."""
    with _open(payload) as src:
        return src.read().astype(np.float32), src.profile.copy()


def load_band_files(
    files: dict[str, object],
    window: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Assemble single-band files into a stack in MODEL_ORDER.

    `files` maps band name -> bytes or path. `window` is (row, col, size) in
    pixels; when given, only that region is decoded from each file. Decoding a
    window rather than the whole band is what makes full-size Sentinel-2
    granules usable -- 10980x10980 x4 bands is ~2 GB of float32 and tens of
    seconds of JPEG-2000 decompression per band.

    Raises ValueError naming exactly which bands are missing, because "expected
    4 bands, got 3" is useless to someone staring at a folder of twelve JP2s.
    """
    missing = [b for b in MODEL_ORDER if b not in files]
    if missing:
        raise ValueError(f"missing band(s): {', '.join(missing)}")

    bands, profile, shape, full = [], None, None, None
    for name in MODEL_ORDER:
        with _open(files[name]) as src:
            if src.count != 1:
                raise ValueError(f"{name}: expected a single-band file, found {src.count}")
            if full is None:
                full = (src.height, src.width)
            elif (src.height, src.width) != full:
                raise ValueError(
                    f"{name} is {src.width}x{src.height} but the first band is "
                    f"{full[1]}x{full[0]} — all four must be the 10 m bands "
                    "(B8A, B11 and B12 are 20 m and will not match)")
            arr = _read(src, window)
            if shape is None:
                shape = arr.shape
                profile = src.profile.copy()
                if window is not None:
                    from rasterio.windows import Window
                    row, col, _ = window
                    profile["transform"] = src.window_transform(
                        Window(col, row, arr.shape[1], arr.shape[0]))
            bands.append(arr)

    profile = {**profile, "count": 4, "driver": "GTiff",
               "height": shape[0], "width": shape[1]}
    return np.stack(bands), profile


def load_scene(uploads: list) -> tuple[np.ndarray, dict, str]:
    """Load whatever the user gave us.

    Accepts one multi-band file, or a set of single-band files that are matched
    to bands by filename. Returns (array, profile, human-readable description).
    """
    if not uploads:
        raise ValueError("no files given")

    payloads = []
    for u in uploads:
        name = getattr(u, "name", str(u))
        data = u.read() if hasattr(u, "read") else u
        payloads.append((name, data))

    if len(payloads) == 1:
        name, data = payloads[0]
        arr, profile = load_single(data)
        if arr.shape[0] == 4:
            return arr, profile, f"{name} (4-band)"
        if arr.shape[0] == 1:
            band = band_from_name(name)
            hint = f" — looks like band {band}" if band else ""
            raise ValueError(
                f"{name} holds a single band{hint}. Upload all four "
                f"({', '.join(MODEL_ORDER)}) together, or supply a 4-band file.")
        raise ValueError(f"{name} has {arr.shape[0]} bands; expected 4 "
                         f"({', '.join(MODEL_ORDER)}) or single-band files per band.")

    matched: dict[str, object] = {}
    unmatched: list[str] = []
    for name, data in payloads:
        band = band_from_name(name)
        if band and band not in matched:
            matched[band] = data
        else:
            unmatched.append(name)

    if len(matched) < 4:
        found = ", ".join(sorted(matched)) or "none"
        raise ValueError(
            f"Could not identify all four bands from the filenames. "
            f"Recognised: {found}. Unmatched: {', '.join(unmatched) or 'none'}. "
            f"Files should carry the band in the name, e.g. …_B04_10m.jp2")

    arr, profile = load_band_files(matched)
    return arr, profile, f"{len(matched)} bands assembled from separate files"
