"""SEN2VENuS paired dataset: real cross-sensor Sentinel-2 / VENuS patches.

Why this dataset: it is the only large, openly licensed source of *genuinely
paired* Sentinel-2 and higher-resolution imagery acquired the same day, over the
same ground, already co-registered. Puri & Kotze (2022) showed that a 25-30 day
acquisition gap between LR and HR makes the model learn changes as if they were
detail, and invent objects at inference. Same-day pairs remove that failure mode
at the source.

SCALE FACTOR -- the important subtlety
--------------------------------------
SEN2VENuS pairs are 10 m Sentinel-2 against 5 m VENuS, i.e. x2 (128 -> 256 px).
Our deployment target is x4 (10 m -> 2.5 m). There is no 2.5 m reference on
Earth-wide open data, so we train the x4 *operator* at a scale where real ground
truth exists, then apply it one octave finer:

    mode "x2"  LR = real S2 10 m           HR = real VENuS 5 m     (x2, fully real)
    mode "x4"  LR = S2 degraded to 20 m    HR = real VENuS 5 m     (x4, real HR)

In x4 mode only the *input* is synthetic, and it is synthesised with the
Sentinel-2 PSF rather than bicubic, so the model still learns to invert a real
sensor response against a real high-resolution target. This is Wald's protocol
used for training rather than evaluation, and it is what makes a x4 claim
defensible without owning 2.5 m imagery.
"""

from __future__ import annotations

import io
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd
import rasterio

# SEN2VENuS stores reflectance as int16 scaled by 10000.
REFL_SCALE = 10_000.0

# Native band order in the b2b3b4b8 archives.
NATIVE_ORDER = ("B02", "B03", "B04", "B08")
# Order our models expect (verified from SEN2SR mlm.json).
MODEL_ORDER = ("B04", "B03", "B02", "B08")
# Permutation from native -> model order.
BAND_PERM = [NATIVE_ORDER.index(b) for b in MODEL_ORDER]

LR_COL = "b2b3b4b8_10m"
HR_COL = "b2b3b4b8_05m"


def read_index(site_dir: str | pathlib.Path) -> pd.DataFrame:
    """Read a site's index.csv (tab separated, first column is the patch id)."""
    site_dir = pathlib.Path(site_dir)
    df = pd.read_csv(site_dir / "index.csv", sep="\t", index_col=0)
    df.attrs["site_dir"] = str(site_dir)
    return df


def _read_member(site_dir: pathlib.Path, ref: str) -> np.ndarray:
    """Read one patch addressed as '<outer>.zip/<inner/path>.tif'."""
    outer, inner = ref.split(".zip/", 1)
    with zipfile.ZipFile(site_dir / f"{outer}.zip") as z:
        payload = z.read(inner)
    with rasterio.open(io.BytesIO(payload)) as src:
        return src.read()


def load_pair(site_dir: str | pathlib.Path, row) -> tuple[np.ndarray, np.ndarray]:
    """Load one (LR 10 m, HR 5 m) pair as float32 reflectance in model band order.

    Negative reflectance occurs in the archives as an atmospheric-correction
    artefact; it is unphysical, so we clip it away rather than train on it.
    """
    site_dir = pathlib.Path(site_dir)
    lr = _read_member(site_dir, row[LR_COL]).astype(np.float32) / REFL_SCALE
    hr = _read_member(site_dir, row[HR_COL]).astype(np.float32) / REFL_SCALE
    lr = np.clip(lr[BAND_PERM], 0.0, 1.0)
    hr = np.clip(hr[BAND_PERM], 0.0, 1.0)
    return lr, hr


def make_x4_input(lr10: np.ndarray) -> np.ndarray:
    """Degrade a real 10 m patch to 20 m so the pair against 5 m VENuS is x4.

    Uses the Sentinel-2 PSF approximation from the trust layer, not bicubic --
    a model trained to invert bicubic learns the wrong operator.
    """
    from ..trust.layer import psf_downsample

    return psf_downsample(lr10, scale=2)


def usable(
    lr: np.ndarray,
    hr: np.ndarray,
    min_std: float = 5e-3,
    max_flat_frac: float = 0.90,
) -> bool:
    """Reject patches that carry no learnable signal.

    Deliberately NOT a zero-fraction test. These archives have no no-data
    sentinel -- values simply cluster near zero over dark targets (open water,
    closed rainforest canopy), where ~30% of pixels are legitimately 0. Treating
    darkness as invalidity throws away entire biomes, so we gate on contrast
    instead: a patch teaches the model nothing if it is flat, however bright.
    """
    if not np.isfinite(lr).all() or not np.isfinite(hr).all():
        return False
    # Saturated patches (cloud tops that survived screening) are not useful.
    if (hr >= 1.0).mean() > 0.5:
        return False
    if (hr == hr.flat[0]).mean() > max_flat_frac:
        return False
    return float(hr.std()) >= min_std


# ---------------------------------------------------------------------------
# Grouping for leak-free splits
# ---------------------------------------------------------------------------
# A SEN2VENuS site is a fixed grid of ground locations imaged repeatedly. In
# KUDALIAR, 500 locations are each imaged on up to 20 dates, so splitting
# train/val by *patch* puts the same ground on both sides and the val score
# measures terrain memorisation rather than generalisation. The filename
# carries the location id, so we group on it:
#
#     KUDALIAR_137_2020-05-13_44QKE_b2b3b4b8_10m.tif
#              ^^^ location id       ^^^^^ MGRS tile
#      ^^^^^^^^ site      ^^^^^^^^^^ acquisition date

_NAME_RE = re.compile(
    r"^(?P<site>.+?)_(?P<pid>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<tile>[^_]+)_"
)


def parse_ref(ref: str) -> dict:
    """Pull site / location id / date / MGRS tile out of a patch reference."""
    name = ref.rsplit("/", 1)[-1]
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"unrecognised SEN2VENuS patch name: {name!r}")
    d = m.groupdict()
    d["pid"] = int(d["pid"])
    return d


def _as_ref(row_or_ref) -> str:
    """Accept either an index row or the LR reference string itself.

    `df.apply(fn, axis=1)` hands over a row, `df[LR_COL].map(fn)` hands over the
    cell. Both call sites are natural, so support both.
    """
    if isinstance(row_or_ref, str):
        return row_or_ref
    return row_or_ref[LR_COL]


def location_group(row_or_ref) -> str:
    """Ground-location key: everything at this key is the same patch of Earth.

    This is the correct grouping for a train/val split. Splitting on the
    acquisition instead would still leak, because every location recurs across
    dates and the model would be validated on terrain it had already seen.

    The MGRS tile is part of the key because location ids restart per tile:
    KUDALIAR spans 44QKE and 44QKF, each numbering its patches from 0.
    """
    p = parse_ref(_as_ref(row_or_ref))
    return f"{p['site']}/{p['tile']}/{p['pid']:05d}"


def acquisition_group(row_or_ref) -> str:
    """Acquisition key (site + date + tile) -- one satellite overpass."""
    p = parse_ref(_as_ref(row_or_ref))
    return f"{p['site']}/{p['tile']}/{p['date']}"
