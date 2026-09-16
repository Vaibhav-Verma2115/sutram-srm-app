"""Streamlit demo for the SRM pipeline.

Runs entirely on precomputed GeoTIFFs so the demo cannot fail live because of a
model download, a GPU hiccup or a missing network. Point it at the outputs of
scripts/run_inference.py.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # weight paths in srm.models are relative to the app root
sys.path.insert(0, str(ROOT / "src"))

import rasterio  # noqa: E402

from srm.trust.layer import ndvi  # noqa: E402

st.set_page_config(page_title="Sutram SRM", layout="wide")


def stretch(rgb: np.ndarray, pct: float = 2.0) -> np.ndarray:
    """Percentile contrast stretch for display only -- never for analysis."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        band = rgb[..., i]
        lo, hi = np.percentile(band, [pct, 100 - pct])
        out[..., i] = np.clip((band - lo) / max(hi - lo, 1e-8), 0, 1)
    return out


@st.cache_data(show_spinner=False)
def load(path: str):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.descriptions, str(src.crs), src.res


def rgb_of(arr: np.ndarray) -> np.ndarray:
    """Band order is B04, B03, B02 -> already R, G, B."""
    return stretch(arr[:3].transpose(1, 2, 0))


st.title("Deep Learning Super-Resolution Mapping")
st.caption("Sentinel-2 10 m → 2.5 m with per-pixel trust. SIH problem statement: SRM from "
           "medium-resolution satellite imagery.")

out_dir = ROOT / "data" / "outputs"
products = sorted(p for p in out_dir.glob("*.tif"))
if not products:
    st.warning("No products found. Run `python scripts/run_inference.py` first.")
    st.stop()

choice = st.sidebar.selectbox("Product", products, format_func=lambda p: p.name)
sr, desc, crs, res = load(str(choice))

# Pair the product with the 10 m scene it came from, by name.
raw_dir = ROOT / "data" / "raw"
cands = [raw_dir / f"{choice.stem}.tif", raw_dir / f"{choice.stem}_10m.tif"]
src_guess = next((c for c in cands if c.exists()), None)
lr = load(str(src_guess))[0] if src_guess else None

st.sidebar.markdown(f"**CRS** {crs}  \n**Pixel size** {res[0]} m  \n**Bands** {len(desc)}")

metrics_path = choice.parent / f"{choice.stem}_metrics.json"
if metrics_path.exists():
    meta = json.loads(metrics_path.read_text())
    fp = meta.get("footprint", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean confidence", f"{meta['trust_summary']['mean_confidence']:.3f}")
    c2.metric("Low-confidence area", f"{meta['trust_summary']['frac_low_confidence']:.1%}")
    c3.metric("Footprint preserved", "yes" if fp.get("ok") else "NO",
              help=f"max bound drift {fp.get('max_bound_drift_m', 0):.2e} m")

tab_img, tab_trust, tab_ndvi, tab_metrics = st.tabs(
    ["Imagery", "Trust layer", "NDVI", "Metrics"]
)

with tab_img:
    cols = st.columns(2)
    if lr is not None:
        cols[0].image(rgb_of(lr), caption="Sentinel-2 input — 10 m", use_container_width=True)
    cols[1].image(rgb_of(sr), caption=f"Super-resolved — {res[0]} m", use_container_width=True)
    st.info("Detail below ~4 m is **inferred by the model, not observed**. "
            "Use the Trust layer tab before drawing conclusions from it.")

with tab_trust:
    names = list(desc)
    if "confidence" in names:
        conf = sr[names.index("confidence")]
        thr = st.slider("Flag pixels below confidence", 0.0, 1.0, 0.5, 0.05)
        c1, c2 = st.columns(2)
        c1.image(conf, caption=f"Confidence (mean {conf.mean():.3f})",
                 use_container_width=True, clamp=True)
        base = rgb_of(sr).copy()
        mask = conf < thr
        base[mask] = [1.0, 0.0, 0.0]  # flag low-trust pixels in red
        c2.image(base, caption=f"{mask.mean():.1%} of pixels below {thr:.2f}",
                 use_container_width=True)
        st.caption("Red = the model invented detail here that the sensor never observed. "
                   "Confidence fuses LR-consistency, spectral angle, branch disagreement "
                   "and diffusion sampling spread.")
    else:
        st.warning("This product has no confidence band.")

with tab_ndvi:
    if lr is not None:
        n_lr, n_sr = ndvi(lr), ndvi(sr[:4])
        c1, c2 = st.columns(2)
        c1.image((n_lr + 1) / 2, caption=f"NDVI at 10 m (mean {n_lr.mean():+.3f})",
                 use_container_width=True, clamp=True)
        c2.image((n_sr + 1) / 2, caption=f"NDVI at {res[0]} m (mean {n_sr.mean():+.3f})",
                 use_container_width=True, clamp=True)
        st.metric("NDVI mean shift", f"{n_sr.mean() - n_lr.mean():+.4f}",
                  help="Should be near zero: super-resolution must sharpen field "
                       "boundaries without moving the vegetation signal itself.")

with tab_metrics:
    bench = out_dir / "benchmark.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        st.subheader(f"Wald protocol (×{b['scale']} degrade → super-resolve → compare)")
        st.dataframe({
            "branch": list(b["metrics"]),
            "PSNR": [round(v["psnr"], 2) for v in b["metrics"].values()],
            "SSIM": [round(v["ssim"], 4) for v in b["metrics"].values()],
            "SAM°": [round(v["sam_deg"], 3) for v in b["metrics"].values()],
            "ERGAS": [round(v["ergas"], 3) for v in b["metrics"].values()],
        }, use_container_width=True)
        calib = b.get("calibration", {})
        if calib.get("bins"):
            st.subheader("Trust-layer calibration")
            st.caption(f"corr(confidence, error) = {calib['confidence_error_corr']:+.4f} — "
                       "negative means the confidence map genuinely predicts error.")
            st.bar_chart({str(round(x["conf_lo"], 1)): x["mean_error"] for x in calib["bins"]})
    if metrics_path.exists():
        st.subheader("Per-branch consistency")
        st.json(json.loads(metrics_path.read_text())["branches"])
