"""Interactive test bench — runs the models live.

Separate from app/demo.py on purpose. The demo is deliberately offline-safe and
reads only precomputed products, because a presentation must not depend on a
model loading correctly in front of an audience. This app is the opposite: it
exists to actually exercise the models, so it loads weights, runs inference and
reports what happened, including failures.

    streamlit run app/testbench.py
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # weight paths in srm.models are relative to the app root
sys.path.insert(0, str(ROOT / "src"))

import rasterio  # noqa: E402
import torch  # noqa: E402

from srm.io.raster import check_footprint, read_bands, sr_profile, to_reflectance, write_cog  # noqa: E402
from srm.metrics.core import evaluate  # noqa: E402
from srm.preprocess.prepare import apply_cloud_mask  # noqa: E402
from srm.trust.layer import confidence_map, ndvi  # noqa: E402
from srm.validate.wald import consistency_check, degrade  # noqa: E402

st.set_page_config(page_title="SRM Test Bench", layout="wide")

BAND_NAMES = ("B04_red", "B03_green", "B02_blue", "B08_nir")


# ---------------------------------------------------------------- helpers

def stretch(rgb: np.ndarray, pct: float = 2.0) -> np.ndarray:
    """Percentile contrast stretch. Display only — never fed back into analysis."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        band = rgb[..., i]
        lo, hi = np.percentile(band, [pct, 100 - pct])
        out[..., i] = np.clip((band - lo) / max(hi - lo, 1e-8), 0, 1)
    return out


def rgb_of(arr: np.ndarray) -> np.ndarray:
    """Band order is B04, B03, B02 → already R, G, B."""
    return stretch(arr[:3].transpose(1, 2, 0))


def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.insert(0, "mps")
    if torch.cuda.is_available():
        devs.insert(0, "cuda")
    return devs


@st.cache_resource(show_spinner=False)
def load_branch(kind: str, device: str, n_samples: int = 4, steps: int = 50):
    """Model loading is expensive; cache per (kind, device) for the session."""
    if kind == "bicubic":
        from srm.models.bicubic_branch import BicubicBranch
        return BicubicBranch()
    if kind == "sen2sr":
        from srm.models.sen2sr_branch import Sen2SRBranch
        return Sen2SRBranch(device=device)
    if kind == "ours":
        from srm.models.ours_branch import OursBranch
        return OursBranch(device=device)
    if kind == "ldsr":
        from srm.models.ldsr_branch import LdsrBranch
        return LdsrBranch(device=device, n_samples=n_samples, steps=steps)
    raise ValueError(kind)


def checkpoint_info() -> str | None:
    ck = ROOT / "checkpoints" / "best.pt"
    if not ck.exists():
        return None
    try:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        return f"epoch {state.get('epoch')}, val PSNR {state.get('best', 0):.2f} dB"
    except Exception:
        return "present"


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Test bench")
st.sidebar.caption("Runs the models live on CPU. For the presentation-safe viewer, "
                   "open the Product browser page.")

st.sidebar.subheader("1. Input")
source = st.sidebar.radio("Source", ["Sample scenes", "Upload GeoTIFF"], label_visibility="collapsed")

arr = None
profile = None
label = ""

if source == "Sample scenes":
    scenes = sorted((ROOT / "data" / "raw").glob("*.tif"))
    if scenes:
        pick = st.sidebar.selectbox("Scene", scenes, format_func=lambda p: p.name)
        arr, profile = read_bands(str(pick))
        label = pick.name
    else:
        st.sidebar.warning("No scenes in data/raw/. Run scripts/make_test_scene.py")
else:
    up = st.sidebar.file_uploader("4-band GeoTIFF (B04 B03 B02 B08)", type=["tif", "tiff"])
    if up:
        with rasterio.open(io.BytesIO(up.read())) as src:
            arr = src.read().astype(np.float32)
            profile = src.profile.copy()
        label = up.name

if arr is not None and arr.shape[0] != 4:
    st.sidebar.error(f"Expected 4 bands, got {arr.shape[0]}. "
                     "Order must be B04, B03, B02, B08.")
    arr = None

if arr is not None:
    if arr.max() > 1.5:
        st.sidebar.info(f"Max value {arr.max():.0f} → treating as L2A DN, scaling by 1/10000")
        arr = to_reflectance(arr)

    st.sidebar.subheader("2. Cloud mask")
    scl_up = st.sidebar.file_uploader("SCL band (optional)", type=["tif", "tiff"], key="scl")
    cloud = {"cloud_masked": False}
    if scl_up:
        with rasterio.open(io.BytesIO(scl_up.read())) as s:
            scl = s.read()
        arr, cloud = apply_cloud_mask(arr, scl)
        st.sidebar.success(f"Masked {cloud['cloud_fraction']:.1%} of scene")

    st.sidebar.subheader("3. Crop")
    max_dim = max(arr.shape[1], arr.shape[2])
    if max_dim > 256:
        size = st.sidebar.slider("Crop size (px)", 128, min(max_dim, 1024),
                                 min(256, max_dim), step=64,
                                 help="Smaller crops run faster. Full scenes can be slow.")
        y = st.sidebar.slider("Row offset", 0, max(arr.shape[1] - size, 0), 0, step=32)
        x = st.sidebar.slider("Col offset", 0, max(arr.shape[2] - size, 0), 0, step=32)
        arr = arr[:, y:y + size, x:x + size]

    st.sidebar.subheader("4. Branches")
    devices = available_devices()
    device = st.sidebar.selectbox("Device", devices,
                                  help="mps = Apple GPU, ~23x faster than cpu for our model")

    have_ckpt = checkpoint_info()
    sel = {
        "bicubic": st.sidebar.checkbox("Bicubic (baseline)", True),
        "sen2sr": st.sidebar.checkbox("SEN2SR (fidelity)", True),
        "ours": st.sidebar.checkbox(f"Ours — {have_ckpt}" if have_ckpt else "Ours (no checkpoint)",
                                    bool(have_ckpt), disabled=not have_ckpt),
        # LDSR-S2 is a 2 GB diffusion checkpoint and needs minutes per tile on a
        # free CPU Space; its published numbers stay in the Metrics tab.
        "ldsr": False,
    }
    if sel["ldsr"]:
        n_samples = st.sidebar.slider("LDSR samples (for sigma)", 2, 8, 4)
        steps = st.sidebar.slider("Diffusion steps", 20, 100, 50, step=10)
    else:
        n_samples, steps = 4, 50

    st.sidebar.subheader("5. Mode")
    wald = st.sidebar.toggle(
        "Wald protocol (measure accuracy)", False,
        help="Degrades the input x4 and super-resolves it back, so the original acts as "
             "ground truth. The only way to get true PSNR/SSIM without owning 2.5 m imagery.")

    run = st.sidebar.button("Run inference", type="primary", use_container_width=True)


# ---------------------------------------------------------------- main

st.title("Super-Resolution Test Bench")

if arr is None:
    st.info("Select or upload a 4-band Sentinel-2 GeoTIFF in the sidebar to begin.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input size", f"{arr.shape[2]} × {arr.shape[1]}")
c2.metric("Bands", arr.shape[0])
c3.metric("Reflectance", f"{arr.min():.3f} – {arr.max():.3f}")
c4.metric("CRS", str(profile.get("crs", "—")).replace("EPSG:", "EPSG "))

if run:
    chosen = [k for k, v in sel.items() if v]
    if not chosen:
        st.error("Select at least one branch.")
        st.stop()

    # In Wald mode the loaded scene plays the role of ground truth and the model
    # sees a degraded copy, so the metrics below are true reference metrics.
    if wald:
        truth = arr.copy()
        lr = degrade(truth, scale=4)
        st.info(f"Wald protocol: {truth.shape[1]}×{truth.shape[2]} degraded to "
                f"{lr.shape[1]}×{lr.shape[2]}, super-resolving back for comparison.")
    else:
        truth = None
        lr = arr

    preds, errors = {}, {}
    bar = st.progress(0.0, "Loading models…")
    for i, kind in enumerate(chosen):
        try:
            bar.progress(i / len(chosen), f"Running {kind}…")
            br = load_branch(kind, device, n_samples, steps)
            t0 = time.perf_counter()
            preds[br.name] = br.predict(lr)
            preds[br.name].seconds = time.perf_counter() - t0
        except Exception as exc:  # surface failures rather than hiding them
            errors[kind] = f"{type(exc).__name__}: {exc}"
    bar.empty()

    for kind, msg in errors.items():
        st.error(f"**{kind}** failed — {msg}")
    if not preds:
        st.stop()

    st.session_state["result"] = {
        "preds": preds, "lr": lr, "truth": truth, "wald": wald,
        "profile": profile, "label": label, "cloud": cloud,
    }

res = st.session_state.get("result")
if not res:
    st.caption("Configure the run in the sidebar, then press **Run inference**.")
    st.stop()

preds, lr, truth = res["preds"], res["lr"], res["truth"]

tabs = st.tabs(["Comparison", "Trust layer", "Metrics", "NDVI", "Export"])

# --- comparison
with tabs[0]:
    names = list(preds)
    cols = st.columns(min(len(names) + 1, 4))
    cols[0].image(rgb_of(lr), caption=f"Input — {lr.shape[2]}×{lr.shape[1]}",
                  use_container_width=True)
    for i, n in enumerate(names[:3], start=1):
        p = preds[n]
        cols[i].image(rgb_of(p.sr),
                      caption=f"{n} — {p.sr.shape[2]}×{p.sr.shape[1]} ({p.seconds*1000:.0f} ms)",
                      use_container_width=True)
    if truth is not None:
        st.image(rgb_of(truth), caption="Ground truth (original before degradation)", width=380)
    st.caption("Contrast-stretched for display only. Analysis uses raw reflectance.")

# --- trust
with tabs[1]:
    fid = preds.get("SEN2SR") or preds.get("Ours") or list(preds.values())[0]
    gen = preds.get("LDSR-S2")
    t = confidence_map(fid.sr, lr, scale=4,
                       sigma=gen.sigma if gen else None,
                       other_branch=gen.sr if gen else None)
    conf = t["confidence"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean confidence", f"{conf.mean():.3f}")
    m2.metric("Low-confidence area", f"{float((conf < 0.5).mean()):.1%}")
    m3.metric("Mean |ΔNDVI|", f"{np.abs(t['delta_ndvi']).mean():.5f}")

    thr = st.slider("Flag pixels below confidence", 0.0, 1.0, 0.5, 0.05)
    a, b = st.columns(2)
    a.image(conf, caption=f"Confidence map (based on {fid.name})",
            use_container_width=True, clamp=True)
    flagged = rgb_of(fid.sr).copy()
    flagged[conf < thr] = [1.0, 0.0, 0.0]
    b.image(flagged, caption=f"{float((conf < thr).mean()):.1%} flagged below {thr:.2f}",
            use_container_width=True)
    st.caption("Red = detail the model produced that the sensor data does not support. "
               "Confidence fuses LR-consistency, spectral angle, branch disagreement and "
               "diffusion sampling spread.")

    with st.expander("Individual trust signals"):
        sig = st.columns(3)
        sig[0].image(t["lr_consistency"], caption="LR-consistency error",
                     use_container_width=True, clamp=True)
        sig[1].image(t["sam"] / max(t["sam"].max(), 1e-8), caption="Spectral angle",
                     use_container_width=True, clamp=True)
        if "disagreement" in t:
            sig[2].image(t["disagreement"], caption="Branch disagreement",
                         use_container_width=True, clamp=True)
        elif "sigma" in t:
            sig[2].image(t["sigma"] / max(t["sigma"].max(), 1e-8), caption="Diffusion sigma",
                         use_container_width=True, clamp=True)

# --- metrics
with tabs[2]:
    if truth is not None:
        st.subheader("Reference metrics (Wald protocol)")
        rows = []
        for n, p in preds.items():
            sr = p.sr[:, :truth.shape[1], :truth.shape[2]]
            m = evaluate(sr, truth, scale=4)
            rows.append({"branch": n, "PSNR (dB)": round(m["psnr"], 2),
                         "SSIM": round(m["ssim"], 4), "SAM (°)": round(m["sam_deg"], 3),
                         "ERGAS": round(m["ergas"], 3), "ms": round(p.seconds * 1000)})
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Higher PSNR/SSIM is better; lower SAM/ERGAS is better.")
    else:
        st.info("Enable **Wald protocol** in the sidebar to get reference metrics. "
                "Without ground truth, only the consistency check below is meaningful.")

    st.subheader("LR-consistency (no ground truth needed)")
    st.caption("Downsamples each output through the Sentinel-2 PSF and compares against the "
               "input. This is the quality signal that works on real scenes.")
    crows = []
    for n, p in preds.items():
        c = consistency_check(p.sr, lr, scale=4)
        crows.append({"branch": n, "MAE": round(c["consistency_mae"], 5),
                      "RMSE": round(c["consistency_rmse"], 5),
                      "SAM (°)": round(c["consistency_sam_deg"], 3),
                      "bias": round(c["consistency_bias"], 6)})
    st.dataframe(crows, use_container_width=True, hide_index=True)

# --- ndvi
with tabs[3]:
    st.caption("Super-resolution should sharpen field boundaries without moving the "
               "vegetation signal. A large mean shift means the model altered a physical "
               "quantity analysts depend on.")
    n_lr = ndvi(lr)
    cols = st.columns(min(len(preds) + 1, 4))
    cols[0].image((n_lr + 1) / 2, caption=f"Input NDVI (mean {n_lr.mean():+.3f})",
                  use_container_width=True, clamp=True)
    for i, (n, p) in enumerate(list(preds.items())[:3], start=1):
        n_sr = ndvi(p.sr)
        cols[i].image((n_sr + 1) / 2,
                      caption=f"{n} (mean {n_sr.mean():+.3f}, shift {n_sr.mean()-n_lr.mean():+.4f})",
                      use_container_width=True, clamp=True)

# --- export
with tabs[4]:
    st.caption("Writes a 6-band COG: 4 super-resolved bands plus sigma and confidence, "
               "with the input's CRS and footprint preserved exactly.")
    which = st.selectbox("Branch to export", list(preds))
    name = st.text_input("Output name", f"{pathlib.Path(res['label']).stem}_{which}_sr")
    if st.button("Write GeoTIFF", type="primary"):
        p = preds[which]
        gen = preds.get("LDSR-S2")
        t = confidence_map(p.sr, lr, scale=4,
                           sigma=gen.sigma if gen else None,
                           other_branch=gen.sr if gen else None)
        sigma = t.get("sigma", np.zeros(p.sr.shape[1:], dtype=np.float32))
        stack = np.concatenate([p.sr, sigma[None], t["confidence"][None]], axis=0)

        out_dir = ROOT / "data" / "outputs"
        out_path = out_dir / f"{name}.tif"
        prof = sr_profile(res["profile"], scale=4, count=stack.shape[0])
        write_cog(out_path, stack, prof, band_names=list(BAND_NAMES) + ["sigma", "confidence"])

        meta = {"branch": which, "seconds": round(p.seconds, 3),
                "preprocessing": res["cloud"],
                "consistency": consistency_check(p.sr, lr, scale=4),
                "trust": {"mean_confidence": float(t["confidence"].mean()),
                          "frac_low_confidence": float((t["confidence"] < 0.5).mean())}}
        try:
            src_ref = ROOT / "data" / "raw" / res["label"]
            if src_ref.exists():
                meta["footprint"] = check_footprint(str(src_ref), str(out_path))
        except Exception:
            pass
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(meta, indent=2))

        st.success(f"Wrote `{out_path}` ({out_path.stat().st_size/1e6:.1f} MB)")
        if meta.get("footprint"):
            fp = meta["footprint"]
            st.write(f"Footprint preserved: **{fp['ok']}** "
                     f"(drift {fp['max_bound_drift_m']:.2e} m, "
                     f"{fp['src_res'][0]:g} m → {fp['dst_res'][0]:g} m)")
        st.json(meta["consistency"])
