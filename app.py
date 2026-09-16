"""Upload a Sentinel-2 scene, super-resolve it, download the 2.5 m product.

The judge-facing counterpart to docs/judge_demo.html: that file is pre-rendered
and cannot fail, this one actually runs the model on imagery someone brings.
Kept deliberately simple -- one upload, one button, one slider, one download --
because app/testbench.py already exists for anyone who wants the knobs.

    streamlit run app/superresolve.py
"""
from __future__ import annotations

import base64
import io
import os
import pathlib
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[0]
os.chdir(ROOT)  # weight paths in srm.models are relative to the app root
sys.path.insert(0, str(ROOT / "src"))

import rasterio  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from srm.io.loaders import load_band_files, load_scene, overview, probe  # noqa: E402
from srm.io.raster import sr_profile, write_cog  # noqa: E402
from srm.preprocess.prepare import apply_cloud_mask  # noqa: E402
from srm.trust.layer import confidence_map  # noqa: E402
from srm.validate.wald import consistency_check  # noqa: E402

st.set_page_config(page_title="Sentinel-2 Super-Resolution", layout="wide")

BANDS = ("B04_red", "B03_green", "B02_blue", "B08_nir")
MAX_INPUT_PX = 640  # above this, inference gets slow enough to hurt a live demo


# ------------------------------------------------------------------ helpers

def stretch(rgb: np.ndarray, pct: float = 2.0) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        lo, hi = np.percentile(rgb[..., i], [pct, 100 - pct])
        out[..., i] = np.clip((rgb[..., i] - lo) / max(hi - lo, 1e-8), 0, 1)
    return out


def b64(arr_rgb: np.ndarray, size: int = 820) -> str:
    img = Image.fromarray((np.clip(arr_rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def mask_b64(conf: np.ndarray, size: int = 820) -> str:
    """Transparent red where confidence is low; invisible where it is high."""
    a = np.clip((0.62 - np.clip(conf, 0, 1)) / 0.62, 0, 1) ** 0.8 * 0.85
    rgba = np.zeros(conf.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = (a * 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def slider_html(before: str, after: str, overlay: str | None) -> str:
    ov = f'<img id="o" src="{overlay}">' if overlay else ""
    return f"""
<style>
 .wrap{{position:relative;border-radius:10px;overflow:hidden;background:#000;aspect-ratio:1;
   user-select:none;cursor:ew-resize;touch-action:none;border:1px solid #30363d}}
 .wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;-webkit-user-drag:none}}
 #a{{clip-path:inset(0 0 0 var(--p,50%))}}
 #o{{opacity:0;transition:.25s;pointer-events:none}} .wrap.on #o{{opacity:1}}
 .h{{position:absolute;top:0;bottom:0;left:var(--p,50%);width:3px;background:#fff;
   box-shadow:0 0 12px rgba(0,0,0,.85);pointer-events:none}}
 .k{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:42px;height:42px;
   border-radius:50%;background:#fff;display:grid;place-items:center;color:#0d1117;
   font-weight:700;font-size:16px;box-shadow:0 3px 14px rgba(0,0,0,.6)}}
 .t{{position:absolute;top:12px;padding:6px 12px;border-radius:6px;font:600 12.5px system-ui;
   background:rgba(13,17,23,.82);color:#e6edf3;border:1px solid #30363d}}
 .t.l{{left:12px}} .t.r{{right:12px;color:#3fb950}}
 .ctl{{display:flex;gap:14px;align-items:center;margin-top:12px;font:13px system-ui;color:#8b949e}}
 .ctl input[type=range]{{flex:1;accent-color:#2f81f7}}
 .ctl label{{display:flex;gap:7px;align-items:center;cursor:pointer;white-space:nowrap}}
</style>
<div class="wrap" id="w">
  <img id="b" src="{before}"><img id="a" src="{after}">{ov}
  <div class="t l">Input &mdash; 10 m</div><div class="t r">Super-resolved &mdash; 2.5 m</div>
  <div class="h"><div class="k">&#8646;</div></div>
</div>
<div class="ctl">
  <input type="range" id="s" min="0" max="100" value="50">
  {'<label><input type="checkbox" id="c"> Show where the model is uncertain</label>' if overlay else ''}
</div>
<script>
 const w=document.getElementById('w'),s=document.getElementById('s');
 const set=p=>{{p=Math.max(0,Math.min(100,p));w.style.setProperty('--p',p+'%');s.value=p;}};
 s.oninput=e=>set(+e.target.value);
 const c=document.getElementById('c'); if(c) c.onchange=()=>w.classList.toggle('on',c.checked);
 let d=false; const mv=e=>{{const r=w.getBoundingClientRect();
   set(((e.touches?e.touches[0].clientX:e.clientX)-r.left)/r.width*100);}};
 w.addEventListener('mousedown',e=>{{d=true;mv(e);}});
 addEventListener('mousemove',e=>{{if(d)mv(e);}}); addEventListener('mouseup',()=>d=false);
 w.addEventListener('touchstart',e=>{{d=true;mv(e);}},{{passive:true}});
 w.addEventListener('touchmove',e=>{{if(d)mv(e);}},{{passive:true}});
 addEventListener('touchend',()=>d=false); set(50);
</script>"""


@st.cache_resource(show_spinner=False)
def get_model(device: str):
    from srm.models.ours_branch import OursBranch
    return OursBranch(device=device)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------ UI

st.title("Sentinel-2 Super-Resolution")
st.caption("Upload a 10 m Sentinel-2 scene and get a 2.5 m product with a per-pixel "
           "confidence band — one file per band, or a single 4-band GeoTIFF.")

left, right = st.columns([1, 1.25])

with left:
    src = st.radio("Image source",
                   ["Upload band files", "Upload 4-band GeoTIFF", "Use a sample scene"],
                   horizontal=True)

    arr = profile = None
    name = ""
    windowed = False

    if src == "Upload band files":
        st.caption("One file per band. The slot decides which band it is, so filenames "
                   "do not matter — accepts `.jp2` straight from a SAFE product, or `.tif`.")

        # Slot order is display order (natural RGB reading), but the stack is
        # assembled in the model's B04/B03/B02/B08 order regardless.
        SLOTS = [
            ("B04", "B04 — Red", "665 nm"),
            ("B03", "B03 — Green", "560 nm"),
            ("B02", "B02 — Blue", "490 nm"),
            ("B08", "B08 — NIR", "842 nm · needed for NDVI"),
        ]
        files: dict[str, object] = {}
        rows = [st.columns(2), st.columns(2)]
        for i, (band, label, hint) in enumerate(SLOTS):
            col = rows[i // 2][i % 2]
            with col:
                f = st.file_uploader(label, type=["jp2", "tif", "tiff"], key=f"band_{band}",
                                     help=hint)
                if f is not None:
                    files[band] = f.read()
                    st.caption(f"✓ {f.name}")
                else:
                    st.caption(f"· {hint}")

        got = len(files)
        if got == 0:
            st.info("Add the four 10 m band files above. In a SAFE download they are under "
                    "`GRANULE/L2A_.../IMG_DATA/R10m/`.")
        elif got < 4:
            missing = [b for b, _, _ in SLOTS if b not in files]
            st.warning(f"{got} of 4 bands loaded — still need {', '.join(missing)}.")
        else:
            try:
                # Probe metadata before decoding anything. A real Sentinel-2 10 m
                # band is 10980x10980; decoding four of them in full is ~2 GB and
                # minutes of JPEG-2000 work, so we window first and decode second.
                meta = probe(files["B04"])
                fh, fw = meta["height"], meta["width"]
                st.success(f"All four bands loaded — {fw} × {fh} px at 10 m "
                           f"({fw*10/1000:.1f} × {fh*10/1000:.1f} km), `{meta['crs']}`")

                if max(fh, fw) > MAX_INPUT_PX:
                    st.caption(f"Full granule — pick a {MAX_INPUT_PX} px window below. "
                               "Only the selected region is decoded.")
                    size = st.slider("Window size (px)", 128,
                                     min(max(fh, fw), MAX_INPUT_PX),
                                     min(384, MAX_INPUT_PX), step=64, key="bw_size")
                    y = st.slider("Vertical position", 0, max(fh - size, 0),
                                  max(fh - size, 0) // 2, step=32, key="bw_y")
                    x = st.slider("Horizontal position", 0, max(fw - size, 0),
                                  max(fw - size, 0) // 2, step=32, key="bw_x")
                    win = (y, x, size)

                    # Show the whole granule with the window marked. A blind
                    # slider over a tile that is largely nodata sends people
                    # straight into an empty region and looks like a broken model.
                    ov = overview(files["B04"], 320)
                    thumb = np.stack([ov] * 3, axis=-1)
                    lo, hi = np.percentile(ov[ov > 0], [2, 98]) if (ov > 0).any() else (0, 1)
                    thumb = np.clip((thumb - lo) / max(hi - lo, 1e-8), 0, 1)
                    ry0, ry1 = int(y / fh * 320), int(min((y + size) / fh * 320, 319))
                    rx0, rx1 = int(x / fw * 320), int(min((x + size) / fw * 320, 319))
                    ry1, rx1 = max(ry1, ry0 + 2), max(rx1, rx0 + 2)
                    for cy, cx in ((ry0, slice(rx0, rx1)), (ry1, slice(rx0, rx1))):
                        thumb[cy, cx] = [1, 0.2, 0.2]
                    for cy, cx in ((slice(ry0, ry1), rx0), (slice(ry0, ry1), rx1)):
                        thumb[cy, cx] = [1, 0.2, 0.2]
                    st.image(thumb, caption="Whole granule — red box is your window. "
                             "Black areas contain no data.", width=340)
                else:
                    win = None

                with st.spinner("Decoding bands…"):
                    arr, profile = load_band_files(files, window=win)
                name = "scene"
                windowed = win is not None

                # A window over nodata produces a black preview and a garbage
                # super-resolution -- say so instead of letting it look broken.
                nz = float((arr > 0).mean())
                if nz < 0.02:
                    st.error("This window is empty — it contains almost no data "
                             f"({nz:.1%} non-zero). Move the red box over an area with "
                             "imagery in the granule overview above.")
                    arr = None
                elif nz < 0.5:
                    st.warning(f"Only {nz:.0%} of this window contains data — it overlaps "
                               "the granule's nodata border. Results there will be poor.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not read those files — {type(exc).__name__}: {exc}")

    elif src == "Upload 4-band GeoTIFF":
        up = st.file_uploader("4-band GeoTIFF (B04, B03, B02, B08 in that order)",
                              type=["tif", "tiff", "jp2"])
        if up:
            try:
                arr, profile, desc = load_scene([up])
                name = pathlib.Path(up.name).stem
                st.success(f"Loaded — {desc}")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not read that file — {type(exc).__name__}: {exc}")

    else:
        samples = sorted((ROOT / "data" / "raw").glob("*.tif"))
        samples = [p for p in samples if "SCL" not in p.name and "2p5m" not in p.name]
        if samples:
            pick = st.selectbox("Sample", samples, format_func=lambda p: p.name)
            if "telangana" in pick.name.lower():
                st.caption("Telangana sample \u2014 Sentinel-2 L2A, Nov 2025, 2.6 km across. "
                           "This area overlaps the KUDALIAR training site, so treat it as a "
                           "demonstration scene, not a held-out benchmark.")
            with rasterio.open(pick) as s:
                arr = s.read().astype(np.float32)
                profile = s.profile.copy()
            name = pick.stem
        else:
            st.warning("No sample scenes in data/raw/.")

    if arr is not None:
        if arr.shape[0] != 4:
            st.error(f"Expected 4 bands, found {arr.shape[0]}.")
            arr = None
        else:
            # L2A ships integer DN scaled by 10000; detect and convert.
            if arr.max() > 1.5:
                arr = np.clip(arr / 10000.0, 0, 1)
                st.info("Detected integer L2A values — converted to reflectance (÷10000).")

    if arr is not None:
        h, w = arr.shape[1], arr.shape[2]
        st.write(f"**{w} × {h} px** at 10 m &nbsp;·&nbsp; {w*10/1000:.1f} × {h*10/1000:.1f} km "
                 f"&nbsp;·&nbsp; `{profile.get('crs')}`")

        if max(h, w) > MAX_INPUT_PX and not windowed:
            st.caption(f"Large scene — a {MAX_INPUT_PX}×{MAX_INPUT_PX} window keeps this "
                       "interactive. Move the sliders to choose the area.")
            size = st.slider("Window size (px)", 128, min(max(h, w), MAX_INPUT_PX),
                             min(384, MAX_INPUT_PX), step=64)
            y = st.slider("Vertical position", 0, max(h - size, 0), max(h - size, 0) // 2, step=32)
            x = st.slider("Horizontal position", 0, max(w - size, 0), max(w - size, 0) // 2, step=32)
            arr = arr[:, y:y + size, x:x + size]
            if profile.get("transform") is not None:
                t = profile["transform"]
                profile = {**profile, "transform": rasterio.Affine(
                    t.a, t.b, t.c + x * t.a, t.d, t.e, t.f + y * t.e)}

        st.image(stretch(arr[:3].transpose(1, 2, 0)), caption="Input preview",
                 use_container_width=True)

        go = st.button("Super-resolve  →  2.5 m", type="primary", use_container_width=True)
        if go:
            dev = pick_device()
            t0 = time.perf_counter()
            with st.spinner(f"Running the model on {dev}…"):
                model = get_model(dev)
                pred = model.predict(np.ascontiguousarray(arr))
                trust = confidence_map(pred.sr, arr, scale=4)
                cons = consistency_check(pred.sr, arr, scale=4)
            st.session_state["out"] = {
                "lr": arr, "sr": pred.sr, "trust": trust, "cons": cons,
                "profile": profile, "name": name, "device": dev,
                "seconds": time.perf_counter() - t0,
            }

with right:
    out = st.session_state.get("out")
    if not out:
        st.info("Choose or upload a scene on the left, then press **Super-resolve**.")
        st.stop()

    lr, sr, trust = out["lr"], out["sr"], out["trust"]
    conf = trust["confidence"]

    # Upsample the input so both panes render at the same size -- otherwise the
    # comparison would be confounded by display scaling rather than resolution.
    import torch.nn.functional as F
    bic = F.interpolate(torch.from_numpy(lr)[None], scale_factor=4,
                        mode="bicubic", align_corners=False)[0].numpy()

    st.components.v1.html(
        slider_html(b64(stretch(bic[:3].transpose(1, 2, 0))),
                    b64(stretch(sr[:3].transpose(1, 2, 0))),
                    mask_b64(conf)),
        height=920,
    )

    a, b, c = st.columns(3)
    a.metric("Output size", f"{sr.shape[2]} × {sr.shape[1]}")
    b.metric("Mean confidence", f"{conf.mean():.3f}")
    c.metric("Processed in", f"{out['seconds']:.1f} s", help=f"on {out['device']}")

    st.caption(
        f"**Sensor consistency:** re-observing this output through the Sentinel-2 "
        f"point-spread function reproduces the input to within "
        f"**{out['cons']['consistency_mae']:.5f}** mean absolute reflectance "
        f"({out['cons']['consistency_sam_deg']:.3f}° spectral angle). "
        f"**{float((conf < 0.5).mean()):.1%}** of pixels are flagged below 0.5 confidence — "
        "the model reports where its detail is inferred rather than observed."
    )

    # Build the downloadable 6-band product.
    sigma = trust.get("sigma", np.zeros(sr.shape[1:], dtype=np.float32))
    stack = np.concatenate([sr, sigma[None].astype(np.float32), conf[None]], axis=0)
    out_dir = ROOT / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{out['name'] or 'scene'}_sr_2p5m.tif"
    try:
        prof = sr_profile(out["profile"], scale=4, count=stack.shape[0])
        write_cog(tmp, stack, prof, band_names=list(BANDS) + ["sigma", "confidence"])
        st.download_button(
            "Download 2.5 m GeoTIFF  (4 bands + sigma + confidence)",
            data=tmp.read_bytes(), file_name=tmp.name, mime="image/tiff",
            type="primary", use_container_width=True)
        st.caption(f"Cloud-Optimised GeoTIFF, {tmp.stat().st_size/1e6:.1f} MB — "
                   "same CRS and footprint as your input, pixel size divided by 4.")
    except Exception as exc:
        st.warning(f"Preview ready, but the GeoTIFF could not be written "
                   f"({type(exc).__name__}). The input may lack a CRS.")
