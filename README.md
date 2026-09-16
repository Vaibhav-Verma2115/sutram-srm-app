---
title: Sutram SRM
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Sentinel-2 10 m to 2.5 m super-resolution with trust map
---

# Sutram SRM — Deep Learning Super-Resolution Mapping

Sentinel-2 **10 m → 2.5 m** super-resolution with **per-pixel trust**, for the SIH problem
statement *"Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution
Satellite Imageries"*.

The problem statement asks for two things that pull against each other: reconstruct
fine-scale detail, *and* be honest that reconstructed detail is inferred rather than
observed. This pipeline never ships a super-resolved pixel without the map that says how
much to trust it.

## Three screens

| page | what it does |
|------|--------------|
| **Sentinel-2 Super-Resolution** | Upload your own scene — four band files straight from a SAFE product, or one 4-band GeoTIFF — and get a 2.5 m product back with a confidence band and a download. |
| **Product browser** | Precomputed products with imagery, trust, NDVI and benchmark tabs. Cannot fail live. |
| **Test bench** | Every branch, every knob, Wald protocol metrics, export. |

Input band order is **B04, B03, B02, B08**. Integer L2A DN is detected and scaled
automatically.

## Branches running here

| branch | role | speed on this Space |
|--------|------|---------------------|
| Bicubic | baseline control | milliseconds |
| SEN2SR | fidelity, hard radiometric constraint (ESA OpenSR) | ~0.4 s / 256² tile |
| **Ours** | team-trained ESRGAN, SEN2VENµS | ~0.3 s / 256² tile |

**LDSR-S2 is not runnable here.** It is a 2 GB diffusion checkpoint needing minutes per
tile without a GPU; its published numbers remain in the benchmark table.

## Trust layer

Confidence fuses LR-consistency, spectral angle, branch disagreement and sampling spread.
The headline check needs no ground truth, so it runs on every product: re-observe the
output through the Sentinel-2 point-spread function and compare against the real input.

| branch | consistency MAE | consistency SAM° |
|--------|-----------------|------------------|
| **Ours** | **0.00138** | **0.827** |
| SEN2SR | 0.00283 | 0.907 |
| bicubic | 0.00351 | 1.043 |

Measured on the bundled Telangana sample. Lower is better.

## Bundled samples

- `test_scene_10m.tif` — 128², the SEN2SR example patch, with `test_ref_2p5m.tif` as its
  2.5 m Wald reference.
- `telangana_2025-11_5km.tif` — 256² Sentinel-2 L2A crop, November 2025, 2.6 km across.
  **This area overlaps the KUDALIAR training site**, so it is a demonstration scene, not a
  held-out benchmark.

## Model

`checkpoints/best.pt` (3.7 MB) — trained on SEN2VENµS v2.0.0
([Zenodo 14603764](https://zenodo.org/records/14603764)): same-day, pre-co-registered
Sentinel-2 / VENµS pairs across five sites, KUDALIAR (Telangana, India) among them.
Same-day pairing means the model learns detail rather than seasonal change.

SEN2VENµS is natively ×2 (10 m → 5 m). No open 2.5 m global reference exists, so the ×4
operator is trained where real ground truth does exist: real 5 m VENµS as the target,
Sentinel-2 degraded to 20 m with the sensor PSF as the input. Only the input is synthetic,
and it is degraded with the real sensor response rather than bicubic.

The objective is not the standard ESRGAN recipe — VGG-19 perceptual loss is trained on
8-bit photographs and mismatched to 16-bit reflectance:

| term | weight | purpose |
|------|--------|---------|
| L1 on reflectance | 1.0 | radiometric accuracy |
| LR-consistency through the PSF | 0.5 | do not invent radiometry |
| spectral angle | 0.1 | spectral consistency |
| gradient | 0.1 | sharpness without a photographic prior |

## Credits

Built on [ESA OpenSR](https://opensr.eu/) — SEN2SR (Aybar et al., *RSE* 2025).
Benchmarking follows the [opensr-test](https://github.com/ESAOpenSR/opensr-test)
consistency / synthesis / hallucination framing.
