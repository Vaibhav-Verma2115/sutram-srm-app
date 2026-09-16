"""End-to-end pipeline: S2 L2A -> branches -> trust layer -> COG product.

This is the code path the flow diagram describes, in one place.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .io.raster import sr_profile, write_cog
from .trust.layer import confidence_map
from .validate.wald import consistency_check

BAND_NAMES = ("B04_red", "B03_green", "B02_blue", "B08_nir")


def run(
    lr: np.ndarray,
    branches: dict,
    scale: int = 4,
    fidelity: str = "SEN2SR",
    generative: str = "LDSR-S2",
) -> dict:
    """Run every branch on one LR array and fuse the results.

    Returns predictions, the trust maps and per-branch consistency stats.
    """
    preds = {name: b.predict(lr) for name, b in branches.items()}

    fid = preds.get(fidelity) or next(iter(preds.values()))
    gen = preds.get(generative)

    trust = confidence_map(
        fid.sr,
        lr,
        scale=scale,
        sigma=gen.sigma if gen is not None else None,
        other_branch=gen.sr if gen is not None else None,
    )

    consistency = {
        name: consistency_check(p.sr, lr, scale=scale) for name, p in preds.items()
    }
    return {
        "predictions": preds,
        "trust": trust,
        "consistency": consistency,
        "primary": fid.name,
    }


def write_product(
    out_path: str | pathlib.Path,
    result: dict,
    src_profile: dict,
    scale: int = 4,
) -> dict:
    """Write the 6-band product: 4 SR bands + sigma + confidence.

    Packing uncertainty into the same GeoTIFF as the imagery is deliberate --
    it makes it impossible for a downstream user to load the sharpened pixels
    without also having the map that says which of them to trust.
    """
    sr = result["predictions"][result["primary"]].sr
    trust = result["trust"]

    sigma = trust.get("sigma")
    if sigma is None:
        sigma = np.zeros(sr.shape[1:], dtype=np.float32)

    stack = np.concatenate(
        [sr, sigma[None].astype(np.float32), trust["confidence"][None]], axis=0
    )
    names = list(BAND_NAMES) + ["sigma", "confidence"]

    profile = sr_profile(src_profile, scale=scale, count=stack.shape[0])
    path = write_cog(out_path, stack, profile, band_names=names)

    return {"path": str(path), "bands": names, "shape": list(stack.shape)}


def write_metrics(out_path: str | pathlib.Path, result: dict) -> pathlib.Path:
    """Persist per-branch timing and consistency stats as JSON."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_branch": result["primary"],
        "branches": {
            name: {
                "seconds": round(p.seconds, 4),
                "has_uncertainty": p.has_uncertainty,
                **result["consistency"][name],
            }
            for name, p in result["predictions"].items()
        },
        "trust_summary": {
            "mean_confidence": float(result["trust"]["confidence"].mean()),
            "frac_low_confidence": float((result["trust"]["confidence"] < 0.5).mean()),
            "mean_abs_delta_ndvi": float(np.abs(result["trust"]["delta_ndvi"]).mean()),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
