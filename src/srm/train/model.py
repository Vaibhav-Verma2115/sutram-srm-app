"""Our trained SR model: SPAN/CNNSR fine-tuned on SEN2VENuS.

We reuse the SPAN backbone that ESA ship with SEN2SRLite rather than inventing
an architecture. The reason is budget: on a free T4, starting from pretrained
weights reaches a useful model in hours, while training a transformer from
scratch would not converge inside the hackathon window at all. What makes the
result *ours* is the training data (real Indian terrain from KUDALIAR) and the
objective (physical consistency + spectral angle, see losses.py) -- not a novel
block diagram.
"""

from __future__ import annotations

import pathlib

import torch

# Matches the released SEN2SRLite checkpoint: 4->4 bands, 24 features, 6 blocks.
LITE_CONFIG = dict(in_channels=4, out_channels=4, feature_channels=24, num_blocks=6)


def build_model(scale: int = 4, train_mode: bool = True, **overrides) -> torch.nn.Module:
    from sen2sr.models.opensr_baseline.cnn import CNNSR

    cfg = {**LITE_CONFIG, **overrides}
    return CNNSR(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        feature_channels=cfg["feature_channels"],
        upscale=scale,
        bias=True,
        train_mode=train_mode,
        num_blocks=cfg["num_blocks"],
    )


def load_pretrained_weights(
    model: torch.nn.Module,
    path: str | pathlib.Path = "models/SEN2SRLite_NonReference_RGBN_x4/model.safetensor",
) -> dict:
    """Warm-start from the released SEN2SRLite weights.

    Returns the load report; mismatched tensors are skipped rather than raising,
    so a different scale or width still gets partial transfer.
    """
    import safetensors.torch

    weights = safetensors.torch.load_file(str(path))
    own = model.state_dict()
    ok = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(weights) - set(ok))
    model.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "total": len(weights), "skipped": skipped}
