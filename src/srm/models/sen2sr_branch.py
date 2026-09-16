"""SEN2SR fidelity branch.

Wraps ESA OpenSR's SEN2SR (Aybar et al., RSE 2025). The key property for us is
the built-in HardConstraint module: the network output is forced to stay
radiometrically consistent with the input, so this branch is the one whose
pixels are safe for quantitative analysis.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from .base import Prediction, SRBranch
from .tiling import tiled_predict

# The pretrained HardConstraint mask is baked to this input size.
TILE = 128
DEFAULT_WEIGHTS = pathlib.Path("models/SEN2SRLite_NonReference_RGBN_x4")


class Sen2SRBranch(SRBranch):
    name = "SEN2SR"

    def __init__(
        self,
        weights_dir: str | pathlib.Path = DEFAULT_WEIGHTS,
        device: str = "cpu",
        scale: int = 4,
    ):
        import mlstac

        self.device = device
        self.scale = scale
        self._loader = mlstac.load(str(weights_dir))
        self.model = self._loader.compiled_model(device=device)

    def _infer_tile(self, patch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(patch).float().to(self.device)
        with torch.no_grad():
            sr = self.model(x[None]).squeeze(0)
        return sr.detach().cpu().numpy().astype(np.float32)

    def predict(self, lr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        if lr.shape[1] == TILE and lr.shape[2] == TILE:
            sr = self._infer_tile(np.ascontiguousarray(lr))
        else:
            sr = tiled_predict(lr, self._infer_tile, tile=TILE, scale=self.scale)
        return Prediction(
            sr=sr,
            name=self.name,
            seconds=time.perf_counter() - t0,
        )
