"""Bicubic baseline branch -- the control every other branch is measured against."""

from __future__ import annotations

import time

import numpy as np
import torch

from .base import Prediction, SRBranch
from .bicubic import bicubic_upsample


class BicubicBranch(SRBranch):
    name = "bicubic"

    def __init__(self, scale: int = 4):
        self.scale = scale

    def predict(self, lr: np.ndarray) -> Prediction:
        x = torch.from_numpy(np.ascontiguousarray(lr)).float()
        t0 = time.perf_counter()
        sr = bicubic_upsample(x, scale=self.scale)
        return Prediction(
            sr=sr.numpy().astype(np.float32),
            name=self.name,
            seconds=time.perf_counter() - t0,
        )
