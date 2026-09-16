"""Common interface for every SR branch in the pipeline.

Each branch takes (C, H, W) float32 reflectance in B04/B03/B02/B08 order and
returns a `Prediction`. Branches that can express uncertainty populate `sigma`;
deterministic ones leave it None. This uniformity is what lets the trust layer
and the evaluation harness treat all four branches identically.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    """Output of one SR branch."""

    sr: np.ndarray                 # (C, H*scale, W*scale) float32 reflectance
    sigma: np.ndarray | None = None  # (H*scale, W*scale) per-pixel std-dev
    name: str = ""
    seconds: float = 0.0

    @property
    def has_uncertainty(self) -> bool:
        return self.sigma is not None


class SRBranch(abc.ABC):
    """Base class for a super-resolution branch."""

    name: str = "branch"
    scale: int = 4

    @abc.abstractmethod
    def predict(self, lr: np.ndarray) -> Prediction:
        """Super-resolve a (C, H, W) reflectance array."""
        raise NotImplementedError
