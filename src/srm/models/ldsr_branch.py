"""LDSR-S2 generative detail branch, with sampling-based uncertainty.

Wraps ESA OpenSR's latent-diffusion model. Because diffusion sampling is
stochastic, running it N times over the same input and taking the per-pixel
standard deviation gives a direct, honest measure of how much of the detail is
invented: pixels the model is sure about barely move between samples, while
hallucinated structure varies from run to run.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from .base import Prediction, SRBranch

DEFAULT_CKPT = pathlib.Path("models/ldsr/opensr_10m_v4_v6.ckpt")


class LdsrBranch(SRBranch):
    name = "LDSR-S2"

    def __init__(
        self,
        ckpt: str | pathlib.Path = DEFAULT_CKPT,
        device: str = "cpu",
        scale: int = 4,
        n_samples: int = 8,
        steps: int = 100,
    ):
        import importlib.resources as ir

        import opensr_model
        from omegaconf import OmegaConf

        self.device = device
        self.scale = scale
        self.n_samples = n_samples
        self.steps = steps

        cfg = OmegaConf.load(str(ir.files("opensr_model").joinpath("configs/config_10m.yaml")))
        self.model = opensr_model.SRLatentDiffusion(cfg, device=device)
        self.model.load_pretrained(str(ckpt))
        self.model.eval()

    def predict(self, lr: np.ndarray) -> Prediction:
        """Run N stochastic samples; return their mean and per-pixel std-dev."""
        x = torch.from_numpy(np.ascontiguousarray(lr)).float()[None].to(self.device)
        t0 = time.perf_counter()
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(x, sampling_steps=self.steps, verbose=False)
                samples.append(out.squeeze(0).detach().cpu().numpy().astype(np.float32))
        stack = np.stack(samples, axis=0)
        return Prediction(
            sr=stack.mean(axis=0),
            # Averaged across bands -> one uncertainty value per pixel.
            sigma=stack.std(axis=0).mean(axis=0),
            name=self.name,
            seconds=time.perf_counter() - t0,
        )
