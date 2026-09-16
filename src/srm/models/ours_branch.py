"""Branch 4: our own model, trained on SEN2VENuS.

Same interface as every other branch, so the pipeline, trust layer and
evaluation harness pick it up with no special-casing.

UNCERTAINTY
-----------
Only the diffusion branch used to report a sigma, and at ~100 s per tile it
never runs in a shipped product -- so every product we wrote carried a sigma
band of zeros, and the trust layer silently fell back to two of its four
signals. `tta=True` fixes that without the diffusion cost: the scene is
super-resolved under all eight dihedral transforms of the input, the outputs
are mapped back, and the per-pixel spread across them becomes sigma.

Be precise about what that measures. It is the model's *equivariance spread* --
how much its answer depends on the orientation it was shown. A patch the model
reconstructs from strong evidence looks the same whichever way it is fed; a
patch it is inventing does not. That is a genuine and cheap uncertainty signal,
but it is narrower than the diffusion branch's sampling sigma, which explores
the full posterior. We report it as what it is.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from .base import Prediction, SRBranch
from .tiling import tiled_predict

DEFAULT_CKPT = pathlib.Path("checkpoints/best.pt")
TILE = 128


class OursBranch(SRBranch):
    name = "Ours"

    def __init__(
        self,
        ckpt: str | pathlib.Path = DEFAULT_CKPT,
        device: str = "cpu",
        scale: int | None = None,
        tta: bool = False,
    ):
        from ..train.model import build_model

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        self.scale = scale or state.get("scale", 4)
        self.device = device
        self.tta = tta
        # Infer width/depth from the checkpoint so lite and wide variants both load.
        import re
        sd = state["model"]
        feat = sd["conv_1.sk.weight"].shape[0] if "conv_1.sk.weight" in sd else 24
        idx = {int(m.group(1)) for k in sd for m in [re.match(r"blocks\.(\d+)\.", k)] if m}
        blocks = (max(idx) + 1) if idx else 6
        # train_mode=False switches SPAN to its inference-time reparameterisation.
        self.model = build_model(scale=self.scale, train_mode=False,
                                 feature_channels=feat, num_blocks=blocks)
        self.model.load_state_dict(state["model"], strict=False)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.epoch = state.get("epoch")
        self.val_psnr = state.get("best")

    @staticmethod
    def _d4(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
        """Apply one dihedral transform to a (C, H, W) array."""
        if flip:
            x = x[:, :, ::-1]
        return np.rot90(x, k, axes=(1, 2))

    @staticmethod
    def _d4_inv(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
        """Undo `_d4` -- rotation first, because _d4 flipped before rotating."""
        x = np.rot90(x, -k, axes=(1, 2))
        if flip:
            x = x[:, :, ::-1]
        return x

    def _infer_tile(self, patch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(patch).float()[None].to(self.device)
        with torch.no_grad():
            sr = self.model(x).clamp(0, 1).squeeze(0)
        return sr.cpu().numpy().astype(np.float32)

    def predict(self, lr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        if not self.tta:
            sr = tiled_predict(lr, self._infer_tile, tile=TILE, scale=self.scale)
            return Prediction(sr=sr, name=self.name, seconds=time.perf_counter() - t0)

        outs = []
        for flip in (False, True):
            for k in range(4):
                x = np.ascontiguousarray(self._d4(lr, k, flip))
                y = tiled_predict(x, self._infer_tile, tile=TILE, scale=self.scale)
                outs.append(np.ascontiguousarray(self._d4_inv(y, k, flip)))

        stack = np.stack(outs)                      # (8, C, H, W)
        sr = stack.mean(axis=0).astype(np.float32)
        # One sigma per pixel: spread across transforms, averaged over bands.
        sigma = stack.std(axis=0).mean(axis=0).astype(np.float32)
        return Prediction(sr=sr, sigma=sigma, name=self.name,
                          seconds=time.perf_counter() - t0)
