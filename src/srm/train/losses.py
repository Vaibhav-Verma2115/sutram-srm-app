"""Loss functions for satellite super-resolution.

Deliberately NOT the standard ESRGAN recipe. Puri & Kotze (2022) showed that
VGG-19 perceptual loss -- trained on 8-bit 0-255 consumer photographs -- is a
poor fit for 16-bit reflectance imagery, and that plain adversarial training
lowers SSIM by inventing artefacts. So instead of borrowing a photographic loss,
we optimise the three things the problem statement actually asks for:

    reconstruction   L1 on reflectance          -- get the radiometry right
    spectral         spectral angle             -- preserve spectral consistency
    physical         LR-consistency through PSF -- do not invent radiometry

The LR-consistency term is the interesting one: it is the differentiable form of
the check the trust layer runs at inference. Training against it means the model
is penalised during learning for exactly the behaviour we flag at deployment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def psf_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Differentiable Sentinel-2 PSF approximation + decimation.

    Mirrors srm.trust.layer.psf_downsample so the training objective and the
    deployment-time trust check measure the same quantity.
    """
    sigma = scale / 2.355
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def sam_loss(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean spectral angle in radians. Brightness-invariant, so it constrains
    spectral *shape* independently of the L1 term's brightness matching."""
    dot = (sr * hr).sum(dim=1)
    norm = sr.norm(dim=1) * hr.norm(dim=1)
    cos = (dot / norm.clamp_min(eps)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean()


def consistency_loss(sr: torch.Tensor, lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Penalise SR output that does not re-observe as its own input."""
    return F.l1_loss(psf_downsample(sr, scale), lr)


def gradient_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """Match spatial gradients, which is what 'sharpness' actually means.

    A cheap, physically neutral stand-in for perceptual loss: it rewards putting
    edges where the reference has edges, without importing a photographic prior.
    """
    def grads(t):
        return t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :]

    sx, sy = grads(sr)
    hx, hy = grads(hr)
    return F.l1_loss(sx, hx) + F.l1_loss(sy, hy)


class SRLoss(torch.nn.Module):
    """Weighted composite objective.

    Defaults put most mass on reconstruction, with consistency weighted heavily
    enough to matter (it is a hard physical requirement, not a nicety) and the
    spectral and gradient terms as shaping penalties.

    On w_consistency: it was 0.5, and at that value the term was decorative.
    Measured over the v2 run's final epoch, the weighted contributions were

        L1 61.5%   SAM 29.0%   gradient 6.7%   consistency 2.8%

    so the one term that encodes the physics -- and the one the trust layer
    checks at inference -- supplied 3% of the gradient.

    That looked like a bug, and it was measured rather than assumed. The sweep
    (scripts/ablate.py, data/outputs/ablation_consistency.json, scored on the
    held-out S2->S2 set) says otherwise:

        w      S2->S2 SSIM   S2->S2 SAM
        0        0.8995        2.530     <- the term does matter
        0.5      0.9042        2.458     <- shipped value, already saturated
        5        0.9042        2.504
        10       0.9044        2.488

    The term is load-bearing (dropping it costs SSIM and spectral accuracy) but
    saturates by 0.5, and its small share of the loss reflects how quickly it is
    satisfied, not that it is ignored. The default therefore stays at 0.5.
    """

    def __init__(
        self,
        scale: int = 4,
        w_l1: float = 1.0,
        w_consistency: float = 0.5,
        w_sam: float = 0.1,
        w_gradient: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.w_l1 = w_l1
        self.w_consistency = w_consistency
        self.w_sam = w_sam
        self.w_gradient = w_gradient

    def forward(self, sr, hr, lr) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(sr, hr)
        cons = consistency_loss(sr, lr, self.scale)
        sam = sam_loss(sr, hr)
        grad = gradient_loss(sr, hr)
        total = (
            self.w_l1 * l1
            + self.w_consistency * cons
            + self.w_sam * sam
            + self.w_gradient * grad
        )
        return total, {
            "loss": float(total.detach()),
            "l1": float(l1.detach()),
            "consistency": float(cons.detach()),
            "sam": float(sam.detach()),
            "gradient": float(grad.detach()),
        }
