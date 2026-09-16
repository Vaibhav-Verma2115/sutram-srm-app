"""Torch dataset over the packed SEN2VENuS shards.

Shards are loaded lazily and cached one at a time: a free-tier Colab VM has
~12 GB RAM, so holding the whole set in memory is not an option, but reopening
a .npz per sample is far too slow.

Ordering is the awkward part. Plain `DataLoader(shuffle=True)` would jump
between shards on every sample and evict the cache each time; the original code
avoided that by not shuffling at all, which meant every epoch presented the
patches in an identical, shard-clustered order. `reshuffle()` is the compromise:
it permutes the shard order and the offsets *within* each shard, then keeps
reads grouped by shard. Each epoch sees a different order, and I/O stays linear.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


class Sen2VenusShards(Dataset):
    def __init__(
        self,
        root: str | pathlib.Path,
        split: str = "train",
        augment: bool = True,
        crop: int | None = None,
        mix_s2s2: float = 0.0,
        seed: int = 0,
    ):
        self.root = pathlib.Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.scale = manifest["scale"]
        self.shards = manifest["shards"][split]
        self.augment = augment and split == "train"
        self.crop = crop
        self.has_s2 = bool(manifest.get("with_s2"))
        self.mix_s2s2 = float(mix_s2s2)
        self._rng = np.random.default_rng(seed)
        if self.mix_s2s2 > 0:
            if not self.has_s2:
                raise ValueError(
                    "mix_s2s2 needs the native 10 m patch; rebuild the dataset with "
                    "scripts/build_dataset.py --with-s2"
                )
            if not crop and self.mix_s2s2 < 1.0:
                raise ValueError(
                    "mixing needs --crop: a VENuS pair is 64->256 and an S2->S2 pair "
                    "is 32->128, so they only batch together at a common crop size. "
                    "mix_s2s2=1.0 (pure S2->S2) is uniform and needs no crop."
                )

        # Index (shard, offset) pairs up front so __len__ is exact.
        self._counts: list[int] = []
        for name in self.shards:
            with np.load(self.root / name) as z:
                self._counts.append(int(z["lr"].shape[0]))
        self.index: list[tuple[int, int]] = [
            (si, i) for si, n in enumerate(self._counts) for i in range(n)
        ]

        self._cache_id: int | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def reshuffle(self, seed: int) -> None:
        """Re-order the epoch without breaking shard locality.

        Shards are visited in a new random order and the patches inside each
        shard are permuted, so the model never sees the same sequence twice,
        but the loader still reads one shard at a time.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.shards))
        index: list[tuple[int, int]] = []
        for si in order:
            offs = rng.permutation(self._counts[si])
            index.extend((int(si), int(o)) for o in offs)
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, si: int):
        if self._cache_id != si:
            with np.load(self.root / self.shards[si]) as z:
                self._cache = (z["lr"], z["hr"],
                               z["s2"] if self.has_s2 else None)
            self._cache_id = si
        return self._cache

    def __getitem__(self, i: int):
        si, off = self.index[i]
        lr_all, hr_all, s2_all = self._shard(si)

        if self.mix_s2s2 > 0 and self._rng.random() < self.mix_s2s2:
            # Same-sensor pair: the real 10 m Sentinel-2 patch is the target and
            # its PSF-degraded 40 m version is the input. Training partly on
            # this removes the S2 -> VENuS sensor transfer that the VENuS pairs
            # bake into the weights but that is absent at deployment.
            from ..trust.layer import psf_downsample

            hr = s2_all[off].astype(np.float32)
            lr = psf_downsample(hr, scale=self.scale)
        else:
            lr = lr_all[off].astype(np.float32)
            hr = hr_all[off].astype(np.float32)

        if self.crop and lr.shape[1] > self.crop:
            y = np.random.randint(0, lr.shape[1] - self.crop + 1)
            x = np.random.randint(0, lr.shape[2] - self.crop + 1)
            lr = lr[:, y : y + self.crop, x : x + self.crop]
            s = self.scale
            hr = hr[:, y * s : (y + self.crop) * s, x * s : (x + self.crop) * s]

        if self.augment:
            # Dihedral augmentation only. No colour jitter: shifting reflectance
            # would teach the model that radiometry is negotiable, which is the
            # one thing this pipeline must not learn.
            k = np.random.randint(4)
            if k:
                lr = np.rot90(lr, k, axes=(1, 2))
                hr = np.rot90(hr, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                lr = lr[:, :, ::-1]
                hr = hr[:, :, ::-1]

        return (
            torch.from_numpy(np.ascontiguousarray(lr)),
            torch.from_numpy(np.ascontiguousarray(hr)),
        )
