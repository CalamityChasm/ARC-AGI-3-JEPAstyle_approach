"""Loads every ARC-1 / ARC-2 grid (train+test, input+output, across both
training and evaluation splits) as a flat list -- used for masked-patch
encoder pretraining. Grid identity (which task/pair it came from) doesn't
matter for this pretext task.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..grid import CANVAS, PAD_CHANNEL, grid_to_tensor

MASK_PATCH = 8  # matches the encoder's 8x8 downsampling stride
MASK_RATIO = 0.4


def _iter_task_files(repo_root: Path):
    for name in ("ARC-AGI-1", "ARC-AGI-2"):
        base = repo_root / "data" / name / "data"
        if not base.exists():
            continue
        yield from base.glob("*/*.json")


def load_all_grids(repo_root: Path) -> list:
    import json

    grids = []
    for f in _iter_task_files(repo_root):
        task = json.loads(f.read_text(encoding="utf-8"))
        for split in ("train", "test"):
            for pair in task.get(split, []):
                for key in ("input", "output"):
                    g = pair.get(key)
                    if g is None:
                        continue
                    h, w = len(g), len(g[0])
                    if h <= CANVAS and w <= CANVAS:
                        grids.append(g)
    return grids


class MaskedGridDataset(Dataset):
    """Returns (masked_tensor, unmasked_tensor, mask) triples.

    mask is (H/8, W/8) bool, True where that 8x8 patch was masked out (and
    is therefore the target for the prediction loss).
    """

    def __init__(self, grids: list, mask_ratio: float = MASK_RATIO):
        self.grids = grids
        self.mask_ratio = mask_ratio
        self.n_patches_per_side = CANVAS // MASK_PATCH

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx: int):
        unmasked = grid_to_tensor(self.grids[idx])  # (17, 64, 64)

        n = self.n_patches_per_side
        patch_mask = np.random.rand(n, n) < self.mask_ratio  # (8, 8) bool

        masked = unmasked.copy()
        full_mask = np.kron(patch_mask, np.ones((MASK_PATCH, MASK_PATCH), dtype=bool))
        masked[:, full_mask] = 0.0
        masked[PAD_CHANNEL, full_mask] = 1.0  # mark masked cells via the pad channel

        return (
            torch.from_numpy(masked),
            torch.from_numpy(unmasked),
            torch.from_numpy(patch_mask),
        )
