"""Shared grid representation used across ARC-1/2 (pretraining) and ARC-3
(dynamics) data: a fixed 64x64 canvas, 16 ARC colors one-hot + 1 pad channel.

ARC-1/2 grids use colors 0-9 and are up to 30x30; ARC-3 frames use colors
0-15 and are exactly 64x64. Unifying both to the same (17, 64, 64) tensor
shape lets the same encoder pretrain on ARC-1/2 and then run on ARC-3.
"""

import numpy as np

NUM_COLORS = 16
PAD_CHANNEL = NUM_COLORS  # index 16
NUM_CHANNELS = NUM_COLORS + 1
CANVAS = 64


def grid_to_tensor(grid: list) -> np.ndarray:
    """(H, W) int grid, colors in [0, NUM_COLORS) -> (NUM_CHANNELS, CANVAS, CANVAS) float32 one-hot.

    Grids smaller than the canvas are placed top-left; the rest of the
    canvas is marked via the pad channel.
    """
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    if h > CANVAS or w > CANVAS:
        raise ValueError(f"grid {h}x{w} exceeds canvas {CANVAS}x{CANVAS}")

    out = np.zeros((NUM_CHANNELS, CANVAS, CANVAS), dtype=np.float32)
    out[PAD_CHANNEL, :, :] = 1.0  # everything starts as "pad"
    out[PAD_CHANNEL, :h, :w] = 0.0
    rows = arr.reshape(-1)
    color_channels = np.clip(rows, 0, NUM_COLORS - 1)
    ys, xs = np.divmod(np.arange(h * w), w)
    out[color_channels, ys, xs] = 1.0
    return out


def arc3_frame_to_tensor(frame: list) -> np.ndarray:
    """ARC-3 FrameData.frame is a list of layers, each a (64, 64) int grid.
    Stage 0 recordings only ever have one layer; flatten defensively if more
    ever appear by taking the first layer (matches what we've observed).
    """
    layer = frame[0]
    return grid_to_tensor(layer)


PATCH = 8


def patch_change_mask(frame_t: list, frame_t1: list) -> np.ndarray:
    """(8, 8) bool: True where that 8x8-pixel patch differs between frames.

    Even "changed" ARC-3 transitions are usually a single small moving
    object against an otherwise static 64x64 grid, so most patches are
    unchanged even then -- this is used to upweight the handful of patches
    that actually carry dynamics signal in the prediction loss.

    ARC-3 frames are always exactly (CANVAS, CANVAS); other sources (e.g.
    MiniGrid, jepa/data/minigrid_data.py) can be smaller -- placed top-left
    on a (CANVAS, CANVAS) "nothing changed" canvas first, matching
    grid_to_tensor's own top-left placement convention, so the reshape
    below always sees a full canvas regardless of the source grid's size.
    """
    a = np.asarray(frame_t[0], dtype=np.int64)
    b = np.asarray(frame_t1[0], dtype=np.int64)
    h, w = a.shape
    if h > CANVAS or w > CANVAS:
        raise ValueError(f"grid {h}x{w} exceeds canvas {CANVAS}x{CANVAS}")

    diff = np.zeros((CANVAS, CANVAS), dtype=bool)
    diff[:h, :w] = a != b
    n = CANVAS // PATCH
    diff = diff.reshape(n, PATCH, n, PATCH)
    return diff.any(axis=(1, 3))  # (8, 8)
