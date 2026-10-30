"""
Turns a loaded series volume (slices, H, W) into the fixed (depth, size, size) array the notebook trained on.
These steps must stay identical to the notebook's preprocessing, or the model sees data it was not trained on:
1st/99th percentile clip, z-score, `depth` evenly spaced slices, bilinear resize to size x size.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEPTH = 32
SIZE = 224


def normalize_intensity(vol: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    lo, hi = np.percentile(vol, [low_pct, high_pct])
    vol = np.clip(vol, lo, hi)
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def sample_slices(vol: np.ndarray, depth: int) -> np.ndarray:
    idx = np.linspace(0, vol.shape[0] - 1, depth).round().astype(int)
    return vol[idx]


def resize_slices(vol: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(1)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy().astype(np.float32)


def preprocess_volume(vol: np.ndarray, depth: int = DEPTH, size: int = SIZE) -> np.ndarray:
    """(slices, H, W) -> (depth, size, size) float32."""
    return resize_slices(sample_slices(normalize_intensity(vol), depth), size)
