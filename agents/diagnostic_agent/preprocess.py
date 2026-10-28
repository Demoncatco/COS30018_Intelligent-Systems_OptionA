"""
Turns a loaded series volume (slices, H, W) into a fixed-size array a network can batch.
Each step is its own function so it can be changed or tested separately.
"""

import numpy as np
import torch
import torch.nn.functional as F


def normalize_intensity(vol: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    """MRI intensities are not standardised between scanners: clip outliers, then z-score."""
    lo, hi = np.percentile(vol, [low_pct, high_pct])
    vol = np.clip(vol, lo, hi)
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def sample_slices(vol: np.ndarray, depth: int) -> np.ndarray:
    """Pick `depth` slices spread evenly through the series (repeats slices if it is shorter)."""
    idx = np.linspace(0, vol.shape[0] - 1, depth).round().astype(int)
    return vol[idx]


def resize_slices(vol: np.ndarray, size: int) -> np.ndarray:
    """Resize every slice to size x size (bilinear)."""
    t = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(1)  # (D, 1, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy().astype(np.float32)


def preprocess_volume(vol: np.ndarray, depth: int = 32, size: int = 224) -> np.ndarray:
    """(slices, H, W) -> (depth, size, size) float32."""
    return resize_slices(sample_slices(normalize_intensity(vol), depth), size)
