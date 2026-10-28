"""
PyTorch Dataset: one item = one study = its sagittal, coronal and axial series.

    images  (3, depth, size, size)   planes stacked in config.PLANES order
    labels  (12,)                    0/1 in config.ABNORMALITIES order
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import ABNORMALITIES, PLANES
from dicom_io import load_series
from paths import resolve_series_dir
from preprocess import preprocess_volume


class KneeStudyDataset(Dataset):
    def __init__(self, label_table: pd.DataFrame, series_selection_csv, train_series_root,
                 depth: int = 32, size: int = 224, cache_dir=None, augment: bool = False):
        sel = pd.read_csv(series_selection_csv).set_index("study_id")
        self.table = label_table[label_table.study_id.isin(sel.index)].reset_index(drop=True)
        dropped = len(label_table) - len(self.table)
        if dropped:
            print(f"WARNING: {dropped} studies have no row in series_selection.csv and were dropped.")
        self.sel = sel
        self.root = Path(train_series_root)
        self.depth, self.size, self.augment = depth, size, augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.table)

    def _load_plane(self, study_id: str, plane: str) -> np.ndarray:
        """Preprocessed (depth, size, size) array for one plane, cached on disk after first use."""
        series_id = self.sel.loc[study_id, f"{plane}_series"]
        cache = self.cache_dir / f"{series_id}_{self.depth}x{self.size}.npy" if self.cache_dir else None
        if cache is not None and cache.exists():
            return np.load(cache)
        series = load_series(resolve_series_dir(self.root, study_id, series_id))
        arr = preprocess_volume(series.volume, self.depth, self.size)
        if cache is not None:
            np.save(cache, arr)
        return arr

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            x = x[..., ::-1]  # horizontal flip
        return x * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.05)

    def __getitem__(self, i: int) -> dict:
        row = self.table.iloc[i]
        x = np.stack([self._load_plane(row.study_id, p) for p in PLANES])
        if self.augment:
            x = self._augment(x)
        return {
            "images": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "labels": torch.from_numpy(row[ABNORMALITIES].to_numpy(dtype=np.float32)),
            "study_id": row.study_id,
            "label_source": row.label_source,
        }
