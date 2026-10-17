# Tiny fake studies (4 slices of 32x32) and label tables for the model tests.

from pathlib import Path

import numpy as np
import pandas as pd

from shared.schema import FINDINGS

N_SLICES, SIZE = 4, 32


def label_table(n_gold: int = 12, n_weak: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 2, size=(n_gold, len(FINDINGS))).astype(float)
    gold[0], gold[1] = 1.0, 0.0  # guarantees both classes for every finding
    weak = rng.integers(0, 2, size=(n_weak, len(FINDINGS))).astype(float)
    weak[:, :3] = np.nan  # the weak labeller "didn't know" the first three findings
    frame = pd.DataFrame(np.vstack([gold, weak]), columns=FINDINGS)
    frame.insert(0, "study_id", [f"gold{i}" for i in range(n_gold)] + [f"weak{i}" for i in range(n_weak)])
    frame.insert(1, "label_source", ["gold"] * n_gold + ["weak"] * n_weak)
    frame.insert(2, "sample_weight", [1.0] * n_gold + [0.5] * n_weak)
    return frame


def write_cache(cache_dir: Path, study_ids: list[str], missing_axial: set[str] = frozenset()) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for study_id in study_ids:
        array = rng.random((3, N_SLICES, SIZE, SIZE)).astype(np.float16)
        present = np.array([True, True, True])
        flags = []
        if study_id in missing_axial:
            array[2] = 0
            present[2] = False
            flags = ["missing_plane:axial"]
        np.savez(cache_dir / f"{study_id}.npz", array=array, plane_present=present,
                 quality_flags=np.array(__import__("json").dumps(flags)))