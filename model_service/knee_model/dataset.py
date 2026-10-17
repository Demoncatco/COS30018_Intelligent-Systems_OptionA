"""Caches preprocessed studies once on Kaggle, then serves them to training as tensors.

Usage: python -m model_service.knee_model.dataset --dataset-root ... --series-csv ... --cache-dir ...
Augmentation never flips left-right, because that would swap medial and lateral findings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model_service.knee_model.labels import label_arrays
from shared.schema import PLANES

MAX_ROTATION_DEGREES = 10.0
CONTRAST_RANGE = (0.9, 1.1)
BRIGHTNESS_RANGE = (-0.05, 0.05)


def cache_file(cache_dir: str | Path, study_id: str) -> Path:
    return Path(cache_dir) / f"{study_id}.npz"


def build_cache(
    study_ids: list[str],
    series_table: pd.DataFrame,
    dataset_root: str | Path,
    cache_dir: str | Path,
    split_folder: str = "train_series",
    overwrite: bool = False,
) -> pd.DataFrame:
    # Failed studies are recorded in the report instead of stopping the run.
    from model_service.knee_model.pipeline import load_study

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from tqdm import tqdm
        progress = tqdm(study_ids, desc="caching studies")
    except ImportError:
        progress = study_ids

    report = []
    for study_id in progress:
        out = cache_file(cache_dir, study_id)
        if out.exists() and not overwrite:
            report.append({"study_id": study_id, "status": "skipped", "quality_flags": "", "error": ""})
            continue
        try:
            study = load_study(study_id, series_table, dataset_root, split_folder=split_folder)
            plane_present = np.array([f"missing_plane:{p}" not in study.quality_flags for p in PLANES])
            np.savez(
                out,
                array=study.array.astype(np.float16),
                plane_present=plane_present,
                quality_flags=np.array(json.dumps(study.quality_flags)),
            )
            report.append({"study_id": study_id, "status": "ok",
                           "quality_flags": ";".join(study.quality_flags), "error": ""})
        except Exception as error:  # noqa: BLE001
            report.append({"study_id": study_id, "status": "failed", "quality_flags": "", "error": repr(error)})
    return pd.DataFrame(report)


def load_cached(cache_dir: str | Path, study_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(cache_file(cache_dir, study_id)) as data:
        return (data["array"].astype(np.float32), data["plane_present"].astype(bool),
                json.loads(str(data["quality_flags"])))


def augment(volume: torch.Tensor, plane_present: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # Same change for every slice of a plane; missing planes stay zero.
    import torchvision.transforms.v2.functional as F

    out = volume.clone()
    for p in range(volume.shape[0]):
        if not plane_present[p]:
            continue
        angle = float(torch.empty(1).uniform_(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES, generator=generator))
        contrast = float(torch.empty(1).uniform_(*CONTRAST_RANGE, generator=generator))
        brightness = float(torch.empty(1).uniform_(*BRIGHTNESS_RANGE, generator=generator))
        plane = F.rotate(out[p], angle)
        out[p] = (plane * contrast + brightness).clamp(0.0, 1.0)
    return out


class KneeDataset(Dataset):
    """One example per row of the label table. Studies missing from the cache should be removed beforehand."""

    def __init__(self, table: pd.DataFrame, cache_dir: str | Path, augment: bool = False, seed: int = 0) -> None:
        self.table = table.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.augment = augment
        self.labels, self.label_mask = label_arrays(self.table)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        array, plane_present, _ = load_cached(self.cache_dir, row["study_id"])
        image = torch.from_numpy(array)
        present = torch.from_numpy(plane_present)
        if self.augment:
            image = augment(image, present, self.generator)
        return {
            "image": image,
            "plane_mask": present,
            "labels": torch.from_numpy(self.labels[index]),
            "label_mask": torch.from_numpy(self.label_mask[index]),
            "sample_weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
            "study_id": row["study_id"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache every manifest study as a preprocessed .npz block.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "manifest.csv")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-folder", default="train_series")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    series = pd.read_csv(args.series_csv)
    report = build_cache(manifest["study_id"].tolist(), series, args.dataset_root, args.cache_dir,
                         split_folder=args.split_folder, overwrite=args.overwrite)
    report.to_csv(args.cache_dir / "cache_report.csv", index=False)
    print(report["status"].value_counts().to_string())
    failed = report[report["status"] == "failed"]
    if len(failed):
        print(f"\n{len(failed)} studies failed - see cache_report.csv:")
        print(failed[["study_id", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()