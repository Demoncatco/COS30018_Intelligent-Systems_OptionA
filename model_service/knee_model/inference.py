"""Turns a trained KneeNet checkpoint into PredictionResponse objects.

Used in two places so both give exactly the same answer for a study:
- the Model Service in MODEL_MODE=live (model_service/api/main.py)
- the Kaggle exporter that writes predictions.json for MODEL_MODE=replay (export_predictions.py)

torch is only imported when a KneePredictor is created, so stub and replay modes run without it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from shared.schema import (
    FINDINGS,
    PLANES,
    Laterality,
    LateralitySource,
    PlaneScores,
    PredictionResponse,
)

if TYPE_CHECKING:
    import torch

# DICOM tags that can say which knee was scanned. RSNA data is anonymised, so often neither is present.
LATERALITY_TAGS = ("Laterality", "ImageLaterality")
_LATERALITY_CODES: dict[str, Laterality] = {"L": "left", "R": "right"}


def build_prediction(
    study_id: str,
    study_probs: np.ndarray,
    plane_probs: np.ndarray,
    plane_present: np.ndarray,
    *,
    quality_flags: list[str],
    laterality: Laterality,
    laterality_source: LateralitySource,
    model_version: str,
    latency_ms: float,
) -> PredictionResponse:
    # Probabilities in, contract out. study_probs: (12,), plane_probs: (3, 12), plane_present: (3,).
    # Missing planes are left out of per_plane (their head only ever saw a black block) and flagged.
    study_probs = np.asarray(study_probs, dtype=np.float64)
    plane_probs = np.asarray(plane_probs, dtype=np.float64)
    plane_present = np.asarray(plane_present, dtype=bool)

    if study_probs.shape != (len(FINDINGS),):
        raise ValueError(f"study_probs must have shape ({len(FINDINGS)},), got {study_probs.shape}")
    if plane_probs.shape != (len(PLANES), len(FINDINGS)):
        raise ValueError(f"plane_probs must have shape ({len(PLANES)}, {len(FINDINGS)}), got {plane_probs.shape}")
    if plane_present.shape != (len(PLANES),):
        raise ValueError(f"plane_present must have shape ({len(PLANES)},), got {plane_present.shape}")
    if not plane_present.any():
        raise ValueError(f"study {study_id} has no planes to score")
    if not (np.isfinite(study_probs).all() and np.isfinite(plane_probs).all()):
        raise ValueError(f"study {study_id} produced NaN or infinite probabilities")

    # float16 sigmoid can round to exactly 0 or 1; clip guards against tiny overshoots from float maths.
    study_probs = np.clip(study_probs, 0.0, 1.0)
    plane_probs = np.clip(plane_probs, 0.0, 1.0)

    per_plane = [
        PlaneScores(plane=plane, scores={name: float(p) for name, p in zip(FINDINGS, plane_probs[i])})
        for i, plane in enumerate(PLANES)
        if plane_present[i]
    ]

    flags = list(dict.fromkeys(quality_flags))  # keeps order, drops repeats
    for i, plane in enumerate(PLANES):
        flag = f"missing_plane:{plane}"
        if not plane_present[i] and flag not in flags:
            flags.append(flag)

    return PredictionResponse(
        study_id=study_id,
        findings={name: float(p) for name, p in zip(FINDINGS, study_probs)},
        per_plane=per_plane,
        laterality=laterality,
        laterality_source=laterality_source,
        quality_flags=flags,
        model_version=model_version,
        latency_ms=max(float(latency_ms), 0.0),
    )


def read_laterality(series_dirs: list[Path]) -> tuple[Laterality, LateralitySource]:
    # Reads the laterality tag from the first file of each selected series (headers only, no pixels).
    # Series that disagree with each other give "unknown" rather than a guess.
    import pydicom

    found: set[Laterality] = set()
    for folder in series_dirs:
        files = sorted(Path(folder).glob("*.dcm"))
        if not files:
            continue
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        for tag in LATERALITY_TAGS:
            value = str(ds.get(tag, "") or "").strip().upper()
            if value in _LATERALITY_CODES:
                found.add(_LATERALITY_CODES[value])
                break
    if len(found) == 1:
        return found.pop(), "dicom_tag"
    return "unknown", "unavailable"


def resolve_device(name: str = "auto") -> torch.device:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MODEL_DEVICE=cuda but no CUDA GPU is available")
    return torch.device(name)


def model_version_from_meta(meta: dict) -> str:
    # train.py saves knee_<version>.pt with meta["version"] = "v0", "v1", ...  ->  "knee_v1"
    version = str(meta.get("version", "unknown"))
    return version if version.startswith("knee") else f"knee_{version}"


class KneePredictor:
    """Loads a checkpoint once and scores studies with it."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        from model_service.knee_model.model import load_checkpoint
        from model_service.knee_model.preprocess import IMAGE_SIZE, N_SLICES

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.device = resolve_device(device)
        self.model, self.meta = load_checkpoint(checkpoint, device=self.device)
        self.model_version = model_version_from_meta(self.meta)
        # The checkpoint records the input shape it was trained on; use it so live input matches training input.
        self.n_slices = int(self.meta.get("n_slices", N_SLICES))
        self.image_size = int(self.meta.get("image_size", IMAGE_SIZE))

    def predict_array(self, array: np.ndarray, plane_present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # array: (3, n_slices, size, size) in 0-1. Returns study probs (12,) and plane probs (3, 12).
        import torch

        expected = (len(PLANES), self.n_slices, self.image_size, self.image_size)
        if tuple(array.shape) != expected:
            raise ValueError(f"input shape {tuple(array.shape)} does not match the checkpoint's {expected}")
        x = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(np.asarray(plane_present, dtype=bool)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            study_logits, plane_logits = self.model(x, mask)
        study = torch.sigmoid(study_logits.float())[0].cpu().numpy()
        planes = torch.sigmoid(plane_logits.float())[0].cpu().numpy()
        return study, planes

    def predict_study(
        self,
        study_id: str,
        series_table: pd.DataFrame,
        dataset_root: str | Path,
        split_folder: str = "train_series",
    ) -> PredictionResponse:
        # Full path from DICOM folders: select series, load, preprocess, score.
        from model_service.knee_model.pipeline import load_study
        from model_service.knee_model.series_selection import series_path

        start = time.perf_counter()
        study = load_study(study_id, series_table, dataset_root, split_folder=split_folder,
                           n_slices=self.n_slices, size=self.image_size)
        plane_present = np.array([f"missing_plane:{p}" not in study.quality_flags for p in PLANES])
        study_probs, plane_probs = self.predict_array(study.array, plane_present)
        dirs = [series_path(dataset_root, study_id, chosen.series_id, split_folder)
                for chosen in study.selection.values() if chosen.series_id is not None]
        laterality, source = read_laterality(dirs)
        return build_prediction(
            study_id, study_probs, plane_probs, plane_present,
            quality_flags=study.quality_flags, laterality=laterality, laterality_source=source,
            model_version=self.model_version, latency_ms=(time.perf_counter() - start) * 1000,
        )

    def predict_cached(
        self,
        study_id: str,
        cache_dir: str | Path,
        series_table: pd.DataFrame | None = None,
        dataset_root: str | Path | None = None,
        split_folder: str = "train_series",
    ) -> PredictionResponse:
        # Fast path on Kaggle: score the .npz block dataset.py already cached for training.
        # Laterality needs the DICOM headers, so it is only read when series_table and dataset_root are given.
        from model_service.knee_model.dataset import load_cached

        start = time.perf_counter()
        array, plane_present, flags = load_cached(cache_dir, study_id)
        study_probs, plane_probs = self.predict_array(array, plane_present)
        laterality: Laterality = "unknown"
        source: LateralitySource = "unavailable"
        if series_table is not None and dataset_root is not None:
            from model_service.knee_model.series_selection import (
                select_series,
                series_path,
            )

            dirs = [series_path(dataset_root, study_id, chosen.series_id, split_folder)
                    for chosen in select_series(study_id, series_table).values() if chosen.series_id is not None]
            laterality, source = read_laterality(dirs)
        return build_prediction(
            study_id, study_probs, plane_probs, plane_present,
            quality_flags=flags, laterality=laterality, laterality_source=source,
            model_version=self.model_version, latency_ms=(time.perf_counter() - start) * 1000,
        )