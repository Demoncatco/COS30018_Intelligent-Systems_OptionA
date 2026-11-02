"""
LocalKneePredictor: study_id -> PredictionResponse, using the checkpoint trained in the Kaggle notebook.
Takes Kaggle study ids (found via LIVE_* settings) and UI upload ids ("upl-...", read from UPLOAD_DIR).

It reuses the repo's own helpers for everything that is not the network, so a study is found and described the same
way the Model Service does it:
  series_selection.select_series / series_path   which series is used for each plane (from train_series.csv)
  dicom_io.load_series                           ordered 3D volume from a series folder
  inference.build_prediction / read_laterality   the PredictionResponse contract and the DICOM laterality tag
Only the model and its preprocessing (local_model/model.py, preprocess.py) are specific to the notebook checkpoint.

torch is imported when the predictor is created, so nothing else in the agent needs it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents.coordinator.runners import StudyNotAvailableError
from model_service.knee_model.dicom_io import load_series
from model_service.knee_model.inference import build_prediction, read_laterality, resolve_device
from model_service.knee_model.series_selection import LEVEL3_PLANE_MISSING, select_series, series_path
from model_service.knee_model.upload import is_upload_id, load_summary, study_dir
from shared.config import Settings
from shared.schema import PLANES, PredictionResponse

from .model import NotebookKneeNet
from .preprocess import DEPTH, SIZE, preprocess_volume

LOW_SLICE_COUNT_THRESHOLD = 15       # same threshold as model_service/knee_model/pipeline.py
DEFAULT_VERSION = "knee_local"


class LocalKneePredictor:
    def __init__(self, checkpoint: str | Path, settings: Settings, *, device: str | None = None,
                 version: str = DEFAULT_VERSION, depth: int = DEPTH, size: int = SIZE) -> None:
        import torch

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.settings = settings
        self.model_version = version
        self.depth, self.size = depth, size
        self.device = resolve_device(device or settings.model_device)
        self.model = NotebookKneeNet(pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and "head.1.weight" not in state:
            state = state["model"]                       # tolerate {"model": state_dict, ...} wrappers
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._series_table: pd.DataFrame | None = None
        self._lock = threading.Lock()                    # one forward pass at a time on a shared model

    # ------------------------------------------------------------------
    def _table(self) -> pd.DataFrame:
        if self._series_table is None:
            path = Path(self.settings.live_series_csv)
            if not path.is_file():
                # No local Kaggle data: a dataset study id cannot be scored, only uploads can
                raise StudyNotAvailableError(
                    f"dataset study ids need the Kaggle data on this machine (series table not found: {path}, "
                    f"set LIVE_SERIES_CSV); upload the study's DICOM files or .zip instead")
            self._series_table = pd.read_csv(path)
        return self._series_table

    def predict_study(self, study_id: str) -> PredictionResponse:
        """Raises StudyNotAvailableError if the study has no usable DICOM data."""
        start = time.perf_counter()
        if is_upload_id(study_id):
            folders, flags, laterality = self._upload_folders(study_id)
        else:
            folders, flags, laterality = self._dataset_folders(study_id)

        arrays: dict[int, np.ndarray] = {}
        for i, folder in folders.items():
            try:
                loaded = load_series(folder)
            except (FileNotFoundError, ValueError) as error:
                raise StudyNotAvailableError(f"study {study_id}: cannot read {PLANES[i]} series: {error}") from error
            if loaded.n_slices < LOW_SLICE_COUNT_THRESHOLD and "low_slice_count" not in flags:
                flags.append("low_slice_count")
            arrays[i] = preprocess_volume(loaded.volume, self.depth, self.size)
        if not arrays:
            raise StudyNotAvailableError(f"study {study_id} has no usable planes")

        study_probs, plane_probs, present = self._forward(arrays)
        if laterality is None:  # dataset studies read it from the DICOM headers here
            laterality = read_laterality(list(folders.values()))
        return build_prediction(study_id, study_probs, plane_probs, present, quality_flags=flags,
                                laterality=laterality[0], laterality_source=laterality[1],
                                model_version=self.model_version, latency_ms=(time.perf_counter() - start) * 1000)

    # ------------------------------------------------------------------
    def _dataset_folders(self, study_id: str) -> tuple[dict[int, Path], list[str], None]:
        # Kaggle study: series picked from train_series.csv, files under LIVE_DATASET_ROOT
        root, split = self.settings.live_dataset_root, self.settings.live_split_folder
        try:
            selection = select_series(study_id, self._table())
        except KeyError as error:
            raise StudyNotAvailableError(f"study {study_id} is not in the series table") from error
        folders: dict[int, Path] = {}
        for i, plane in enumerate(PLANES):
            chosen = selection[plane]
            if chosen.level == LEVEL3_PLANE_MISSING or chosen.series_id is None:
                continue
            folder = series_path(root, study_id, chosen.series_id, split)
            if not folder.is_dir():
                raise StudyNotAvailableError(f"study {study_id}: series folder not found: {folder}")
            folders[i] = folder
        return folders, [], None

    def _upload_folders(self, study_id: str) -> tuple[dict[int, Path], list[str], tuple[str, str]]:
        # UI upload: series already picked at upload time (upload.json), files under UPLOAD_DIR
        try:
            summary = load_summary(self.settings.upload_dir, study_id)
        except FileNotFoundError as error:
            raise StudyNotAvailableError(f"upload {study_id} not found in {self.settings.upload_dir}") from error
        root = study_dir(self.settings.upload_dir, study_id)
        folders = {i: root / p.series_folder for i, p in enumerate(summary.planes)
                   if p.present and p.series_folder}
        return folders, list(summary.quality_flags), (summary.laterality, summary.laterality_source)

    def _forward(self, arrays: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        present = np.array([i in arrays for i in range(len(PLANES))])
        batch = {i: torch.from_numpy(a).unsqueeze(0).to(self.device) for i, a in arrays.items()}
        with self._lock, torch.no_grad():
            fused, per_plane = self.model.plane_logits(batch)
        study_probs = torch.sigmoid(fused.float())[0].cpu().numpy()
        plane_probs = np.zeros((len(PLANES), study_probs.shape[0]))
        plane_probs[np.flatnonzero(present)] = torch.sigmoid(per_plane.float())[0].cpu().numpy()
        return study_probs, plane_probs, present

    def predict(self, study_id: str) -> PredictionResponse:
        # StudyScorer shape, so the baseline can score with the same model as the agent
        return self.predict_study(study_id)

    def __call__(self, study_id: str, timeout: float | None = None) -> PredictionResponse:
        # Same call shape as ModelServiceClient, so the agent can use either. `timeout` cannot interrupt a local run.
        return self.predict_study(study_id)