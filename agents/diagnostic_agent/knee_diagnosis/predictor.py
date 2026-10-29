"""
Uploaded DICOM files -> 12 abnormality probabilities.

An upload is just a pile of .dcm files (often several series from one study, usually without
train_series.csv). So this module does what the training notebook got from the CSVs:
  1. read the headers and group files into series (SeriesInstanceUID)
  2. work out each series' imaging plane from ImageOrientationPatient
  3. pick one series per plane (prefer fat-suppressed, then the one with most slices)
  4. load each with dicom_io.load_series, preprocess exactly as in training, run the model
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import torch

from .config import ABNORMALITIES, PLANES
from .dicom_io import load_series
from .model import KneeNet
from .preprocess import preprocess_volume

log = logging.getLogger(__name__)

# Training used the competition's Fluid_Sensitive flag (identical to Fat_Suppression there). Uploads have no such
# flag, so it is guessed from the series description. "FS" must not match "FSE" (fast spin echo), hence the lookarounds.
_FAT_SUPPRESSED = re.compile(
    r"(?<![a-z])(fs|spir|spair|stir|tirm|fatsat|fat[ _-]?sat\w*|fat[ _-]?suppress\w*)(?![a-z])", re.I
)
_HEADER_TAGS = ["SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription", "ScanOptions",
                "ImageOrientationPatient", "ImagePositionPatient", "Modality"]
_MAX_FILE_BYTES = 200 * 1024 * 1024


@dataclass
class SeriesInfo:
    uid: str
    study_uid: str
    description: str
    plane: str
    fat_suppressed: bool
    paths: list[Path] = field(default_factory=list)

    @property
    def n_slices(self) -> int:
        return len(self.paths)


def plane_from_orientation(iop) -> str:
    """Slices stack along the normal of the image plane: x -> sagittal, y -> coronal, z -> axial (patient coords)."""
    iop = np.asarray(iop, dtype=float)
    normal = np.abs(np.cross(iop[:3], iop[3:]))
    return PLANES[int(np.argmax(normal))]   # PLANES = [sagittal, coronal, axial] = [x, y, z]


def _expand(inputs) -> list[Path]:
    """Accepts a file, a folder, or a list of either."""
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file()]
        elif p.is_file():
            files.append(p)
    return files


def scan_dicom_files(inputs) -> tuple[list[SeriesInfo], dict]:
    """Group uploaded files into series. Returns (series, stats). Non-DICOM and non-image files are skipped."""
    groups: dict[str, SeriesInfo] = {}
    skipped = 0
    for f in _expand(inputs):
        try:
            if f.stat().st_size > _MAX_FILE_BYTES:
                skipped += 1
                continue
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=_HEADER_TAGS, force=True)
            uid = str(ds.SeriesInstanceUID)
            iop, _pos = ds.ImageOrientationPatient, ds.ImagePositionPatient     # missing -> AttributeError -> skipped
            plane = plane_from_orientation(iop)
        except Exception:
            skipped += 1
            continue
        if uid not in groups:
            desc = str(ds.get("SeriesDescription", ""))
            text = f"{desc} {ds.get('ScanOptions', '')}"
            groups[uid] = SeriesInfo(uid, str(ds.get("StudyInstanceUID", "")), desc, plane,
                                     bool(_FAT_SUPPRESSED.search(text)))
        groups[uid].paths.append(f)
    return list(groups.values()), {"files_seen": len(_expand(inputs)), "files_skipped": skipped}


def select_series(series: list[SeriesInfo], min_slices: int) -> tuple[dict[str, SeriesInfo], list[str]]:
    """One series per plane: fat-suppressed first, then most slices, then UID (repeatable). Drops localisers."""
    warnings, by_plane = [], defaultdict(list)
    for s in series:
        if s.n_slices < min_slices:
            warnings.append(f"ignored series '{s.description}' ({s.plane}, {s.n_slices} files): fewer than {min_slices} slices")
            continue
        by_plane[s.plane].append(s)
    chosen = {}
    for plane, cands in by_plane.items():
        chosen[plane] = sorted(cands, key=lambda s: (not s.fat_suppressed, -s.n_slices, s.uid))[0]
        if len(cands) > 1:
            warnings.append(f"{plane}: {len(cands)} series uploaded, used '{chosen[plane].description}'")
    return chosen, warnings


def _load_volume(info: SeriesInfo) -> tuple[np.ndarray, list[str]]:
    """dicom_io.load_series reads a folder of *.dcm, so expose the uploaded files through a temp folder."""
    with tempfile.TemporaryDirectory(prefix="knee_series_") as d:
        for i, p in enumerate(info.paths):
            dst = Path(d) / f"{i:05d}.dcm"
            try:
                os.symlink(p.resolve(), dst)
            except OSError:
                shutil.copy(p, dst)
        loaded = load_series(d)
    return loaded.volume, loaded.warnings


class KneePredictor:
    def __init__(self, model_path, device: str | None = None, depth: int = 32, size: int = 224,
                 threshold: float = 0.5, min_slices: int = 8):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.depth, self.size, self.threshold, self.min_slices = depth, size, threshold, min_slices
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise FileNotFoundError(f"model weights not found: {self.model_path}")
                state = torch.load(self.model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model = KneeNet(pretrained=False)
                model.load_state_dict(state)
                self._model = model.to(self.device).eval()
        return self._model

    def predict(self, inputs) -> dict:
        """
        inputs: uploaded .dcm file paths and/or folders.
        Raises ValueError if no usable series is found. If a plane is missing it is replaced by a blank
        input and listed in `missing_planes` (the model was trained on all three, so confidence is lower).
        """
        series, stats = scan_dicom_files(inputs)
        chosen, warnings = select_series(series, self.min_slices)
        if not chosen:
            raise ValueError(f"no usable DICOM series found ({stats['files_seen']} files, "
                             f"{stats['files_skipped']} not readable as DICOM images, "
                             f"{len(series)} series shorter than {self.min_slices} slices)")

        planes, used = [], {}
        for plane in PLANES:
            info = chosen.get(plane)
            if info is None:
                planes.append(np.zeros((self.depth, self.size, self.size), dtype=np.float32))
                continue
            volume, w = _load_volume(info)
            warnings += [f"{plane}: {x}" for x in w]
            planes.append(preprocess_volume(volume, self.depth, self.size))
            used[plane] = {"series_uid": info.uid, "description": info.description,
                           "n_slices": info.n_slices, "fat_suppressed_guess": info.fat_suppressed}
        missing = [p for p in PLANES if p not in chosen]
        if missing:
            warnings.append(f"missing plane(s): {', '.join(missing)}; predictions are less reliable")

        x = torch.from_numpy(np.stack(planes)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self._get_model()(x))[0].cpu().numpy()

        study_uid = Counter(s.study_uid for s in chosen.values() if s.study_uid).most_common(1)
        return {
            "study_instance_uid": study_uid[0][0] if study_uid else None,
            "probabilities": {a: float(p) for a, p in zip(ABNORMALITIES, probs)},
            "threshold": self.threshold,
            "series_used": used,
            "missing_planes": missing,
            "warnings": warnings,
        }
