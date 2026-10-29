# Turns a user's uploaded knee MRI (loose .dcm files or a .zip of the study) into the same (3, 20, 224, 224) block as load_study().
# Kaggle studies get their plane and fluid-sensitive flag from train_series.csv; uploads have no such table, so both are read from the DICOM headers.

from __future__ import annotations

import io
import json
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydantic import BaseModel, Field

from model_service.knee_model.dicom_io import load_series
from model_service.knee_model.preprocess import IMAGE_SIZE, N_SLICES, preprocess_series
from model_service.knee_model.series_selection import (
    LEVEL1_FLUID_SENSITIVE,
    LEVEL2_ANY_SERIES,
    LEVEL3_PLANE_MISSING,
)
from shared.schema import PLANES, Laterality, LateralitySource

UPLOAD_ID = re.compile(r"^upl-[0-9a-f]{12}$")  # also the study_id sessions use, so it must pass SessionCreate's pattern
STUDIES_FOLDER = "studies"  # uploads live in <UPLOAD_DIR>/studies/<upload_id>/<series folder>/*.dcm
SUMMARY_FILE = "upload.json"
ARRAY_FILE = "model_input.npz"  # same keys as the Kaggle training cache (array, plane_present, quality_flags)
MIN_SERIES_SLICES = 5  # smaller groups are scouts/localisers, not diagnostic series
LOW_SLICE_COUNT_THRESHOLD = 15  # same rule as pipeline.py

# Words in SeriesDescription / ScanOptions that mean fat-suppressed, i.e. fluid-sensitive (training used the same idea).
_FLUID_TOKENS = {"FS", "FATSAT", "SPIR", "SPAIR", "STIR", "TIRM", "PDFS", "T2FS", "PDWFS", "IRFS", "SFS", "FSAT"}
_PLANE_TOKENS = {"SAG": "sagittal", "SAGITTAL": "sagittal", "COR": "coronal", "CORONAL": "coronal",
                 "AX": "axial", "AXIAL": "axial", "TRA": "axial", "TRANS": "axial", "TRANSVERSE": "axial"}


class UploadError(ValueError):
    """The upload cannot become a study; the message is shown to the user as-is."""


class UploadedSeries(BaseModel):
    folder: str
    description: str = ""
    plane: str  # sagittal / coronal / axial, or "unknown" when the orientation could not be read
    fluid_sensitive: bool
    n_slices: int
    used: bool = False


class UploadedPlane(BaseModel):
    plane: str
    present: bool
    series_folder: str | None = None
    description: str = ""
    level: int = LEVEL3_PLANE_MISSING
    n_original_slices: int = 0
    original_shape: tuple[int, int] = (0, 0)


class UploadSummary(BaseModel):
    upload_id: str
    created_at: float = Field(default_factory=time.time)
    n_files: int
    n_images: int
    n_skipped: int
    skipped_examples: list[str] = Field(default_factory=list)
    series: list[UploadedSeries]
    planes: list[UploadedPlane]
    quality_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    laterality: Laterality = "unknown"
    laterality_source: LateralitySource = "unavailable"
    n_slices: int = N_SLICES
    image_size: int = IMAGE_SIZE


@dataclass
class _Image:
    data: bytes
    series_uid: str
    plane: str
    description: str
    fluid: bool


def new_upload_id() -> str:
    return f"upl-{uuid.uuid4().hex[:12]}"


def is_upload_id(study_id: str) -> bool:
    return bool(UPLOAD_ID.match(study_id))


def study_dir(upload_root: str | Path, upload_id: str) -> Path:
    if not is_upload_id(upload_id):
        raise UploadError(f"not an upload id: {upload_id!r}")
    return Path(upload_root) / STUDIES_FOLDER / upload_id


# ------------------------------------------------------------
# Reading the files
# ------------------------------------------------------------

def expand_files(files: list[tuple[str, bytes]], max_bytes: int, max_files: int) -> list[tuple[str, bytes]]:
    # Unpacks any .zip into its members. Member names are never used as paths, so zip-slip is not possible.
    out: list[tuple[str, bytes]] = []
    total = 0
    for name, data in files:
        if name.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
            try:
                archive = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile as error:
                raise UploadError(f"{name} is not a readable .zip file") from error
            for member in archive.infolist():
                if member.is_dir() or member.filename.startswith("__MACOSX/"):
                    continue
                total += member.file_size
                if total > max_bytes:
                    raise UploadError(f"the unzipped study is larger than {max_bytes // 1_000_000} MB")
                out.append((member.filename, archive.read(member)))
        else:
            total += len(data)
            if total > max_bytes:
                raise UploadError(f"the upload is larger than {max_bytes // 1_000_000} MB")
            out.append((name, data))
        if len(out) > max_files:
            raise UploadError(f"the upload has more than {max_files} files")
    return out


def plane_from_orientation(iop) -> str | None:
    # The slice normal (row x column) points left-right for sagittal, front-back for coronal, head-foot for axial.
    try:
        values = np.asarray([float(v) for v in iop], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.shape != (6,):
        return None
    normal = np.cross(values[:3], values[3:])
    if not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        return None
    return PLANES[int(np.argmax(np.abs(normal)))]


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", text.upper()) if t]


def plane_from_description(description: str) -> str | None:
    for token in _tokens(description):
        if token in _PLANE_TOKENS:
            return _PLANE_TOKENS[token]
    return None


def is_fluid_sensitive(description: str, scan_options: str = "") -> bool:
    text = f"{description} {scan_options}".upper()
    if "FAT SAT" in text or "FAT-SAT" in text or "FAT SUP" in text:
        return True
    return any(token in _FLUID_TOKENS for token in _tokens(text))


def _read_image(name: str, data: bytes) -> tuple[_Image | None, str]:
    # Returns (image, "") or (None, reason it was skipped).
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=True, stop_before_pixels=False)
    except Exception:  # noqa: BLE001 - anything unreadable is simply not a DICOM image
        return None, f"{name}: not a DICOM file"
    if "PixelData" not in ds or "Rows" not in ds:
        return None, f"{name}: no image data"
    image_type = [str(v).upper() for v in (ds.get("ImageType") or [])]
    if "LOCALIZER" in image_type:
        return None, f"{name}: localiser image"
    if int(ds.get("SamplesPerPixel", 1) or 1) != 1:
        return None, f"{name}: colour image, not an MRI slice"
    description = str(ds.get("SeriesDescription", "") or "")
    plane = plane_from_orientation(ds.get("ImageOrientationPatient")) or plane_from_description(description) or "unknown"
    series_uid = str(ds.get("SeriesInstanceUID", "") or "no-series-uid")
    scan_options = ds.get("ScanOptions", "") or ""
    if not isinstance(scan_options, str):  # multi-valued tag
        scan_options = " ".join(map(str, scan_options))
    return _Image(data, series_uid, plane, description, is_fluid_sensitive(description, scan_options)), ""


# ------------------------------------------------------------
# Building the study
# ------------------------------------------------------------

def _write_series(images: list[_Image], target: Path) -> list[UploadedSeries]:
    # One folder per (series, plane); a series whose slices point different ways is split, not mixed.
    groups: dict[tuple[str, str], list[_Image]] = {}
    for image in images:
        groups.setdefault((image.series_uid, image.plane), []).append(image)
    ordered = sorted(groups.items(), key=lambda kv: (PLANES.index(kv[0][1]) if kv[0][1] in PLANES else 9, kv[0][0]))
    series = []
    for n, ((_, plane), members) in enumerate(ordered, start=1):
        folder = f"s{n:02d}-{plane}"
        path = target / folder
        path.mkdir(parents=True)
        for i, image in enumerate(members):
            (path / f"{i:04d}.dcm").write_bytes(image.data)
        series.append(UploadedSeries(folder=folder, description=members[0].description, plane=plane,
                                     fluid_sensitive=members[0].fluid, n_slices=len(members)))
    return series


def _candidates(series: list[UploadedSeries], plane: str) -> list[tuple[int, UploadedSeries]]:
    # Same ladder as series_selection.py: fluid-sensitive first, then any series. More slices wins a tie.
    usable = [s for s in series if s.plane == plane and s.n_slices >= MIN_SERIES_SLICES]
    level1 = sorted((s for s in usable if s.fluid_sensitive), key=lambda s: (-s.n_slices, s.folder))
    level2 = sorted((s for s in usable if not s.fluid_sensitive), key=lambda s: (-s.n_slices, s.folder))
    return [(LEVEL1_FLUID_SENSITIVE, s) for s in level1] + [(LEVEL2_ANY_SERIES, s) for s in level2]


def build_array(target: Path, planes: list[UploadedPlane], n_slices: int = N_SLICES,
                size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # (3, n_slices, size, size) in 0-1, the plane mask and warnings. Used at upload time and again if a checkpoint wants another size.
    blocks, present, warnings = [], [], []
    for plane in planes:
        if not plane.present or plane.series_folder is None:
            blocks.append(np.zeros((n_slices, size, size), dtype=np.float32))
            present.append(False)
            continue
        loaded = load_series(target / plane.series_folder)
        processed = preprocess_series(loaded.volume, n_slices=n_slices, size=size)
        blocks.append(processed.array)
        present.append(True)
        warnings += [f"{plane.plane}: {w}" for w in loaded.warnings + processed.warnings]
    return np.stack(blocks), np.array(present, dtype=bool), warnings


def ingest_upload(files: list[tuple[str, bytes]], upload_root: str | Path, *, max_bytes: int = 1_000_000_000,
                  max_files: int = 5000) -> UploadSummary:
    # The whole upload step: unpack, sort into series, pick one per plane, preprocess, save. Raises UploadError for the user.
    from model_service.knee_model.inference import read_laterality

    if not files:
        raise UploadError("no files were uploaded")
    expanded = expand_files(files, max_bytes, max_files)
    images, skipped = [], []
    for name, data in expanded:
        image, reason = _read_image(name, data)
        if image is None:
            skipped.append(reason)
        else:
            images.append(image)
    if not images:
        raise UploadError("No MRI images found. Upload the study's DICOM (.dcm) files, the study folder, or a .zip of it. "
                          "Photos and screenshots (JPG, PNG) can't be analysed: the model needs the full MRI series.")

    upload_id = new_upload_id()
    target = study_dir(upload_root, upload_id)
    target.mkdir(parents=True)
    try:
        series = _write_series(images, target)
        planes, quality_flags, warnings = _select_and_load(target, series)
        if not any(p.present for p in planes):
            raise UploadError("None of the images could be used: no series had enough slices in a sagittal, coronal "
                              f"or axial plane (at least {MIN_SERIES_SLICES} needed).")
        array, present, load_warnings = build_array(target, planes)
        warnings += load_warnings
        laterality, source = read_laterality([target / p.series_folder for p in planes if p.series_folder])
        np.savez_compressed(target / ARRAY_FILE, array=array.astype(np.float16), plane_present=present,
                            quality_flags=np.array(json.dumps(quality_flags)))
        summary = UploadSummary(upload_id=upload_id, n_files=len(expanded), n_images=len(images),
                                n_skipped=len(skipped), skipped_examples=skipped[:5], series=series, planes=planes,
                                quality_flags=quality_flags, warnings=warnings, laterality=laterality,
                                laterality_source=source)
        (target / SUMMARY_FILE).write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        series_table(summary).to_csv(target / "series.csv", index=False)
        return summary
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _select_and_load(target: Path, series: list[UploadedSeries]) -> tuple[list[UploadedPlane], list[str], list[str]]:
    # Tries each plane's candidates in ladder order until one loads, so one broken series does not sink the plane.
    planes, flags, warnings = [], [], []
    for plane in PLANES:
        chosen = None
        for level, candidate in _candidates(series, plane):
            try:
                loaded = load_series(target / candidate.folder)
            except Exception as error:  # noqa: BLE001
                warnings.append(f"{plane}: skipped series '{candidate.description or candidate.folder}' ({error})")
                continue
            candidate.used = True
            chosen = UploadedPlane(plane=plane, present=True, series_folder=candidate.folder,
                                   description=candidate.description, level=level, n_original_slices=loaded.n_slices,
                                   original_shape=(int(loaded.volume.shape[1]), int(loaded.volume.shape[2])))
            break
        if chosen is None:
            chosen = UploadedPlane(plane=plane, present=False)
            flags.append(f"missing_plane:{plane}")
        elif chosen.n_original_slices < LOW_SLICE_COUNT_THRESHOLD and "low_slice_count" not in flags:
            flags.append("low_slice_count")
        planes.append(chosen)
    return planes, flags, warnings


def series_table(summary: UploadSummary) -> pd.DataFrame:
    # train_series.csv-shaped table of the upload, handy for debugging with the Kaggle tools.
    return pd.DataFrame([{"StudyInstanceUID": summary.upload_id, "SeriesInstanceUID": s.folder,
                          "Anatomical_Plane": s.plane.capitalize(), "Fluid_Sensitive": int(s.fluid_sensitive),
                          "Fat_Suppression": int(s.fluid_sensitive), "SeriesDescription": s.description,
                          "n_slices": s.n_slices} for s in summary.series])


# ------------------------------------------------------------
# Reading a saved upload (backend previews, Model Service scoring)
# ------------------------------------------------------------

def load_summary(upload_root: str | Path, upload_id: str) -> UploadSummary:
    path = study_dir(upload_root, upload_id) / SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError(upload_id)
    return UploadSummary.model_validate_json(path.read_text(encoding="utf-8"))


def load_model_input(upload_root: str | Path, upload_id: str, n_slices: int = N_SLICES,
                     size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Saved block if its shape fits, else rebuilt from the stored DICOMs at the requested size.
    target = study_dir(upload_root, upload_id)
    with np.load(target / ARRAY_FILE) as data:
        array = data["array"].astype(np.float32)
        present = data["plane_present"].astype(bool)
        flags = json.loads(str(data["quality_flags"]))
    if array.shape[1:] != (n_slices, size, size):
        array, present, _ = build_array(target, load_summary(upload_root, upload_id).planes, n_slices, size)
    return array, present, flags


def delete_upload(upload_root: str | Path, upload_id: str) -> bool:
    target = study_dir(upload_root, upload_id)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True