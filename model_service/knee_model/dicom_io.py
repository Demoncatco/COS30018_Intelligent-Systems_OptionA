# Reads one MRI series (a folder of .dcm files) and turns it into a single, correctly ordered 3D block of numbers.
#
# A DICOM file is one 2D image plus a few hundred metadata fields. Three of those fields must be honoured
# or the pixels are wrong, and one of them fails silently if ignored:
#
#   1. RescaleSlope / RescaleIntercept  - real = stored * slope + intercept. Puts every slice on the same brightness scale.
#   2. PhotometricInterpretation        - MONOCHROME1 means low values are WHITE (a photo negative). Must be inverted.
#   3. ImagePositionPatient             - where each slice sits in space. Files are NOT in anatomical order on disk.
#
# Usage:
#     from model_service.knee_model.dicom_io import load_series
#     series = load_series("/kaggle/input/.../train_series/<study>/<series>")
#     series.volume.shape   -> (n_slices, height, width), float32, slices in anatomical order
#
# Step 3 of the Role 1 plan. Findings F3, F4 and F5 from Step 1 are implemented here and marked below.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class LoadedSeries:
    """Everything load_series() learned about one series. The volume plus the metadata that travels with it."""

    volume: np.ndarray                  # (n_slices, height, width) float32, rescaled, inverted if needed, anatomically ordered
    paths: list[Path]                   # the file each slice came from, in the same order as volume
    instance_numbers: np.ndarray        # InstanceNumber per slice, in volume order. For cross-checking only.
    slice_positions: np.ndarray         # each slice's distance along the slice normal, in mm, in volume order
    slice_normal: np.ndarray            # the (x, y, z) direction the stack was sorted along
    slice_spacing_mm: float             # median gap between consecutive slices
    rescale_slope_constant: bool        # True if every slice had the same RescaleSlope (F3)
    ordering_agrees: bool               # True if InstanceNumber order matches position order, forwards or backwards (F4)
    photometric: str                    # MONOCHROME1 or MONOCHROME2, as found on disk
    series_description: str             # e.g. "DP SPIR CS_SAG"
    warnings: list[str] = field(default_factory=list)   # anything odd, so a batch run can log it rather than crash

    @property
    def n_slices(self) -> int:
        return self.volume.shape[0]


def _slice_normal(ds: pydicom.Dataset) -> np.ndarray:
    # F5. ImageOrientationPatient holds two unit vectors: the direction of a row of pixels and the direction of a
    # column of pixels, in patient space. Their cross product points straight out of the image, i.e. along the
    # direction slices are stacked. Sorting by distance along that direction is correct for any tilt of the scan.
    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    return normal / np.linalg.norm(normal)


def _to_real_values(ds: pydicom.Dataset) -> np.ndarray:
    # F3. Apply the scanner's stored-to-real mapping. .get() with defaults means a vendor that omits the tags
    # gets a harmless "multiply by 1, add 0" rather than a crash.
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    px = ds.pixel_array.astype(np.float32)
    return px * slope + intercept


def load_series(series_dir: str | Path) -> LoadedSeries:
    """Load every .dcm in a folder, fix the three metadata issues, and return the ordered volume with its metadata."""
    series_dir = Path(series_dir)
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")

    warnings: list[str] = []
    datasets = [pydicom.dcmread(p) for p in paths]

    # --- Step A: work out the anatomical order --------------------------------------------------------------
    # Every slice in a series should share the same orientation, so take the normal from the first one.
    first = datasets[0]
    if "ImageOrientationPatient" in first and "ImagePositionPatient" in first:
        normal = _slice_normal(first)
        # Distance of each slice along the normal. This one number per slice is what we sort by.
        positions = np.array([float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=np.float64), normal)) for ds in datasets])
    else:
        # Fallback for files with no geometry at all. Should never happen on this dataset; logged loudly if it does.
        warnings.append("no ImageOrientationPatient/ImagePositionPatient; fell back to InstanceNumber ordering")
        normal = np.array([np.nan, np.nan, np.nan])
        positions = np.array([float(ds.get("InstanceNumber", i)) for i, ds in enumerate(datasets)])

    order = np.argsort(positions, kind="stable")
    datasets = [datasets[i] for i in order]
    paths = [paths[i] for i in order]
    positions = positions[order]

    # --- Step B: F4 cross-check. InstanceNumber is the scanner's own slice counter. ------------------------
    # If it agrees with the geometric order (in either direction) we have two independent sources saying the same
    # thing. If it does not, something is odd about this series and it must be looked at by a human.
    instance_numbers = np.array([int(ds.get("InstanceNumber", -1)) for ds in datasets])
    diffs = np.diff(instance_numbers)
    ordering_agrees = bool(len(diffs) == 0 or np.all(diffs > 0) or np.all(diffs < 0))
    if not ordering_agrees:
        warnings.append(f"InstanceNumber order disagrees with ImagePositionPatient order: {instance_numbers.tolist()}")

    # --- Step C: F3 check. Is RescaleSlope the same on every slice? ------------------------------------------
    # Philips can write a different slope per slice. If that happens, skipping the rescale would leave slices on
    # different brightness scales inside one volume. We apply it regardless; this flag just tells us if it mattered.
    slopes = [float(ds.get("RescaleSlope", 1.0)) for ds in datasets]
    rescale_slope_constant = bool(np.allclose(slopes, slopes[0]))
    if not rescale_slope_constant:
        warnings.append(f"RescaleSlope varies within series: min={min(slopes)} max={max(slopes)}")

    # --- Step D: pixels -> real values, then stack -----------------------------------------------------------
    slices = [_to_real_values(ds) for ds in datasets]
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"slices in {series_dir} have different pixel shapes: {sorted(shapes)}")
    volume = np.stack(slices).astype(np.float32)

    # --- Step E: MONOCHROME1 inversion --------------------------------------------------------------------------
    # MONOCHROME1 = smallest value is white. Flip it so that, like MONOCHROME2, bright means bright.
    # Inverting as (max - value) keeps the values in the same range rather than making them negative.
    photometric = str(first.get("PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        volume = volume.max() - volume
        warnings.append("MONOCHROME1 series inverted")

    # --- Step F: slice spacing, for the Step 5 sanity check that sampling evenly by index is even in mm ----------
    spacing = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0

    return LoadedSeries(
        volume=volume,
        paths=paths,
        instance_numbers=instance_numbers,
        slice_positions=positions,
        slice_normal=normal,
        slice_spacing_mm=spacing,
        rescale_slope_constant=rescale_slope_constant,
        ordering_agrees=ordering_agrees,
        photometric=photometric,
        series_description=str(first.get("SeriesDescription", "")),
        warnings=warnings,
    )
