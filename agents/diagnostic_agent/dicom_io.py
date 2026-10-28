# This file reads one MRI series, which is a folder of .dcm files, and turns it into a single, correctly ordered 3D block of numbers.
# A DICOM file is one 2D image, plus a few hundred metadata header fields. Three of these fields affect how pixels are represented:
# 1. RescaleSlope / RescaleIntercept - real = stored * slope + intercept. This puts every slice on the same brightness scale
# 2. PhotometricInterpretation - MONOCHROME1 means low values are White (a photo negative), this must be inverted.
# 3. ImagePositionPatient - Where each slice sits in space. Note that files are not in anatomical order in each folder. Order must be worked out

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class LoadedSeries:
    # Everything load_series() learned about one series.

    volume: np.ndarray # The actual 3D Image, shape (slices, height, width) in float32 decimal type
    paths: list[Path] # The list of files in the same order as the slices, so that any slice can be traced back to its file.
    instance_numbers: np.ndarray # Scanner's own slice counter per slice. This is kept for cross-checking order of slices
    slice_positions: np.ndarray  # Each slice's distance in millimeters along the stacking direction
    slice_normal: np.ndarray # The stacking direction as an (x, y or z) arrow
    slice_spacing_mm: float  # The typical gap between slices in millimeters
    rescale_slope_constant: bool # True or False on whether the Rescale Slope is the same for all slices
    ordering_agrees: bool # True if InstanceNumber order matches position order
    photometric: str # MONOCHROME1 or MONOCHROME2
    series_description: str # Doctor Description of that series, for example: "DP SPIR CS_SAG"
    warnings: list[str] = field(default_factory=list) # anything odd, so a batch run can log it rather than crash

    @property
    def n_slices(self) -> int:
        return self.volume.shape[0]


def _slice_normal(ds: pydicom.Dataset) -> np.ndarray:
    # ImageOrientationPatient holds six numbers, two arrows of three numbers each.
    # The first arrow is the direction a row of pixels runs through the patient's body, whilst the second is the direction a column runs.
    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64) # Turns these arrows into arrays
    row_dir = iop[:3] # iop[:3] is the first three numbers
    col_dir = iop[3:] # iop[3:] is the last three numbers
    # The cross product of two arrows is a third arrow at right angles to both. A slice is flat along its rows and columns,
    # so the arrow at right angles to both is the direction the slices are stacked in.
    normal = np.cross(row_dir, col_dir)
    # Dividing by its own length makes the arrow exactly 1 long, so distances measured along it come out in millimeters.
    return normal / np.linalg.norm(normal)


def _slice_positions(datasets: list[pydicom.Dataset], normal: np.ndarray) -> np.ndarray:
    # Gives every slice one number: how far along the stacking direction it sits, in millimeters. The slices are sorted by this number.
    positions = []
    for ds in datasets:
        # ImagePositionPatient is the (x, y, z) point in the room where the first pixel of the slice sits.
        corner = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        # The dot product measures how far that point is along the normal. A scan is usually tilted a few degrees,
        # so using x, y or z on its own could put slices in the wrong order. This works at any tilt.
        positions.append(float(np.dot(corner, normal)))
    return np.array(positions)


def _positions_from_instance_numbers(datasets: list[pydicom.Dataset]) -> np.ndarray:
    # Fallback for files with no geometry at all. Should never happen on this dataset though.
    # The scanner's own slice counter is used as the position
    positions = []
    for i, ds in enumerate(datasets):
        positions.append(float(ds.get("InstanceNumber", i)))
    return np.array(positions)


def _instance_numbers_agree(instance_numbers: np.ndarray) -> bool:
    # instance_numbers is the scanner's slice counter, already in the sorted order.
    # np.diff gives the step from each number to the next one, so [3, 2, 1] gives [-1, -1].
    diffs = np.diff(instance_numbers)

    # A series of one slice has no steps, so there is nothing that could disagree.
    if len(diffs) == 0:
        return True

    # Some scanners count in the opposite direction to the position axis. That is consistent, not wrong,
    # so the order agrees if the counter goes up at every step or goes down at every step.
    counts_up = bool(np.all(diffs > 0))
    counts_down = bool(np.all(diffs < 0))
    return counts_up or counts_down


def _to_real_values(ds: pydicom.Dataset) -> np.ndarray:
    # Scanner stores pixel values compactly, meaning they need to be converted to get the real pixels.
    # real = stored * slope + intercept
    slope = float(ds.get("RescaleSlope", 1.0)) # A file without the tag gets slope 1, which changes nothing
    intercept = float(ds.get("RescaleIntercept", 0.0)) # A file without the tag gets intercept 0, which changes nothing
    px = ds.pixel_array.astype(np.float32)
    return px * slope + intercept


def _stack_slices(datasets: list[pydicom.Dataset], series_dir: Path) -> np.ndarray:
    # Converts every slice to real values, then stacks them into one block of shape (slices, height, width).
    # datasets must already be in anatomical order, because the block keeps the order it is given.
    slices = []
    shapes = set() # A set keeps one copy of each different value, so it ends up holding every different slice size
    for ds in datasets:
        pixels = _to_real_values(ds)
        slices.append(pixels)
        shapes.add(pixels.shape)

    # If the slices are different sizes, then they cannot be stacked into a rectangular block. There is no way to continue, so an error is raised.
    if len(shapes) != 1:
        raise ValueError(f"slices in {series_dir} have different pixel shapes: {sorted(shapes)}")

    return np.stack(slices).astype(np.float32)


def _slice_spacing(positions: np.ndarray) -> float:
    # Provides the typical gap between neighbouring slices, in millimeters. positions must already be sorted.
    # One slice has no neighbour, so there is no gap to measure.
    if len(positions) < 2:
        return 0.0
    gaps = np.abs(np.diff(positions))
    # The median is the middle gap. It is used so one odd gap does not change the answer.
    return float(np.median(gaps))


def load_series(series_dir: str | Path) -> LoadedSeries:
    # Loads every .dcm file in a folder, fixes the three metadata issues and returns the ordered volume, with its metadata.

    # Accepts the folder as a file path and then finds every file ending in .dcm. sorting makes the starting order predictable, it is not the actual order.
    series_dir = Path(series_dir)
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")

    warnings = []
    datasets = [pydicom.dcmread(p) for p in paths]

    # The first file is used to read the things that are the same for the whole series: the orientation, the photometric tag and the description.
    first = datasets[0]

    # Works out where every slice sits. Checks that the orientation and position are in the first dataset, as the orientation is required for the normal and the position for the sort key.
    if "ImageOrientationPatient" in first and "ImagePositionPatient" in first:
        normal = _slice_normal(first)
        positions = _slice_positions(datasets, normal)
    else:
        warnings.append("no ImageOrientationPatient/ImagePositionPatient; fell back to InstanceNumber ordering")
        normal = np.array([np.nan, np.nan, np.nan])
        positions = _positions_from_instance_numbers(datasets)

    # Puts the slices in anatomical order. argsort does not move anything, it returns the order to pick the slices in: the lowest position first.
    # kind="stable" means two slices at the same position keep the order they already had, so the result is the same on every run.
    order = np.argsort(positions, kind="stable")
    # Reordering. Builds a new list by picking datasets in the correct sorted order. The file paths and positions are reordered the same way so all three stay lined up.
    datasets = [datasets[i] for i in order]
    paths = [paths[i] for i in order]
    positions = positions[order]

    # The below gets the InstanceNumber order which is the scanner's own slice counter.
    # This is another way of checking the order, comparing it against the geometric order obtained above.
    # If they agree then we have two independent sources saying the same order which we can proceed with.
    # If they do not match, then something has not been calculated right and it requires human intervention.
    instance_numbers = np.array([int(ds.get("InstanceNumber", -1)) for ds in datasets]) # Reads the scanner's slice counter from each file in new order.
    ordering_agrees = _instance_numbers_agree(instance_numbers)
    # If the ordering does not agree then a warning is provided that includes the actual sequence
    if not ordering_agrees:
        warnings.append(f"InstanceNumber order disagrees with ImagePositionPatient order: {instance_numbers.tolist()}")

    # The scanner can write a different slope per slice, affecting brightness scales on the slice.
    slopes = [float(ds.get("RescaleSlope", 1.0)) for ds in datasets]
    rescale_slope_constant = bool(np.allclose(slopes, slopes[0])) # True if every slope is equal to the first one
    # If not equal, then a warning is provided that says the slope differs, from minimum value to maximum value. Rescale is still applied anyway, each slice with its own slope.
    if not rescale_slope_constant:
        warnings.append(f"RescaleSlope varies within series: min={min(slopes)} max={max(slopes)}")

    # Turns every slice into real values and stacks them. After this the volume is one block of shape (slices, height, width) on the real brightness scale.
    volume = _stack_slices(datasets, series_dir)

    # MONOCHROME1 means that the smallest values are white, and the highest values are black, this is inverted to normal. We need to flip this
    # Inverting as (max - value) keeps the values in the same range, avoiding making them negative.
    photometric = str(first.get("PhotometricInterpretation", "MONOCHROME2")) # Reads the tag that says which way the brightness is.
    # Default is MONOCHROME2. If MONOCHROME1, then it needs to invert as below:
    if photometric == "MONOCHROME1":
        volume = volume.max() - volume
        warnings.append("MONOCHROME1 series inverted")

    # Provides the gaps between each slice.
    spacing = _slice_spacing(positions)

    # Returns a loadedseries which has all these details.
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
