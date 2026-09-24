# This file reads one MRI series which is a folder of .dcm files, and turns in into a single, correctly ordered 3D block of numbers.
# A DICOM file in one 2D image, plus a few hundred metadata header fields. Three of these fields affect how pixels are represented:
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
    # ImageOritentationPateient holds six numbers, two arrows fo three numbers each.
    # The first arrow is the direction a row of pixels run through the pateient's body, whilst the second is the direction a column runs
    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64) # Turns these arrows into arrays
    row_dir, col_dir = iop[:3], iop[3:] # iop[:3] is first three numbers, iop[3:] is the last three numbers 
    normal = np.cross(row_dir, col_dir)
    return normal / np.linalg.norm(normal)


def _to_real_values(ds: pydicom.Dataset) -> np.ndarray:
    # Scanner stores pixel values compactly, meaning they need to be converted to get the real pixels.
    # real = stored * slope + intercept
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    px = ds.pixel_array.astype(np.float32)
    return px * slope + intercept


def load_series(series_dir: str | Path) -> LoadedSeries:
    # Loads every .dcm file in a folder, fixes the three metadata issues and returns the ordered volume, with its metadata.
    
    # Accepts the folder as a file path and then finds every file ending in .dcm. sorting makes the starting order predictable, it is not the actual order.
    series_dir = Path(series_dir)
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")
    
    warnings: list[str] = []
    datasets = [pydicom.dcmread(p) for p in paths]


    # The below takes the first slice and check it has both geometry tags. If it does then it computes the normal. The single number slice is what is sorted by
    first = datasets[0]
    
    # Checks that the orientation and position is in the first dataset, as this is required for the normal and the position for the sort key.
    if "ImageOrientationPatient" in first and "ImagePositionPatient" in first:
        normal = _slice_normal(first)
        positions = np.array([float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=np.float64), normal)) for ds in datasets])
    else:
        # Fallback for files with no geometry at all. Should never happen on this dataset though.
        warnings.append("no ImageOrientationPatient/ImagePositionPatient; fell back to InstanceNumber ordering")
        normal = np.array([np.nan, np.nan, np.nan])
        positions = np.array([float(ds.get("InstanceNumber", i)) for i, ds in enumerate(datasets)])

    order = np.argsort(positions, kind="stable")
    # Reordering. Builds a new list by picking datasets in the correct sorted order.
    datasets = [datasets[i] for i in order]
    paths = [paths[i] for i in order]
    positions = positions[order]

    # The below gets the InstanceNumber order which is the scanner's own slice counter.
    # This is another way of checking the order, comparing it against the geometric order obtained above.
    # If they are queal then we have two independent sources saying the same order which we can proceed with.
    # If it does not match, then something has not caculated right and must require human intervention.
    instance_numbers = np.array([int(ds.get("InstanceNumber", -1)) for ds in datasets]) # Reads the scanner's slice counter from each file in new order.
    diffs = np.diff(instance_numbers)
    # Compares every element and provides an array of true or false. True only if every element is truly in the correct order.
    ordering_agrees = bool(len(diffs) == 0 or np.all(diffs > 0) or np.all(diffs < 0)) 
    # If the ordering does not agree then warning is provided that includes the actual sequence
    if not ordering_agrees:
        warnings.append(f"InstanceNumber order disagrees with ImagePositionPatient order: {instance_numbers.tolist()}")

    # The scanner can write a different slope per slice, affecting brightness scales on the slice. 
    slopes = [float(ds.get("RescaleSlope", 1.0)) for ds in datasets]
    rescale_slope_constant = bool(np.allclose(slopes, slopes[0])) # True if every element is equal to eachother
    # If not equal, then provide warning that there is a different in slope, from minimum value to maximum value. Rescale is still applied anyway
    if not rescale_slope_constant:
        warnings.append(f"RescaleSlope varies within series: min={min(slopes)} max={max(slopes)}")

    # --- Step D: pixels -> real values, then stack -----------------------------------------------------------
    # Runs the second helper function on every slice. After it is ran, every slice is a list of 2D float32 arrays on the real brightness scale.
    slices = [_to_real_values(ds) for ds in datasets]
    shapes = {s.shape for s in slices}
    # If the slices are different sizes, then they cannot be stacked into a rectangular block. There is no ability to continue on and will throw an error stopping from continuing this script.
    if len(shapes) != 1:
        raise ValueError(f"slices in {series_dir} have different pixel shapes: {sorted(shapes)}")
    volume = np.stack(slices).astype(np.float32)

    # MONOCHROME1 means that the smallest values are white, and the highest values are black, this is inverted to normal. We need to flip this
    # Inverting as (max - value) keeps the values in the same range, avoiding making them negative.
    photometric = str(first.get("PhotometricInterpretation", "MONOCHROME2")) # Reads the tag that says which way the brightness is.
    # Default is MONOCHROME2. If MONOCHROME1, then it needs to invert as below:
    if photometric == "MONOCHROME1":
        volume = volume.max() - volume
        warnings.append("MONOCHROME1 series inverted")

    # Provides the gaps between each slice.
    spacing = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0

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
