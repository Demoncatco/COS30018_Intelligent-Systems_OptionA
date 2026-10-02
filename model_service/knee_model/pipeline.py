# This file brings everything together
# It takes a study ID and returns the block of numbers that the model reads.
# Firstly it calls series_selection.select_series which picks one series for each plane, using the fallback levels and the tie-break logic.
# Secondly it then calls dicom_io.load_series which reads the series folder into a correctly ordered 3D volume
# Lastly it then turns the volume into a fixed (n_slices, size, size) block with values in 0 - 1 (as they are now normalised) via the function preprocess.preprocess_series
# The three blocks are then stacked into the output contract as shape (3 planes, n_slices, size, size), float32, values are between 0 - 1, with the planes in the order of sagittal, coronal and axial.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model_service.knee_model.dicom_io import load_series
from model_service.knee_model.preprocess import IMAGE_SIZE, N_SLICES, preprocess_series
from model_service.knee_model.series_selection import (
    LEVEL3_PLANE_MISSING,
    SelectedSeries,
    select_series,
    series_path,
)
from shared.schema import PLANES

# This is for the point of providing a warning to the model that at least 6 of the 20 output slices are repeated.
LOW_SLICE_COUNT_THRESHOLD = 15


@dataclass
class LoadedStudy:
    # Everything load_study() produced for one study.

    study_id: str  # The StudyInstanceUID that was asked for
    array: np.ndarray  # The finished block, shape (3, n_slices, size, size), float32, values in 0 - 1
    selection: dict[str, SelectedSeries]  # Which series was used for each plane, the level it came from and how many were tied
    original_slice_counts: dict[str, int]  # How many slices each plane's series had before sampling. 0 for a missing plane
    original_shapes: dict[str, tuple[int, int]]  # The (height, width) in pixels of each plane's slices before padding and resizing. (0, 0) for a missing plane
    quality_flags: list[str] = field(default_factory=list)  # Short flags in the wording shared/schema.py uses: low_slice_count, missing_plane:axial
    warnings: list[str] = field(default_factory=list)  # The full sentences from the loader and the preprocessing, each starting with its plane


def _empty_plane(n_slices: int, size: int) -> np.ndarray:
    # If there is an empty plane, it makes a block that stands in for a plane that is not in the study. Using a block of zeroes creates a black image.
    return np.zeros((n_slices, size, size), dtype=np.float32)


def _original_shape(volume: np.ndarray) -> tuple[int, int]:
    # Reads the height and width a series had before it was resized.
    # The volume is (slices, height, width), with height and width being the pixel size the scanner produced. Once every slice is 224x224 there is no way to know the original size.
    height = int(volume.shape[1])
    width = int(volume.shape[2])
    return (height, width)


def _with_plane_name(plane: str, messages: list[str]) -> list[str]:
    # A loop that puts the planes name in front of the warning
    # For example: a warning "series has 12 slices ..." becomes "sagittal: series has 12 slices..."
    named = []
    for message in messages:
        named.append(f"{plane}: {message}")
    return named

# Picks, loads and preprocesses each plane in turn, then stacks the three into one block.
def load_study(
    study_id: str,
    series_table: pd.DataFrame,
    dataset_root: str | Path,
    split_folder: str = "train_series",
    n_slices: int = N_SLICES,
    size: int = IMAGE_SIZE,
) -> LoadedStudy:
    # Turns one study ID into the model-ready block.
    # series_table is train_series.csv loaded to a table format. dataset_root is the folder that holds train_series on kaggle.
    # This needs to be kaggle ready as the data can only be used on Kaggle, not locally.
    selection = select_series(study_id, series_table)

    # These are filled in one plane at a time by the loop below.
    planes = []  # The finished block of each plane, in the order sagittal, coronal, axial
    original_slice_counts = {}
    original_shapes = {}
    quality_flags = []
    warnings = []

    for plane in PLANES:
        chosen = selection[plane]

        if chosen.level == LEVEL3_PLANE_MISSING:
            # There is no series to load for this plane, so it is filled in with zeros to replicate black image and recorded as missing.
            planes.append(_empty_plane(n_slices, size))
            original_slice_counts[plane] = 0
            original_shapes[plane] = (0, 0)
            quality_flags.append(f"missing_plane:{plane}")
            warnings.append(f"{plane}: no series in this plane. Series has been filled with zeros")
        else:
            # Finds the series folder, then loads the series folder and then preprocesses all slices.
            folder = series_path(dataset_root, study_id, chosen.series_id, split_folder)
            loaded = load_series(folder)
            processed = preprocess_series(loaded.volume, n_slices=n_slices, size=size)

            planes.append(processed.array)
            original_slice_counts[plane] = processed.n_original_slices
            original_shapes[plane] = _original_shape(loaded.volume)

            # The flag goes by how many slices the series really had, not by whether a slice was repeated.
            # It is added once even if more than one plane is that short. The warnings say which planes.
            is_short = processed.n_original_slices < LOW_SLICE_COUNT_THRESHOLD
            already_flagged = "low_slice_count" in quality_flags
            if is_short and not already_flagged:
                quality_flags.append("low_slice_count")

            # The warnings from the loader come first, then the ones from the preprocessing so that it replicates order of data flow.
            warnings.extend(_with_plane_name(plane, loaded.warnings + processed.warnings))

    # Stacks the three (n_slices, size, size) blocks along a new first axis, giving (3, n_slices, size, size).
    array = np.stack(planes)

    return LoadedStudy(
        study_id=study_id,
        array=array,
        selection=selection,
        original_slice_counts=original_slice_counts,
        original_shapes=original_shapes,
        quality_flags=quality_flags,
        warnings=warnings,
    )
