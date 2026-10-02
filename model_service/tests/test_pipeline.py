# Tests for the full pipeline, checking that one study ID goes in and one (3, 20, 224, 224) block comes out.
# As the real Kaggle data cannot leave Kaggle, each test builds a small fake study on disk in the same folder layout Kaggle uses
# (train_series/<study>/<series>/<slice>.dcm) and a matching fake series table.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_service.knee_model.pipeline import load_study
from model_service.tests.test_dicom_io import _write_slice

STUDY = "study1"


# Three different pictures, so a test can tell which series ended up in which plane of the output.
def _picture(kind: str, size: int = 32) -> np.ndarray:
    px = np.zeros((size, size), dtype=np.uint16)
    if kind == "left":
        px[:, : size // 2] = 800  # bright left half
    elif kind == "right":
        px[:, size // 2 :] = 800  # bright right half
    elif kind == "top":
        px[: size // 2, :] = 800  # bright top half
    return px


# Writes one fake series folder with n slices that all show the same picture, 4 mm apart.
def _write_series(root: Path, series_id: str, kind: str, n: int = 24) -> None:
    folder = root / "train_series" / STUDY / series_id
    folder.mkdir(parents=True)
    for i in range(n):
        _write_slice(folder, position=(i * 4.0, 0.0, 0.0), instance_number=i + 1, pixels=_picture(kind))


# Builds the fake series table. Each row is given as (series, fluid_sensitive, plane).
def _table(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=["SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"])
    table.insert(0, "StudyInstanceUID", STUDY)
    table["Fat_Suppression"] = table["Fluid_Sensitive"]
    return table


# A normal study on disk: sagittal shows "left", coronal shows "right", axial shows "top".
def _write_normal_study(root: Path) -> pd.DataFrame:
    _write_series(root, "s.200", "left")
    _write_series(root, "s.300", "right")
    _write_series(root, "s.400", "top")
    return _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal"), ("s.400", 1, "Axial")])


# This is the output contract. A normal study must give (3, 20, 224, 224), float32, inside 0 - 1, with no NaN and nothing flagged.
def test_study_comes_out_in_the_contract_shape(tmp_path: Path) -> None:
    table = _write_normal_study(tmp_path)
    study = load_study(STUDY, table, tmp_path)
    assert study.array.shape == (3, 20, 224, 224)
    assert study.array.dtype == np.float32
    assert study.array.min() >= 0.0 and study.array.max() <= 1.0
    assert not np.isnan(study.array).any()
    assert study.quality_flags == []
    assert study.original_slice_counts == {"sagittal": 24, "coronal": 24, "axial": 24}


# The contract fixes the plane order as sagittal, coronal, axial. Each plane was given its own picture, so position 0 must
# be bright on the left, position 1 bright on the right, and position 2 bright at the top.
def test_planes_are_stacked_in_contract_order(tmp_path: Path) -> None:
    table = _write_normal_study(tmp_path)
    sagittal, coronal, axial = load_study(STUDY, table, tmp_path).array
    assert sagittal[:, :, :100].mean() > 0.9 and sagittal[:, :, 124:].mean() < 0.1
    assert coronal[:, :, 124:].mean() > 0.9 and coronal[:, :, :100].mean() < 0.1
    assert axial[:, :100, :].mean() > 0.9 and axial[:, 124:, :].mean() < 0.1


# Two sagittal series on disk. The one that is not fluid-sensitive has the lower UID and shows "top", the fluid-sensitive one
# shows "left". The output must show "left", proving the selection from Step 4 is really what gets loaded.
def test_the_selected_series_is_the_one_that_gets_loaded(tmp_path: Path) -> None:
    table = _write_normal_study(tmp_path)
    _write_series(tmp_path, "s.100", "top")
    table = pd.concat([table, _table([("s.100", 0, "Sagittal")])], ignore_index=True)
    study = load_study(STUDY, table, tmp_path)
    assert study.selection["sagittal"].series_id == "s.200"
    assert study.array[0][:, :, :100].mean() > 0.9


# The study has no axial series. The shape must not change, the axial block must be all zeros, and the flag must use the
# exact wording shared/schema.py gives as its example, missing_plane:axial.
def test_missing_plane_is_filled_with_zeros_and_flagged(tmp_path: Path) -> None:
    _write_series(tmp_path, "s.200", "left")
    _write_series(tmp_path, "s.300", "right")
    table = _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal")])
    study = load_study(STUDY, table, tmp_path)
    assert study.array.shape == (3, 20, 224, 224)
    assert study.array[2].max() == 0.0
    assert study.array[0].max() == 1.0
    assert study.quality_flags == ["missing_plane:axial"]
    assert study.original_slice_counts["axial"] == 0


# The coronal series has only 8 slices, well under the threshold of 15. The shape must still be right, the low_slice_count
# flag must be raised, and the warning must name the plane so it can be traced.
def test_very_short_series_is_flagged_as_low_slice_count(tmp_path: Path) -> None:
    _write_series(tmp_path, "s.200", "left")
    _write_series(tmp_path, "s.300", "right", n=8)
    _write_series(tmp_path, "s.400", "top")
    table = _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal"), ("s.400", 1, "Axial")])
    study = load_study(STUDY, table, tmp_path)
    assert study.array.shape == (3, 20, 224, 224)
    assert study.quality_flags == ["low_slice_count"]
    assert study.original_slice_counts["coronal"] == 8
    assert any(w.startswith("coronal:") and "fewer than" in w for w in study.warnings)


# The coronal series has 18 slices. That is under the 20 wanted, so two slices are repeated and a warning must say so,
# but it is not under the threshold of 15, so the flag must NOT be raised. This is the common case in the real data
# (56 of 308 studies) and the reason the threshold exists.
def test_slightly_short_series_is_warned_about_but_not_flagged(tmp_path: Path) -> None:
    _write_series(tmp_path, "s.200", "left")
    _write_series(tmp_path, "s.300", "right", n=18)
    _write_series(tmp_path, "s.400", "top")
    table = _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal"), ("s.400", 1, "Axial")])
    study = load_study(STUDY, table, tmp_path)
    assert study.quality_flags == []
    assert any(w.startswith("coronal:") and "fewer than" in w for w in study.warnings)


# The threshold itself. A series of exactly 15 slices is not flagged and a series of 14 is. Checking both sides of the
# line catches an off-by-one mistake (using "15 or fewer" where "fewer than 15" was meant).
def test_low_slice_count_threshold_is_fewer_than_15(tmp_path: Path) -> None:
    _write_series(tmp_path, "s.200", "left", n=15)
    _write_series(tmp_path, "s.300", "right")
    _write_series(tmp_path, "s.400", "top")
    table = _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal"), ("s.400", 1, "Axial")])
    assert load_study(STUDY, table, tmp_path).quality_flags == []

    _write_series(tmp_path, "s.250", "left", n=14)
    table = _table([("s.250", 1, "Sagittal"), ("s.300", 1, "Coronal"), ("s.400", 1, "Axial")])
    assert load_study(STUDY, table, tmp_path).quality_flags == ["low_slice_count"]


# The sagittal series is 32 x 32 and the coronal series is 20 high by 40 wide, and there is no axial series. After preprocessing
# every plane is 224 x 224, so the sizes the scanner produced must be kept separately as (height, width), with (0, 0) for the missing plane.
def test_original_slice_sizes_are_recorded(tmp_path: Path) -> None:
    _write_series(tmp_path, "s.200", "left")
    wide_folder = tmp_path / "train_series" / STUDY / "s.300"
    wide_folder.mkdir(parents=True)
    for i in range(24):
        _write_slice(wide_folder, position=(i * 4.0, 0.0, 0.0), instance_number=i + 1, pixels=np.full((20, 40), 500, dtype=np.uint16))
    table = _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal")])
    study = load_study(STUDY, table, tmp_path)
    assert study.original_shapes == {"sagittal": (32, 32), "coronal": (20, 40), "axial": (0, 0)}
    assert study.array.shape == (3, 20, 224, 224)


# The table lists an axial series but its folder is not on disk. That is broken data, so it must raise an error and not
# quietly hand back a black plane that looks like a valid result.
def test_series_folder_missing_on_disk_raises(tmp_path: Path) -> None:
    _write_series(tmp_path, "s.200", "left")
    _write_series(tmp_path, "s.300", "right")
    table = _table([("s.200", 1, "Sagittal"), ("s.300", 1, "Coronal"), ("s.400", 1, "Axial")])
    with pytest.raises(FileNotFoundError):
        load_study(STUDY, table, tmp_path)


# N and the image size are passed through. Asking for 16 slices at 64 pixels must give (3, 16, 64, 64).
def test_slice_count_and_size_are_passed_through(tmp_path: Path) -> None:
    table = _write_normal_study(tmp_path)
    study = load_study(STUDY, table, tmp_path, n_slices=16, size=64)
    assert study.array.shape == (3, 16, 64, 64)
