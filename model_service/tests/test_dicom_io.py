# Tests for the DICOM Loader, ensuring that it loads all DICOM files correctly.
# As the real Kaggle data cannot leave Kaggle due to policy reasons, these tests create fake series and tests how the dicom loader file deals with any issues.

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from model_service.knee_model.dicom_io import load_series


# Writes one fake DICOM file.
def _write_slice(
    folder: Path,
    *,
    position: tuple[float, float, float],
    instance_number: int,
    pixels: np.ndarray,
    orientation=(0, 1, 0, 0, 0, 1),
    slope: float = 1.0,
    intercept: float = 0.0,
    photometric: str = "MONOCHROME2",
    include_rescale: bool = True,
) -> Path:
    # Writes one minimal valid DICOM File. Filename is a fresh UID.
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.SeriesDescription = "FAKE SAG"
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = list(position)
    ds.ImageOrientationPatient = list(orientation)
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    if include_rescale:
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept

    path = folder / f"{ds.SOPInstanceUID}.dcm"
    ds.save_as(path, enforce_file_format=True)
    return path

# Writes a whole fake series.
def _make_series(folder: Path, n: int = 5, **kw) -> list[Path]:
    # builds n amount of slices, default is 5, stacked 4 mm apart along the x axis.
    scrambled = list(range(n))[::-1]
    paths = []
    for i in scrambled:
        px = np.full((8, 8), i * 100, dtype=np.uint16)
        paths.append(_write_slice(folder, position=(i * 4.0, 0.0, 0.0), instance_number=i + 1, pixels=px, **kw))
    return paths

# Builds five srambled slices, loads them, reads the centre pixel for each. Must read 0, 100, 200, 300, 400, proving that the loader sorted by position rather than by filename or disk order.
def test_slices_come_out_in_anatomical_order_not_filename_order(tmp_path: Path) -> None:
    _make_series(tmp_path)
    s = load_series(tmp_path)
    # Each slice was filled with index*100, so the centre pixel tells us which slice it really is.
    centre = s.volume[:, 4, 4]
    assert centre.tolist() == [0, 100, 200, 300, 400]
    assert s.instance_numbers.tolist() == [1, 2, 3, 4, 5]
    assert s.ordering_agrees
    assert s.n_slices == 5
    assert s.volume.dtype == np.float32

# Real scans are angled a few degrees, so slices can drift in y and z as well as x. THe test build orientation arrows rotated by 15 degrees, computing the true normal from them
# Then places three slices along that tilted direction. Files are then written in order 2, 0, 1. After loading these slices, they must come out as 0, 100, 200 showing that the loader
# has been able to correctly sort them even when on a tilt.
def test_sorting_follows_the_slice_normal_when_stack_is_tilted(tmp_path: Path) -> None:
    theta = np.deg2rad(15)
    row = (0.0, np.cos(theta), -np.sin(theta))
    col = (0.0, np.sin(theta), np.cos(theta))
    normal = np.cross(row, col)                               # ~ (1, 0, 0)
    for i in [2, 0, 1]:
        pos = tuple((i * 4.0) * normal + np.array([0.0, 0.3 * i, 0.5 * i]))
        _write_slice(tmp_path, position=pos, instance_number=i + 1, pixels=np.full((8, 8), i * 100, dtype=np.uint16), orientation=row + col)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [0, 100, 200]
    assert np.allclose(s.slice_normal, normal / np.linalg.norm(normal))

# Slices are spaced 4mm apart. Reported spacing must be 4.0.
def test_slice_spacing_is_measured(tmp_path: Path) -> None:
    _make_series(tmp_path)
    s = load_series(tmp_path)
    assert s.slice_spacing_mm == pytest.approx(4.0)

# Two slices with slope 2.5 and intercept 10, stored value 0 must become 10, stored 100 must become 260. 100 * 2.5 + 10 = 260.
# Ensures that this is correctly calculated and applied
def test_rescale_slope_and_intercept_are_applied(tmp_path: Path) -> None:
    _make_series(tmp_path, n=2, slope=2.5, intercept=10.0)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [10.0, 260.0]
    assert s.rescale_slope_constant

# Some scanners omit rescale tags. Test writes files without them and checks that the pixel values come through unchanged.
def test_missing_rescale_tags_are_a_harmless_no_op(tmp_path: Path) -> None:
    _make_series(tmp_path, n=2, include_rescale=False)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [0.0, 100.0]
    assert s.rescale_slope_constant
    assert s.warnings == []

# Slope may differ per slice. Two slices both storing 100, one with slope 1 and one with slop 3. Loader must apply each slice's own slope, giving 100 and 300.
# Loader must be able to do this and flag that it was changed.
def test_varying_slope_within_series_is_flagged_not_hidden(tmp_path: Path) -> None:
    _write_slice(tmp_path, position=(0, 0, 0), instance_number=1, pixels=np.full((8, 8), 100, dtype=np.uint16), slope=1.0)
    _write_slice(tmp_path, position=(4, 0, 0), instance_number=2, pixels=np.full((8, 8), 100, dtype=np.uint16), slope=3.0)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [100.0, 300.0]
    assert not s.rescale_slope_constant
    assert any("RescaleSlope varies" in w for w in s.warnings)

# Tests that the Monochrome1 is inverted. Three slices are written at 0, 100, 200 marked at MONOCHROME1. After loading they must be 200, 100, 0, inverting the original slices.
def test_monochrome1_is_inverted_so_bright_means_bright(tmp_path: Path) -> None:
    _make_series(tmp_path, n=3, photometric="MONOCHROME1")
    s = load_series(tmp_path)
    # Raw values were 0, 100, 200. After inversion the slice that was darkest-on-disk is brightest.
    assert s.volume[:, 4, 4].tolist() == [200.0, 100.0, 0.0]
    assert s.photometric == "MONOCHROME1"
    assert any("inverted" in w for w in s.warnings)

# Only MONOCHROME1 must be inverted. Makes sure that a slice tagged as MONOCHROME2 is not inverted.
def test_monochrome2_is_left_alone(tmp_path: Path) -> None:
    _make_series(tmp_path, n=3)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [0.0, 100.0, 200.0]
    assert s.photometric == "MONOCHROME2"

# Three slices are at x = 0, 4, 8, however the scanner numbers 3, 1, 2. Loader must trust geometry so Instance Numbers in volume order come out as 3, 1, 2. Agreement flag must be False
def test_instance_number_disagreement_is_flagged(tmp_path: Path) -> None:
    # Geometry says slice at x=0 comes first, but the scanner numbered it 3. Position wins; the mismatch is logged.
    _write_slice(tmp_path, position=(0, 0, 0), instance_number=3, pixels=np.zeros((8, 8), dtype=np.uint16))
    _write_slice(tmp_path, position=(4, 0, 0), instance_number=1, pixels=np.zeros((8, 8), dtype=np.uint16))
    _write_slice(tmp_path, position=(8, 0, 0), instance_number=2, pixels=np.zeros((8, 8), dtype=np.uint16))
    s = load_series(tmp_path)
    assert s.instance_numbers.tolist() == [3, 1, 2]
    assert not s.ordering_agrees
    assert any("disagrees" in w for w in s.warnings)

# Some scanners count slices from far end, so numbers go 3, 2, 1 as position goes 0, 4, 8.
def test_reversed_instance_numbers_still_count_as_agreement(tmp_path: Path) -> None:
    # Some scanners count slices in the opposite direction to the position axis. That is consistent, not wrong.
    for i, inst in [(0, 3), (1, 2), (2, 1)]:
        _write_slice(tmp_path, position=(i * 4.0, 0, 0), instance_number=inst, pixels=np.zeros((8, 8), dtype=np.uint16))
    s = load_series(tmp_path)
    assert s.ordering_agrees

# Points the loader at an empty path and ensure that it raises FileNotFoundError rather than returning an empty volume.
def test_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_series(tmp_path)

# Creates one 8x8 slice and one 16x16 slice. These slices cannot be stacked together and the loader must raise a ValueError containing different pixel shapes.
def test_mixed_pixel_shapes_raise(tmp_path: Path) -> None:
    _write_slice(tmp_path, position=(0, 0, 0), instance_number=1, pixels=np.zeros((8, 8), dtype=np.uint16))
    _write_slice(tmp_path, position=(4, 0, 0), instance_number=2, pixels=np.zeros((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="different pixel shapes"):
        load_series(tmp_path)
