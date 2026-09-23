# Tests for the DICOM loader, using tiny DICOM files built on the fly.
#
# The real Kaggle data cannot leave Kaggle, so these tests manufacture small fake series in a temporary folder.
# Each test plants one specific problem (files out of order, a photo-negative series, a varying rescale slope)
# and checks the loader handles it. The visual check on real knees still happens in the Kaggle notebook;
# these tests cover the things a human eye cannot see.

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from model_service.knee_model.dicom_io import load_series


def _write_slice(
    folder: Path,
    *,
    position: tuple[float, float, float],
    instance_number: int,
    pixels: np.ndarray,
    orientation=(0, 1, 0, 0, 0, 1),       # rows run along +y, columns along +z, so the normal (row x col) is +x.
                                          # Real sagittal scans often use (0,1,0, 0,0,-1) whose normal is -x; the loader
                                          # handles either, it just stacks in the direction the normal points.
    slope: float = 1.0,
    intercept: float = 0.0,
    photometric: str = "MONOCHROME2",
    include_rescale: bool = True,
) -> Path:
    """Write one minimal but valid DICOM file. Filename is a fresh UID, exactly like the real dataset."""
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


def _make_series(folder: Path, n: int = 5, **kw) -> list[Path]:
    """n slices stacked along x, 4 mm apart, each filled with its own slice index so we can tell them apart.
    InstanceNumber counts up with x. Files are written in a scrambled order on purpose."""
    scrambled = list(range(n))[::-1]        # written last-to-first so disk order is the reverse of anatomical order
    paths = []
    for i in scrambled:
        px = np.full((8, 8), i * 100, dtype=np.uint16)
        paths.append(_write_slice(folder, position=(i * 4.0, 0.0, 0.0), instance_number=i + 1, pixels=px, **kw))
    return paths


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


def test_sorting_follows_the_slice_normal_when_stack_is_tilted(tmp_path: Path) -> None:
    # Tilt the plane so the normal is not a pure axis. Positions move mostly in x but also in y and z,
    # like the real series measured in Step 1 (x 97 mm, y 3 mm, z 6 mm). Sorting by x alone would still work
    # here; sorting by y alone would be wrong. The loader must not care which axis dominates.
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


def test_slice_spacing_is_measured(tmp_path: Path) -> None:
    _make_series(tmp_path)
    s = load_series(tmp_path)
    assert s.slice_spacing_mm == pytest.approx(4.0)


def test_rescale_slope_and_intercept_are_applied(tmp_path: Path) -> None:
    _make_series(tmp_path, n=2, slope=2.5, intercept=10.0)
    s = load_series(tmp_path)
    # stored 0 -> 0*2.5+10 = 10 ; stored 100 -> 100*2.5+10 = 260
    assert s.volume[:, 4, 4].tolist() == [10.0, 260.0]
    assert s.rescale_slope_constant


def test_missing_rescale_tags_are_a_harmless_no_op(tmp_path: Path) -> None:
    _make_series(tmp_path, n=2, include_rescale=False)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [0.0, 100.0]
    assert s.rescale_slope_constant
    assert s.warnings == []


def test_varying_slope_within_series_is_flagged_not_hidden(tmp_path: Path) -> None:
    # Philips can write a different slope per slice. We still apply each slice's own slope (that is the correct
    # thing to do), but the flag must trip so the batch run can log the series.
    _write_slice(tmp_path, position=(0, 0, 0), instance_number=1, pixels=np.full((8, 8), 100, dtype=np.uint16), slope=1.0)
    _write_slice(tmp_path, position=(4, 0, 0), instance_number=2, pixels=np.full((8, 8), 100, dtype=np.uint16), slope=3.0)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [100.0, 300.0]
    assert not s.rescale_slope_constant
    assert any("RescaleSlope varies" in w for w in s.warnings)


def test_monochrome1_is_inverted_so_bright_means_bright(tmp_path: Path) -> None:
    _make_series(tmp_path, n=3, photometric="MONOCHROME1")
    s = load_series(tmp_path)
    # Raw values were 0, 100, 200. After inversion the slice that was darkest-on-disk is brightest.
    assert s.volume[:, 4, 4].tolist() == [200.0, 100.0, 0.0]
    assert s.photometric == "MONOCHROME1"
    assert any("inverted" in w for w in s.warnings)


def test_monochrome2_is_left_alone(tmp_path: Path) -> None:
    _make_series(tmp_path, n=3)
    s = load_series(tmp_path)
    assert s.volume[:, 4, 4].tolist() == [0.0, 100.0, 200.0]
    assert s.photometric == "MONOCHROME2"


def test_instance_number_disagreement_is_flagged(tmp_path: Path) -> None:
    # Geometry says slice at x=0 comes first, but the scanner numbered it 3. Position wins; the mismatch is logged.
    _write_slice(tmp_path, position=(0, 0, 0), instance_number=3, pixels=np.zeros((8, 8), dtype=np.uint16))
    _write_slice(tmp_path, position=(4, 0, 0), instance_number=1, pixels=np.zeros((8, 8), dtype=np.uint16))
    _write_slice(tmp_path, position=(8, 0, 0), instance_number=2, pixels=np.zeros((8, 8), dtype=np.uint16))
    s = load_series(tmp_path)
    assert s.instance_numbers.tolist() == [3, 1, 2]
    assert not s.ordering_agrees
    assert any("disagrees" in w for w in s.warnings)


def test_reversed_instance_numbers_still_count_as_agreement(tmp_path: Path) -> None:
    # Some scanners count slices in the opposite direction to the position axis. That is consistent, not wrong.
    for i, inst in [(0, 3), (1, 2), (2, 1)]:
        _write_slice(tmp_path, position=(i * 4.0, 0, 0), instance_number=inst, pixels=np.zeros((8, 8), dtype=np.uint16))
    s = load_series(tmp_path)
    assert s.ordering_agrees


def test_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_series(tmp_path)


def test_mixed_pixel_shapes_raise(tmp_path: Path) -> None:
    _write_slice(tmp_path, position=(0, 0, 0), instance_number=1, pixels=np.zeros((8, 8), dtype=np.uint16))
    _write_slice(tmp_path, position=(4, 0, 0), instance_number=2, pixels=np.zeros((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="different pixel shapes"):
        load_series(tmp_path)
