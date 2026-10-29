# Builds a small fake knee study as DICOM bytes (3 planes, a localiser and a stray file) for the upload tests.

from __future__ import annotations

import io
import zipfile

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# Orientation (row, column) per plane; the slice normal is x, y or z respectively.
ORIENTATIONS = {"sagittal": (0, 1, 0, 0, 0, -1), "coronal": (1, 0, 0, 0, 0, -1), "axial": (1, 0, 0, 0, 1, 0)}
NORMAL_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def dicom_bytes(*, series_uid: str, description: str, orientation, position, number: int, pixels: np.ndarray,
                image_type=("ORIGINAL", "PRIMARY"), laterality: str | None = None) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.ImageType = list(image_type)
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = description
    ds.InstanceNumber = number
    ds.ImagePositionPatient = list(position)
    ds.ImageOrientationPatient = list(orientation)
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    if laterality:
        ds.Laterality = laterality
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def series_files(plane: str, description: str, n: int = 18, size: int = 48, seed: int = 0,
                 laterality: str | None = None) -> list[tuple[str, bytes]]:
    rng = np.random.default_rng(seed)
    uid = generate_uid()
    files = []
    for i in range(n):
        position = [0.0, 0.0, 0.0]
        position[NORMAL_AXIS[plane]] = i * 3.0
        pixels = rng.integers(0, 1000, size=(size, size)).astype(np.uint16)
        files.append((f"{plane}_{seed}_{i}.dcm", dicom_bytes(series_uid=uid, description=description,
                                                       orientation=ORIENTATIONS[plane], position=position,
                                                       number=i + 1, pixels=pixels, laterality=laterality)))
    return files


def fake_study(planes=("sagittal", "coronal", "axial"), n: int = 18, laterality: str | None = "R") -> list[tuple[str, bytes]]:
    names = {"sagittal": "PD FS SAG", "coronal": "COR PD SPAIR", "axial": "AX T2 FATSAT"}
    files = []
    for k, plane in enumerate(planes):
        files += series_files(plane, names[plane], n=n, seed=k, laterality=laterality)
        files += series_files(plane, f"{plane[:3].upper()} T1 TSE", n=n, seed=10 + k)[:6]  # level-2 fallback
    files.append(("scout.dcm", dicom_bytes(series_uid=generate_uid(), description="3-plane loc", orientation=(1, 0, 0, 0, 1, 0),
                                           position=(0, 0, 0), number=1, pixels=np.zeros((16, 16)),
                                           image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"))))
    files.append(("notes.txt", b"not an image"))
    return files


def zipped(files: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files:
            archive.writestr(f"study/{name}", data)
    return buffer.getvalue()