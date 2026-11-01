"""Uploaded scans for the UI: saves them under UPLOAD_DIR (shared with the Model Service) and renders slice previews."""

from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from model_service.knee_model.upload import (
    UploadError,
    UploadSummary,
    ingest_upload,
    is_upload_id,
    load_model_input,
    load_summary,
    study_dir,
)
from model_service.knee_model.upload import delete_upload as remove_upload
from shared.config import Settings
from shared.schema import PLANES

logger = logging.getLogger(__name__)

MAX_FILES = 5000
CHUNK = 1 << 20
CACHED_ARRAYS = 6  # preview arrays kept in memory (about 6 MB each)


class UploadStore:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.upload_dir
        self.max_bytes = settings.upload_max_mb * 1_000_000
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def ingest(self, files: list[UploadFile]) -> UploadSummary:
        received, total = [], 0
        for upload in files:
            parts = []
            while chunk := upload.file.read(CHUNK):  # read in chunks so the size limit stops huge uploads early
                total += len(chunk)
                if total > self.max_bytes:
                    raise UploadError(f"the upload is larger than {self.max_bytes // 1_000_000} MB (UPLOAD_MAX_MB)")
                parts.append(chunk)
            received.append((upload.filename or "file", b"".join(parts)))
        summary = ingest_upload(received, self.root, max_bytes=self.max_bytes, max_files=MAX_FILES)
        logger.info("upload ingested", extra={"data": {"upload_id": summary.upload_id, "images": summary.n_images,
                                                       "skipped": summary.n_skipped,
                                                       "flags": summary.quality_flags}})
        return summary

    def exists(self, upload_id: str) -> bool:
        return is_upload_id(upload_id) and (study_dir(self.root, upload_id) / "upload.json").exists()

    def summary(self, upload_id: str) -> UploadSummary:
        if not self.exists(upload_id):
            raise HTTPException(status_code=404, detail=f"upload {upload_id} not found")
        return load_summary(self.root, upload_id)

    def slice_png(self, upload_id: str, plane: str, index: int) -> bytes:
        array = self._array(upload_id)
        p = PLANES.index(plane)
        if not 0 <= index < array.shape[1]:
            raise HTTPException(status_code=404, detail=f"slice {index} out of range 0-{array.shape[1] - 1}")
        pixels = (np.clip(array[p, index], 0, 1) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def delete(self, upload_id: str) -> None:
        if not self.exists(upload_id):
            raise HTTPException(status_code=404, detail=f"upload {upload_id} not found")
        with self._lock:
            self._arrays.pop(upload_id, None)
        remove_upload(self.root, upload_id)

    def _array(self, upload_id: str) -> np.ndarray:
        if not self.exists(upload_id):
            raise HTTPException(status_code=404, detail=f"upload {upload_id} not found")
        with self._lock:
            if upload_id in self._arrays:
                self._arrays.move_to_end(upload_id)
                return self._arrays[upload_id]
        array, _, _ = load_model_input(self.root, upload_id)
        with self._lock:
            self._arrays[upload_id] = array
            while len(self._arrays) > CACHED_ARRAYS:
                self._arrays.popitem(last=False)
        return array