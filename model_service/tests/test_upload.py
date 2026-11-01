# Tests for scans uploaded through the UI: header-based plane/series picking, preprocessing, and Model Service scoring.

import numpy as np
import pytest
from fastapi.testclient import TestClient

from model_service.api.main import create_app
from model_service.knee_model.upload import (
    UploadError,
    ingest_upload,
    is_fluid_sensitive,
    load_model_input,
    load_summary,
    plane_from_orientation,
)
from model_service.tests.fake_upload import fake_study, zipped
from shared.config import Settings
from shared.schema import FINDINGS, PLANES


@pytest.mark.parametrize("iop, plane", [
    ((0, 1, 0, 0, 0, -1), "sagittal"),
    ((1, 0, 0, 0, 0, -1), "coronal"),
    ((1, 0, 0, 0, 1, 0), "axial"),
    ((0.99, 0.1, 0, -0.1, 0.99, 0.05), "axial"),  # a few degrees of tilt
    ((1, 0, 0), None),
])
def test_plane_comes_from_the_orientation_tag(iop, plane):
    assert plane_from_orientation(iop) == plane


@pytest.mark.parametrize("description, expected", [
    ("PD FS SAG", True), ("COR PD SPAIR", True), ("AX T2 FATSAT", True), ("sag stir", True),
    ("Fat Sat T2 AX", True), ("SAG T1 TSE", False), ("COR PD", False), ("AX T2 FSE", False),
])
def test_fluid_sensitive_keywords(description, expected):
    assert is_fluid_sensitive(description) is expected


def test_zip_upload_becomes_a_model_ready_study(tmp_path):
    summary = ingest_upload([("study.zip", zipped(fake_study()))], tmp_path)
    assert [p.plane for p in summary.planes] == list(PLANES)
    assert all(p.present and p.level == 1 for p in summary.planes)  # the fat-suppressed series wins every plane
    assert summary.quality_flags == []
    assert summary.laterality == "right" and summary.laterality_source == "dicom_tag"
    assert summary.n_skipped == 2  # the localiser and the text file
    array, present, flags = load_model_input(tmp_path, summary.upload_id)
    assert array.shape == (3, 20, 224, 224) and present.all() and flags == []
    assert 0.0 <= array.min() and array.max() <= 1.0
    assert load_summary(tmp_path, summary.upload_id) == summary


def test_loose_files_with_a_missing_plane_are_flagged(tmp_path):
    summary = ingest_upload(fake_study(planes=("sagittal", "coronal"), n=10, laterality=None), tmp_path)
    assert summary.quality_flags == ["low_slice_count", "missing_plane:axial"]
    assert summary.laterality == "unknown"
    _, present, _ = load_model_input(tmp_path, summary.upload_id)
    assert present.tolist() == [True, True, False]


def test_upload_is_rebuilt_at_another_size_for_a_different_checkpoint(tmp_path):
    summary = ingest_upload(fake_study(), tmp_path)
    array, _, _ = load_model_input(tmp_path, summary.upload_id, n_slices=4, size=32)
    assert array.shape == (3, 4, 32, 32)


@pytest.mark.parametrize("files, message", [
    ([("photo.jpg", b"\xff\xd8\xff\xe0 jpeg")], "No MRI images found"),
    ([("bad.zip", b"PK\x03\x04 broken")], "not a readable .zip"),
    ([], "no files"),
])
def test_unusable_uploads_are_refused_with_a_clear_message(tmp_path, files, message):
    with pytest.raises(UploadError, match=message):
        ingest_upload(files, tmp_path)
    assert not any((tmp_path / "studies").glob("*")) if (tmp_path / "studies").exists() else True


def test_size_limit(tmp_path):
    with pytest.raises(UploadError, match="larger than"):
        ingest_upload(fake_study(), tmp_path, max_bytes=10_000)


# --- Model Service scoring of uploads ---

def test_stub_mode_scores_uploads_without_the_missing_plane(tmp_path):
    summary = ingest_upload(fake_study(planes=("sagittal", "axial")), tmp_path)
    client = TestClient(create_app(Settings(_env_file=None, model_mode="stub", upload_dir=str(tmp_path))))
    assert client.get("/health").json()["uploads"] == "stub"
    body = client.post("/predict", json={"study_id": summary.upload_id}).json()
    assert [p["plane"] for p in body["per_plane"]] == ["sagittal", "axial"]
    assert "missing_plane:coronal" in body["quality_flags"]
    assert body["laterality"] == "right" and body["model_version"].startswith("stub-")
    mean = np.mean([[p["scores"][f] for f in FINDINGS] for p in body["per_plane"]], axis=0)
    assert np.allclose([body["findings"][f] for f in FINDINGS], mean)


def test_unknown_upload_is_404(tmp_path):
    client = TestClient(create_app(Settings(_env_file=None, model_mode="stub", upload_dir=str(tmp_path))))
    assert client.post("/predict", json={"study_id": "upl-000000000000"}).status_code == 404


def test_replay_mode_without_a_checkpoint_explains_why_uploads_cannot_be_scored(tmp_path):
    summary = ingest_upload(fake_study(), tmp_path)
    settings = Settings(_env_file=None, model_mode="replay", upload_dir=str(tmp_path),
                        replay_predictions=str(tmp_path / "missing.json"), model_checkpoint=str(tmp_path / "no.pt"))
    client = TestClient(create_app(settings))
    assert client.get("/health").json()["uploads"] == "unavailable"
    response = client.post("/predict", json={"study_id": summary.upload_id})
    assert response.status_code == 503 and "checkpoint" in response.json()["detail"]


def test_checkpoint_scores_uploads(tmp_path, monkeypatch):
    # A fake predictor stands in for KneePredictor so this runs without torch.
    summary = ingest_upload(fake_study(), tmp_path)
    checkpoint = tmp_path / "knee_v1.pt"
    checkpoint.write_bytes(b"x")

    class FakePredictor:
        model_version, device, n_slices, image_size = "knee_v1", "cpu", 4, 32

        def __init__(self, *args, **kwargs):
            pass

        def predict_array(self, array, present):
            assert array.shape == (3, 4, 32, 32)
            return np.full(12, 0.7), np.full((3, 12), 0.7)

    monkeypatch.setattr("model_service.knee_model.inference.KneePredictor", FakePredictor)
    settings = Settings(_env_file=None, model_mode="replay", upload_dir=str(tmp_path),
                        replay_predictions=str(tmp_path / "missing.json"), model_checkpoint=str(checkpoint))
    client = TestClient(create_app(settings))
    body = client.post("/predict", json={"study_id": summary.upload_id}).json()
    assert body["model_version"] == "knee_v1" and body["findings"]["ACL"] == pytest.approx(0.7)