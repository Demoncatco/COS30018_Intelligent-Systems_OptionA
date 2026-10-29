# Tests for MODEL_MODE=replay and MODEL_MODE=live, plus the helpers they share with the Kaggle exporter.
# Live tests build a tiny untrained checkpoint (4 slices of 32x32) and a fake DICOM study, so no GPU, dataset or
# downloaded weights are needed.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from fastapi.testclient import TestClient

from model_service.api.main import create_app
from model_service.knee_model.inference import build_prediction, read_laterality
from shared.config import Settings
from shared.schema import FINDINGS, PLANES, PredictionResponse, ReplayPredictionsFile

SMALL_SLICES, SMALL_SIZE = 4, 32


def _probs(value: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    return np.full(len(FINDINGS), value), np.full((len(PLANES), len(FINDINGS)), value)


def _prediction(study_id: str, acl: float, version: str = "knee_v1") -> PredictionResponse:
    study, planes = _probs()
    study[FINDINGS.index("ACL")] = acl
    return build_prediction(study_id, study, planes, np.array([True, True, True]), quality_flags=[],
                            laterality="left", laterality_source="dicom_tag", model_version=version, latency_ms=900.0)


def _write_replay(path: Path) -> Path:
    replay = ReplayPredictionsFile(model_version="knee_v1", source="test",
                                   predictions=[_prediction("study-a", 0.91), _prediction("study-b", 0.05)])
    path.write_text(replay.model_dump_json(indent=2))
    return path


# --- build_prediction ---

def test_missing_planes_are_dropped_from_per_plane_and_flagged():
    study, planes = _probs()
    response = build_prediction("s", study, planes, np.array([True, True, False]), quality_flags=["low_slice_count"],
                                laterality="unknown", laterality_source="unavailable", model_version="knee_v1",
                                latency_ms=3.0)
    assert [p.plane for p in response.per_plane] == ["sagittal", "coronal"]
    assert response.quality_flags == ["low_slice_count", "missing_plane:axial"]


def test_existing_missing_plane_flag_is_not_duplicated():
    study, planes = _probs()
    response = build_prediction("s", study, planes, np.array([True, False, True]),
                                quality_flags=["missing_plane:coronal"], laterality="unknown",
                                laterality_source="unavailable", model_version="knee_v1", latency_ms=1.0)
    assert response.quality_flags == ["missing_plane:coronal"]


@pytest.mark.parametrize("study, planes, present", [
    (np.full(11, 0.2), np.full((3, 12), 0.2), np.ones(3, bool)),          # wrong finding count
    (np.full(12, np.nan), np.full((3, 12), 0.2), np.ones(3, bool)),       # NaN from a broken checkpoint
    (np.full(12, 0.2), np.full((3, 12), 0.2), np.zeros(3, bool)),         # nothing to score
])
def test_build_prediction_rejects_bad_model_output(study, planes, present):
    with pytest.raises(ValueError):
        build_prediction("s", study, planes, present, quality_flags=[], laterality="unknown",
                         laterality_source="unavailable", model_version="knee_v1", latency_ms=1.0)


# --- replay mode ---

def test_replay_serves_the_exported_predictions(tmp_path):
    client = TestClient(create_app(Settings(model_mode="replay", replay_predictions=str(_write_replay(tmp_path / "p.json")),
                                            model_checkpoint=str(tmp_path / "none.pt"))))

    health = client.get("/health").json()
    assert health == {"status": "ok", "mode": "replay", "model_version": "knee_v1", "n_studies": 2, "detail": "",
                      "uploads": "unavailable"}
    assert client.get("/studies").json()["study_ids"] == ["study-a", "study-b"]

    body = PredictionResponse.model_validate(client.post("/predict", json={"study_id": "study-a"}).json())
    assert body.findings["ACL"] == pytest.approx(0.91)
    assert body.model_version == "knee_v1"
    assert body.latency_ms < 900.0  # the service's own time, not the recorded Kaggle time


def test_replay_returns_404_for_a_study_that_was_not_exported(tmp_path):
    client = TestClient(create_app(Settings(model_mode="replay", replay_predictions=str(_write_replay(tmp_path / "p.json")))))
    response = client.post("/predict", json={"study_id": "never-exported"})
    assert response.status_code == 404
    assert "replay" in response.json()["detail"]


def test_missing_replay_file_reports_503_instead_of_crashing(tmp_path):
    client = TestClient(create_app(Settings(model_mode="replay", replay_predictions=str(tmp_path / "nope.json"))))
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["status"] == "error"
    assert "nope.json" in health.json()["detail"]
    assert client.post("/predict", json={"study_id": "x"}).status_code == 503


def test_stub_output_in_a_replay_file_is_refused(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"model_version": "stub-0.1.0", "predictions": []}))
    client = TestClient(create_app(Settings(model_mode="replay", replay_predictions=str(path))))
    assert client.get("/health").status_code == 503


# --- laterality ---

def _write_fake_study(root: Path, study_id: str = "study1", laterality: str | None = None) -> pd.DataFrame:
    from model_service.tests.test_dicom_io import _write_slice

    rows = []
    for plane, series_id in zip(("Sagittal", "Coronal", "Axial"), ("s.200", "s.300", "s.400")):
        folder = root / "train_series" / study_id / series_id
        folder.mkdir(parents=True)
        for i in range(6):
            pixels = np.random.default_rng(i).integers(0, 800, size=(32, 32)).astype(np.uint16)
            path = _write_slice(folder, position=(i * 4.0, 0.0, 0.0), instance_number=i + 1, pixels=pixels)
            if laterality is not None:
                ds = pydicom.dcmread(path)
                ds.Laterality = laterality
                ds.save_as(path)
        rows.append((study_id, series_id, 1, plane))
    table = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"])
    table["Fat_Suppression"] = 1
    return table


def test_laterality_is_read_from_the_dicom_tag(tmp_path):
    _write_fake_study(tmp_path, laterality="R")
    dirs = sorted((tmp_path / "train_series" / "study1").iterdir())
    assert read_laterality(dirs) == ("right", "dicom_tag")


def test_laterality_without_a_tag_is_unknown(tmp_path):
    _write_fake_study(tmp_path)
    dirs = sorted((tmp_path / "train_series" / "study1").iterdir())
    assert read_laterality(dirs) == ("unknown", "unavailable")


# --- live mode (needs torch) ---

@pytest.fixture
def small_checkpoint(tmp_path) -> Path:
    pytest.importorskip("torch")
    from model_service.knee_model.model import KneeNet, save_checkpoint

    path = tmp_path / "knee_vtest.pt"
    save_checkpoint(path, KneeNet(pretrained=False), {"version": "vtest", "n_slices": SMALL_SLICES,
                                                     "image_size": SMALL_SIZE})
    return path


def _live_settings(tmp_path: Path, checkpoint: Path) -> Settings:
    table = _write_fake_study(tmp_path / "data", laterality="L")
    series_csv = tmp_path / "data" / "train_series.csv"
    table.to_csv(series_csv, index=False)
    return Settings(model_mode="live", model_checkpoint=str(checkpoint), live_dataset_root=str(tmp_path / "data"),
                    live_series_csv=str(series_csv), model_device="cpu")


def test_live_mode_scores_a_dicom_study_end_to_end(tmp_path, small_checkpoint):
    client = TestClient(create_app(_live_settings(tmp_path, small_checkpoint)))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["mode"] == "live" and health["model_version"] == "knee_vtest"
    assert health["n_studies"] == 1

    response = client.post("/predict", json={"study_id": "study1"})
    assert response.status_code == 200, response.text
    body = PredictionResponse.model_validate(response.json())
    assert set(body.findings) == set(FINDINGS)
    assert [p.plane for p in body.per_plane] == list(PLANES)
    assert body.laterality == "left" and body.laterality_source == "dicom_tag"
    assert "low_slice_count" in body.quality_flags  # the fake series only has 6 slices


def test_live_mode_returns_404_for_an_unknown_study(tmp_path, small_checkpoint):
    client = TestClient(create_app(_live_settings(tmp_path, small_checkpoint)))
    assert client.post("/predict", json={"study_id": "not-a-study"}).status_code == 404


def test_live_mode_without_a_checkpoint_reports_503(tmp_path, small_checkpoint):
    settings = _live_settings(tmp_path, small_checkpoint).model_copy(update={"model_checkpoint": str(tmp_path / "x.pt")})
    client = TestClient(create_app(settings))
    assert client.get("/health").status_code == 503


def test_live_and_exported_replay_agree_for_the_same_study(tmp_path, small_checkpoint):
    # The exporter and live mode share inference.py, so replaying an export must give live's numbers.
    from model_service.knee_model.export_predictions import export
    from model_service.knee_model.inference import KneePredictor

    settings = _live_settings(tmp_path, small_checkpoint)
    live = TestClient(create_app(settings)).post("/predict", json={"study_id": "study1"}).json()

    predictor = KneePredictor(small_checkpoint, device="cpu")
    replay, failures = export(predictor, ["study1", "missing"], cache_dir=None,
                              series_table=pd.read_csv(settings.live_series_csv),
                              dataset_root=Path(settings.live_dataset_root), split_folder="train_series", source="test")
    assert [f["study_id"] for f in failures] == ["missing"]
    path = tmp_path / "predictions.json"
    path.write_text(replay.model_dump_json())

    replayed = TestClient(create_app(Settings(model_mode="replay", replay_predictions=str(path)))).post(
        "/predict", json={"study_id": "study1"}).json()
    assert replayed["findings"] == pytest.approx(live["findings"])
    assert replayed["per_plane"] == live["per_plane"]


def test_exporter_cli_uses_the_cache_and_the_validation_split(tmp_path, small_checkpoint):
    from model_service.knee_model.export_predictions import main
    from model_service.tests._fakes import write_cache

    write_cache(tmp_path / "cache", ["gold0", "gold1", "gold2"], missing_axial={"gold1"})
    pd.DataFrame({"study_id": ["gold0", "gold1", "gold2"], "label_source": "gold",
                  "split": ["val", "val", "train"]}).to_csv(tmp_path / "splits.csv", index=False)
    out = tmp_path / "out" / "predictions.json"
    code = main(["--checkpoint", str(small_checkpoint), "--splits-csv", str(tmp_path / "splits.csv"),
                 "--cache-dir", str(tmp_path / "cache"), "--device", "cpu", "--out", str(out)])
    assert code == 0
    replay = ReplayPredictionsFile.model_validate_json(out.read_text())
    assert [p.study_id for p in replay.predictions] == ["gold0", "gold1"]  # train studies are not exported
    by_id = {p.study_id: p for p in replay.predictions}
    assert by_id["gold1"].quality_flags == ["missing_plane:axial"]
    assert [p.plane for p in by_id["gold1"].per_plane] == ["sagittal", "coronal"]