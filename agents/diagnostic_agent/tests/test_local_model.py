"""Local Diagnostic Agent model on UI uploads: scoring, missing uploads, and sharing the scorer with the baseline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from agents.coordinator.runners import StudyNotAvailableError  # noqa: E402
from agents.diagnostic_agent.agent import DiagnosticAgent  # noqa: E402
from agents.diagnostic_agent.local_model.model import NotebookKneeNet  # noqa: E402
from agents.diagnostic_agent.local_model.predictor import LocalKneePredictor  # noqa: E402
from model_service.knee_model.upload import ingest_upload  # noqa: E402
from model_service.tests.fake_upload import fake_study  # noqa: E402
from shared.config import Settings  # noqa: E402
from shared.schema import FINDINGS, DiagnosticAgentRequest  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path):
    # Random weights in the notebook's layout; small and fast, no download needed
    torch.manual_seed(0)
    path = tmp_path / "tiny.pt"
    torch.save(NotebookKneeNet(pretrained=False).state_dict(), path)
    return path


@pytest.fixture
def settings(tmp_path, checkpoint):
    return Settings(_env_file=None, upload_dir=str(tmp_path / "uploads"), diagnostic_local_checkpoint=str(checkpoint),
                    model_device="cpu", log_dir=str(tmp_path / "logs"))


def test_scores_an_upload(settings, checkpoint):
    summary = ingest_upload(fake_study(), settings.upload_dir)
    predictor = LocalKneePredictor(checkpoint, settings, depth=8, size=64)
    result = predictor.predict(summary.upload_id)
    assert result.study_id == summary.upload_id
    assert set(result.findings) == set(FINDINGS)
    assert [p.plane for p in result.per_plane] == ["sagittal", "coronal", "axial"]
    assert result.laterality == "right" and result.laterality_source == "dicom_tag"


def test_missing_plane_is_flagged_not_scored(settings, checkpoint):
    summary = ingest_upload(fake_study(planes=("sagittal", "coronal")), settings.upload_dir)
    result = LocalKneePredictor(checkpoint, settings, depth=8, size=64).predict(summary.upload_id)
    assert "missing_plane:axial" in result.quality_flags
    assert [p.plane for p in result.per_plane] == ["sagittal", "coronal"]


def test_unknown_upload_is_not_available(settings, checkpoint):
    with pytest.raises(StudyNotAvailableError):
        LocalKneePredictor(checkpoint, settings, depth=8, size=64)("upl-000000000000")


def test_agent_uses_local_model_and_exposes_it_for_the_baseline(settings):
    agent = DiagnosticAgent(llm=None, settings=settings)
    assert agent.source.startswith("local model") and agent.scorer is agent.predictor
    summary = ingest_upload(fake_study(), settings.upload_dir)
    response = agent(DiagnosticAgentRequest(study_id=summary.upload_id), session_id="t")
    assert response.impression and any("local model" in line for line in response.trace)


def test_model_service_path_has_no_local_scorer(tmp_path):
    settings = Settings(_env_file=None, diagnostic_local_checkpoint=str(tmp_path / "absent.pt"))
    agent = DiagnosticAgent(llm=None, settings=settings)
    assert agent.source.startswith("Model Service") and agent.scorer is None

def test_dataset_id_without_kaggle_data_is_not_available(settings, checkpoint, tmp_path):
    settings.live_series_csv = str(tmp_path / "missing.csv")
    predictor = LocalKneePredictor(checkpoint, settings, depth=8, size=64)
    with pytest.raises(StudyNotAvailableError, match="upload"):
        predictor.predict("1.2.826.0.1.3680043.8.498.10001")