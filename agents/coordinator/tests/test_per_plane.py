# Per-plane scores: Model Service -> DiagnosticAgentResponse.per_plane -> ReportedFinding.per_plane.

import pytest
from coordinator_fakes import make_settings
from pydantic import ValidationError

from agents.coordinator import CoordinatorAgent
from agents.coordinator.fakes import ScriptedDiagnostic, ScriptedRecommendation, diagnostic, recommendation
from agents.coordinator.runners import prediction_to_diagnostic
from agents.diagnostic_agent.agent import DiagnosticAgent
from shared.schema import FINDINGS, PLANES, DiagnosticAgentRequest, PlaneScores, PredictionResponse

SID = "1.2.826.0.1.3680043.8.498.10001"


def _scores(acl: float) -> dict[str, float]:
    return {name: 0.1 for name in FINDINGS} | {"ACL": acl}


def _prediction(planes=PLANES, flags=()) -> PredictionResponse:
    per_plane = [PlaneScores(plane=p, scores=_scores(acl)) for p, acl in zip(planes, (0.9, 0.8, 0.7))]
    return PredictionResponse(study_id=SID, findings=_scores(0.8), per_plane=per_plane, laterality="left",
                              laterality_source="dicom_tag", quality_flags=list(flags), model_version="knee_v1",
                              latency_ms=5.0)


def test_schema_accepts_per_plane_and_defaults_to_empty():
    assert diagnostic(SID).per_plane == {}
    result = diagnostic(SID, per_plane={"axial": _scores(0.7)})
    assert result.per_plane["axial"]["ACL"] == 0.7


@pytest.mark.parametrize("bad", [{"sideways": _scores(0.5)}, {"axial": {"ACL": 0.5}}, {"axial": _scores(1.5)}])
def test_schema_rejects_bad_per_plane(bad):
    with pytest.raises(ValidationError):
        diagnostic(SID, per_plane=bad)


def test_interim_adapter_copies_per_plane_scores():
    result = prediction_to_diagnostic(_prediction(), DiagnosticAgentRequest(study_id=SID))
    assert list(result.per_plane) == list(PLANES)
    assert [result.per_plane[p]["ACL"] for p in PLANES] == [0.9, 0.8, 0.7]


def test_missing_plane_is_left_out_not_zero_filled():
    result = prediction_to_diagnostic(_prediction(planes=("sagittal", "coronal"), flags=["missing_plane:axial"]),
                                      DiagnosticAgentRequest(study_id=SID))
    assert set(result.per_plane) == {"sagittal", "coronal"}


def test_diagnostic_agent_passes_per_plane_through(tmp_path):
    agent = DiagnosticAgent(None, make_settings(tmp_path), predictor=lambda study_id, timeout: _prediction())
    result = agent(DiagnosticAgentRequest(study_id=SID), session_id="s1")
    assert result.per_plane["coronal"]["ACL"] == 0.8


def test_multi_agent_report_fills_per_plane(tmp_path):
    planes = {p: _scores(acl) for p, acl in zip(PLANES, (0.9, 0.8, 0.7))}
    diag = ScriptedDiagnostic([diagnostic(SID, {"ACL": 0.8}, per_plane=planes)])
    coordinator = CoordinatorAgent(diagnostic=diag, recommendation=ScriptedRecommendation([recommendation()]),
                                   llm=None, settings=make_settings(tmp_path), driver="fixed")
    report = coordinator.run(SID, session_id="s1").report
    acl = next(f for f in report.findings if f.name == "ACL")
    assert acl.per_plane == {"sagittal": 0.9, "coronal": 0.8, "axial": 0.7}


def test_report_per_plane_stays_empty_when_scorer_gave_none(tmp_path):
    diag = ScriptedDiagnostic([diagnostic(SID, {"ACL": 0.8})])
    coordinator = CoordinatorAgent(diagnostic=diag, recommendation=ScriptedRecommendation([recommendation()]),
                                   llm=None, settings=make_settings(tmp_path), driver="fixed")
    assert all(f.per_plane == {} for f in coordinator.run(SID, session_id="s1").report.findings)