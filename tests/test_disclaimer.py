"""Every report that can reach a user carries the fixed disclaimer, whichever path produced it."""

from __future__ import annotations

import pytest

from agents.coordinator import CoordinatorAgent
from agents.coordinator.fakes import (
    ScriptedDiagnostic,
    ScriptedRecommendation,
    diagnostic,
    recommendation,
)
from agents.recommendation_agent.nodes.synthesize import (
    enforce_disclaimer,
    has_required_disclaimer,
)
from backend.tests.fakes import (
    SID,
    WAIT,
    FakeOllama,
    app_client,
    chat_reply,
    make_services,
    make_settings,
)
from shared.schema import DISCLAIMER, FinalReport, SessionStatus

ok_diag = diagnostic(SID, {"ACL": 0.85})
bad_images = diagnostic(SID, {"ACL": 0.7}, decision="escalate", flags=["missing_plane:axial"])

# (diagnostic script, recommendation script, expected status)
MULTI_AGENT_PATHS = {
    "completed": ([ok_diag], [recommendation()], SessionStatus.COMPLETED),
    "partial_no_evidence": ([ok_diag], [recommendation(evidence="no_credible_source_found")], SessionStatus.PARTIAL),
    "partial_recommendation_failed": ([ok_diag], [RuntimeError("down")], SessionStatus.PARTIAL),
    "inconclusive": ([bad_images], [recommendation()], SessionStatus.INCONCLUSIVE),
    "failed": ([RuntimeError("model service down")], [recommendation()], SessionStatus.FAILED),
}


@pytest.mark.parametrize("path", MULTI_AGENT_PATHS)
def test_every_multi_agent_outcome_has_the_disclaimer(tmp_path, path):
    diag, rec, expected = MULTI_AGENT_PATHS[path]
    coordinator = CoordinatorAgent(ScriptedDiagnostic(diag), ScriptedRecommendation(rec),
                                   settings=make_settings(tmp_path), driver="fixed")
    report = coordinator.run(SID).report
    assert report.status == expected
    assert DISCLAIMER in report.disclaimer


def test_coordinator_crash_still_ends_with_a_disclaimer(tmp_path, monkeypatch):
    coordinator = CoordinatorAgent(ScriptedDiagnostic([ok_diag]), ScriptedRecommendation([recommendation()]),
                                   settings=make_settings(tmp_path), driver="fixed")
    monkeypatch.setattr(coordinator, "_run_recommendation", lambda run: 1 / 0)
    report = coordinator.run(SID).report
    assert report.status == SessionStatus.FAILED and DISCLAIMER in report.disclaimer


@pytest.mark.parametrize("replies, missing, expected", [
    ([chat_reply('{"impression": "You have an ACL tear.", "recommendation_text": "Surgery.", "urgency": "urgent"}')],
     set(), "completed"),
    ([chat_reply("not json")], set(), "partial"),
    ([chat_reply("unused")], {SID}, "failed"),
])
def test_every_baseline_outcome_has_the_disclaimer(tmp_path, replies, missing, expected):
    # Even when MedGemma states a diagnosis outright, the report keeps the fixed disclaimer.
    services = make_services(tmp_path, ollama=FakeOllama(replies), missing=missing)
    with app_client(services) as client:
        body = client.post("/baseline", params=WAIT, json={"study_id": SID}).json()
    assert body["status"] == expected
    assert DISCLAIMER in body["final_report"]["disclaimer"]


def test_the_disclaimer_cannot_be_removed_or_replaced():
    for text in ("", "No disclaimer needed.", "   "):
        report = FinalReport(session_id="s", study_id=SID, status=SessionStatus.COMPLETED, disclaimer=text)
        assert DISCLAIMER in report.disclaimer
    copied = FinalReport(session_id="s", study_id=SID, status=SessionStatus.COMPLETED)
    assert DISCLAIMER in FinalReport.model_validate(copied.model_dump() | {"disclaimer": ""}).disclaimer


def test_api_report_endpoint_returns_the_disclaimer(tmp_path):
    with app_client(make_services(tmp_path)) as client:
        sid = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["session_id"]
        assert DISCLAIMER in client.get(f"/sessions/{sid}/report").json()["disclaimer"]


@pytest.mark.parametrize("text", ["Rest and ice.", "You have a torn ACL, get surgery.", ""])
def test_recommendation_text_always_says_may_indicate_and_consult(text):
    fixed, _ = enforce_disclaimer(text)
    assert has_required_disclaimer(fixed)