# Tests the Coordinator loop end to end with scripted agents and a fake Ollama. No model, GPU or dataset needed.

import json

import httpx
import pytest
from coordinator_fakes import ScriptedOllama, make_settings, reply

from agents.coordinator import CoordinatorAgent
from agents.coordinator.fakes import (
    ScriptedDiagnostic,
    ScriptedRecommendation,
    Slow,
    diagnostic,
    recommendation,
)
from agents.coordinator.report import NO_EVIDENCE_TEXT
from shared.logging_setup import TraceWriter
from shared.schema import (
    DISCLAIMER,
    AgentName,
    DiagnosticDecision,
    EvidenceStatus,
    PatientMetadata,
    SessionStatus,
)

SID = "1.2.826.0.1.3680043.8.498.10001"
DIAG = "run_diagnostic"
REC = "run_recommendation"
FIN = "finalize_report"
REUP = "request_reupload"


def _agents(diags, recs):
    return ScriptedDiagnostic(diags), ScriptedRecommendation(recs)


def _coordinator(tmp_path, diag, rec, llm=None, driver="tools", **settings):
    s = make_settings(tmp_path, **settings)
    return CoordinatorAgent(diagnostic=diag, recommendation=rec, llm=llm.client(s) if llm else None, settings=s,
                            driver=driver)


# --- Happy path ---

def test_fixed_driver_happy_path_produces_a_complete_report_and_trace(tmp_path):
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, driver="fixed").run(SID, session_id="s1")
    report = result.report
    assert report.status == SessionStatus.COMPLETED
    assert [f.name for f in report.findings] == ["ACL"]
    assert report.recommendation_text and report.citations and report.disclaimer == DISCLAIMER
    assert result.executed_tools == [DIAG, REC, FIN]
    assert result.state.is_finished and result.state.final_report == report
    # Trace file has every event.
    written = TraceWriter("s1", str(tmp_path)).read()
    assert [e.seq for e in written] == list(range(len(result.state.events)))


def test_qwen_tool_calls_drive_the_session(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {"mode": "initial", "reason": "start"})]),
                          reply([(REC, {})]),
                          reply([(FIN, {"summary": "Both agents agree on an ACL tear."})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert result.report.status == SessionStatus.COMPLETED
    assert result.stats.valid_call_rate == 1.0 and result.override_rate == 0.0
    assert result.stats.llm_calls == 3 and result.stats.tokens_in == 360
    assert "Coordinator note: Both agents agree" in result.report.coordination_summary
    # Tool results reach the model.
    tool_messages = [m for m in llm.bodies[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[-1]["content"])["allowed_tools"] == [REC, FIN]
    # study_id is never a tool argument.
    tool_params = {t["function"]["name"]: t["function"]["parameters"] for t in llm.bodies[0]["tools"]}
    assert set(tool_params) == {DIAG, REC, REUP, FIN}
    assert all("study_id" not in p.get("properties", {}) for p in tool_params.values())


def test_recommendation_request_carries_impression_and_patient_metadata(tmp_path):
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9}, impression="Likely ACL tear")], [recommendation()])
    patient = PatientMetadata(age=24, sex="female", symptoms="twisting injury")
    _coordinator(tmp_path, diag, rec, driver="fixed").run(SID, patient)
    sent = rec.requests[0]
    assert sent.impression == "Likely ACL tear" and sent.patient_metadata == patient
    assert sent.task_type == "initial" and sent.finding["ACL"] == 0.9


# --- Guardrails acting on LLM choices ---

def test_finalize_before_any_work_is_refused(tmp_path):
    llm = ScriptedOllama([reply([(FIN, {})]), reply([(DIAG, {})]), reply([(REC, {})]), reply([(FIN, {})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert result.report.status == SessionStatus.COMPLETED
    assert result.stats.refused_calls == 1
    assert result.decisions[0].outcome == "refused"
    assert result.override_rate == pytest.approx(1 / 4)


def test_invalid_arguments_are_sent_back_to_the_model(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {"mode": "recheck", "focus_findings": ["Knee cancer"]})]),
                          reply([(DIAG, {"mode": "initial"})]), reply([(REC, {})]), reply([(FIN, {})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert result.stats.proposed_calls == 4 and result.stats.valid_calls == 3
    feedback = llm.bodies[1]["messages"][-1]
    assert feedback["role"] == "tool" and "unknown finding names" in feedback["content"]
    assert result.report.status == SessionStatus.COMPLETED


def test_finding_names_are_matched_case_insensitively(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {})]), reply([(REC, {})]),
                          reply([(DIAG, {"mode": "recheck", "focus_findings": ["fracture"], "reason": "odd"})]),
                          reply([(REC, {})]), reply([(FIN, {})])])
    diag, rec = _agents([diagnostic(SID, {"Fracture": 0.8}), diagnostic(SID, {"Fracture": 0.2})],
                        [recommendation(recheck=["Fracture"]), recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert diag.requests[1].focus_findings == ["Fracture"]
    assert diag.requests[1].task_type == "recheck: odd"
    assert result.report.recheck_rounds == 1 and not result.report.disagreements


def test_recheck_limit_is_enforced_and_dispute_is_reported(tmp_path):
    recheck = (DIAG, {"mode": "recheck", "focus_findings": ["MCL"], "reason": "senior doubts MCL"})
    llm = ScriptedOllama([reply([(DIAG, {})]), reply([(REC, {})]),
                          reply([recheck]), reply([(REC, {})]),
                          reply([recheck]), reply([(REC, {})]),
                          reply([recheck]),  # third recheck: refused
                          reply([(FIN, {"unresolved": ["MCL still disputed"]})])])
    diag, rec = _agents([diagnostic(SID, {"MCL": 0.7})], [recommendation(concerns=["MCL odd"], recheck=["MCL"])])
    result = _coordinator(tmp_path, diag, rec, llm, max_recheck_rounds=2).run(SID)
    assert result.report.recheck_rounds == 2 and diag.calls == 3
    assert result.stats.refused_calls == 1
    assert any("MCL" in d for d in result.report.disagreements)
    assert "Coordinator note: MCL still disputed" in result.report.disagreements
    mcl = next(f for f in result.report.findings if f.name == "MCL")
    assert mcl.low_confidence  # still disputed


def test_finalising_over_an_open_senior_request_needs_two_attempts(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {})]), reply([(REC, {})]), reply([(FIN, {})]), reply([(FIN, {})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.7})], [recommendation(recheck=["ACL"])])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert [d.outcome for d in result.decisions] == ["executed", "executed", "refused", "executed"]
    assert "Senior review still questions ACL after 0 recheck round(s)." in result.report.disagreements


def test_reupload_refused_when_images_were_fine(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {})]), reply([(REUP, {"user_action": "upload again"})]),
                          reply([(REC, {})]), reply([(FIN, {})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert result.report.status == SessionStatus.COMPLETED and REUP not in result.executed_tools


# --- The LLM misbehaving or failing ---

def test_text_replies_without_tool_calls_hand_one_step_to_the_fixed_policy(tmp_path):
    llm = ScriptedOllama([reply(content="I think it is an ACL tear."), reply(content="Still thinking."),
                          reply([(REC, {})]), reply([(FIN, {})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert result.stats.no_tool_replies == 2 and result.stats.fallback_steps == 1
    assert result.decisions[0].source == "fallback" and result.decisions[0].tool == DIAG
    assert result.report.status == SessionStatus.COMPLETED


def test_ollama_down_falls_back_to_fixed_policy(tmp_path):
    llm = ScriptedOllama([httpx.ConnectError("refused")])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert result.report.status == SessionStatus.COMPLETED
    assert result.stats.llm_errors == 1 and len(llm.bodies) == 1
    assert any("LLM unavailable" in e for e in result.report.errors)


def test_step_limit_forces_an_end_state(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {})]), reply([(REC, {})]), reply([(REC, {})])])  # refused repeats
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm, max_coordinator_steps=4).run(SID)
    assert result.stats.forced_stops == 1 and result.decisions[-1].source == "guardrail"
    assert result.state.is_finished and result.report.status == SessionStatus.COMPLETED


def test_extra_calls_after_the_end_are_ignored(tmp_path):
    llm = ScriptedOllama([reply([(DIAG, {})]), reply([(REC, {})]), reply([(FIN, {}), (DIAG, {})])])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm).run(SID)
    assert diag.calls == 1 and result.executed_tools[-1] == FIN


def test_json_driver_for_models_without_tool_calling(tmp_path):
    def action(tool, arguments=None):
        return reply(content=json.dumps({"tool": tool, "arguments": arguments or {}, "reason": "next step"}))

    llm = ScriptedOllama([action(DIAG, {"mode": "initial"}), action(REC), action(FIN)])
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, llm, driver="json", coordinator_model="medgemma:4b").run(SID)
    assert result.report.status == SessionStatus.COMPLETED and result.stats.valid_call_rate == 1.0
    assert "tools" not in llm.bodies[0] and llm.bodies[0]["format"]["properties"]["tool"]["enum"]


# --- Agent failures and conflicting output ---

def test_unusable_images_lead_to_reupload_without_recommendation(tmp_path):
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.6}, decision=DiagnosticDecision.ESCALATE,
                                    flags=["missing_plane:axial"])], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, driver="fixed").run(SID)
    assert result.report.status == SessionStatus.INCONCLUSIVE
    assert "axial" in result.report.user_action and rec.calls == 0
    assert result.report.findings == [] and result.report.disclaimer == DISCLAIMER


def test_no_credible_source_is_shown_as_a_gap(tmp_path):
    diag, rec = _agents([diagnostic(SID, {"Synovitis": 0.7})],
                        [recommendation("Invented advice", evidence=EvidenceStatus.NO_CREDIBLE_SOURCE_FOUND,
                                        citations=["https://example.com"])])
    report = _coordinator(tmp_path, diag, rec, driver="fixed").run(SID).report
    assert report.status == SessionStatus.PARTIAL
    assert report.recommendation_text == NO_EVIDENCE_TEXT and report.citations == []


def test_diagnostic_timeout_is_recorded_and_retried(tmp_path):
    diag, rec = _agents([Slow(0.5, diagnostic(SID, {"ACL": 0.9})), diagnostic(SID, {"ACL": 0.9})],
                        [recommendation()])
    result = _coordinator(tmp_path, diag, rec, driver="fixed", tool_timeout_s=0.2).run(SID)
    assert result.report.status == SessionStatus.COMPLETED
    assert any("ToolTimeoutError" in e for e in result.report.errors)


def test_result_for_another_study_is_rejected(tmp_path):
    diag, rec = _agents([diagnostic("other-study", {"ACL": 0.9})], [recommendation()])
    result = _coordinator(tmp_path, diag, rec, driver="fixed").run(SID)
    assert result.report.status == SessionStatus.FAILED and result.state.diagnostic_results == []
    assert rec.calls == 0


def test_recommendation_down_gives_partial_report_with_findings(tmp_path):
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [RuntimeError("down")])
    report = _coordinator(tmp_path, diag, rec, driver="fixed").run(SID).report
    assert report.status == SessionStatus.PARTIAL and [f.name for f in report.findings] == ["ACL"]
    assert report.recommendation_text == "" and len(report.errors) == 2


def test_events_are_streamed_to_the_callback(tmp_path):
    seen = []
    diag, rec = _agents([diagnostic(SID, {"ACL": 0.9})], [recommendation()])
    s = make_settings(tmp_path)
    CoordinatorAgent(diagnostic=diag, recommendation=rec, settings=s, driver="fixed",
                     on_event=seen.append, write_traces=False).run(SID)
    assert seen[0].kind == "status" and seen[-1].data["status"] == "completed"
    assert {e.agent for e in seen} >= {AgentName.COORDINATOR, AgentName.DIAGNOSTIC, AgentName.RECOMMENDATION}


def test_llm_driver_requires_a_client(tmp_path):
    with pytest.raises(ValueError):
        CoordinatorAgent(diagnostic=None, recommendation=None, settings=make_settings(tmp_path), driver="tools")


# --- Interim Model Service adapter ---

def test_interim_adapter_turns_a_prediction_into_a_diagnostic_response():
    from agents.coordinator.runners import prediction_to_diagnostic
    from model_service.api.main import StubBackend
    from shared.schema import DiagnosticAgentRequest

    prediction = StubBackend().predict(SID)
    result = prediction_to_diagnostic(prediction, DiagnosticAgentRequest(study_id=SID))
    assert result.study_id == SID and result.findings == prediction.findings
    assert all(0 <= c <= 1 for c in result.per_finding_confidence.values())
    missing = prediction.model_copy(update={"per_plane": prediction.per_plane[:2],
                                            "quality_flags": ["missing_plane:axial"]})
    assert prediction_to_diagnostic(missing, DiagnosticAgentRequest(study_id=SID)).decision == "escalate"