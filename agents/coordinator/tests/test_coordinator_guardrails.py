# Unit tests for the Coordinator's guardrails and fixed policy, on hand-built states.

import pytest
from coordinator_fakes import make_settings
from pydantic import ValidationError

from agents.coordinator import guardrails as g
from agents.coordinator.fakes import diagnostic, recommendation
from agents.coordinator.tools import (
    TOOLS,
    FinalizeReportArgs,
    RequestReuploadArgs,
    RunDiagnosticArgs,
    RunRecommendationArgs,
)
from shared.schema import DiagnosticDecision, SessionState

SID = "study-1"


def _ctx(tmp_path, diags=(), recs=(), reviewed=None, **settings) -> g.RunContext:
    state = SessionState(session_id="s", study_id=SID)
    state.diagnostic_results.extend(diags)
    state.recommendation_results.extend(recs)
    return g.RunContext(state=state, settings=make_settings(tmp_path, **settings), reviewed_index=reviewed)


def _recheck(*names, reason="x"):
    return RunDiagnosticArgs(mode="recheck", focus_findings=list(names), reason=reason)


# --- Tool argument models ---

def test_recheck_needs_focus_and_known_names():
    with pytest.raises(ValidationError):
        RunDiagnosticArgs(mode="recheck")
    with pytest.raises(ValidationError, match="unknown finding names"):
        _recheck("Knee cancer")
    assert _recheck("bAKER'S", "acl").focus_findings == ["ACL", "Baker's"]  # canonical spelling and order


def test_long_text_is_truncated_not_rejected():
    assert len(FinalizeReportArgs(summary="x" * 5000).summary) == 1000
    assert FinalizeReportArgs(unresolved="one string").unresolved == ["one string"]


def test_tool_schemas_are_flat_and_named():
    assert [t["function"]["name"] for t in TOOLS] == ["run_diagnostic", "run_recommendation", "request_reupload",
                                                       "finalize_report"]
    assert all("$defs" not in str(t) for t in TOOLS)


# --- Ordering rules ---

def test_only_diagnostic_is_allowed_at_the_start(tmp_path):
    ctx = _ctx(tmp_path)
    assert g.allowed_tools(ctx) == ["run_diagnostic"]
    assert not g.check_call(ctx, "run_recommendation", RunRecommendationArgs()).allowed
    assert not g.check_call(ctx, "finalize_report", FinalizeReportArgs()).allowed
    assert g.fallback_action(ctx)[0] == "run_diagnostic"


def test_second_initial_diagnostic_is_refused(tmp_path):
    ctx = _ctx(tmp_path, [diagnostic(SID, {"ACL": 0.9})])
    assert not g.check_call(ctx, "run_diagnostic", RunDiagnosticArgs()).allowed


def test_recommendation_refused_on_unusable_images_and_reupload_suggested(tmp_path):
    bad = diagnostic(SID, decision=DiagnosticDecision.ESCALATE, flags=["missing_plane:axial", "low_slice_count"])
    ctx = _ctx(tmp_path, [bad])
    assert g.needs_reupload(bad)
    assert not g.check_call(ctx, "run_recommendation", RunRecommendationArgs()).allowed
    assert not g.check_call(ctx, "finalize_report", FinalizeReportArgs()).allowed
    name, args = g.fallback_action(ctx)
    assert name == "request_reupload" and "axial" in args.user_action and "15 slices" in args.user_action


def test_uncertain_result_without_quality_flags_is_not_a_reupload(tmp_path):
    ctx = _ctx(tmp_path, [diagnostic(SID, {"ACL": 0.6}, decision=DiagnosticDecision.RETRY)])
    assert not g.needs_reupload(ctx.latest)
    assert not g.check_call(ctx, "request_reupload", RequestReuploadArgs(user_action="again")).allowed
    assert g.fallback_action(ctx)[0] == "run_recommendation"


# --- Junior / senior rule ---

def test_senior_requests_define_the_recheck_focus(tmp_path):
    d = diagnostic(SID, {"ACL": 0.9, "Fracture": 0.7})
    ctx = _ctx(tmp_path, [d], [recommendation(recheck=["Fracture"])], reviewed=0)
    assert g.recheck_candidates(ctx) == ["Fracture"]
    verdict = g.check_call(ctx, "run_diagnostic", _recheck("Fracture", "ACL"))
    assert verdict.allowed and verdict.args.focus_findings == ["Fracture"] and "dropped" in verdict.note
    assert not g.check_call(ctx, "run_diagnostic", _recheck("ACL")).allowed
    name, args = g.fallback_action(ctx)
    assert name == "run_diagnostic" and args.focus_findings == ["Fracture"]


def test_before_senior_review_low_confidence_findings_may_be_rechecked(tmp_path):
    d = diagnostic(SID, {"MCL": 0.7}, confidence={"MCL": 0.4})
    ctx = _ctx(tmp_path, [d])
    assert g.recheck_candidates(ctx) == ["MCL"]
    assert g.check_call(ctx, "run_diagnostic", _recheck("MCL")).allowed


def test_senior_must_review_a_recheck_before_another(tmp_path):
    d = diagnostic(SID, {"MCL": 0.7})
    ctx = _ctx(tmp_path, [d, d], [recommendation(recheck=["MCL"])], reviewed=0)
    ctx.state.recheck_rounds = 1
    verdict = g.check_call(ctx, "run_diagnostic", _recheck("MCL"))
    assert not verdict.allowed and "run_recommendation" in verdict.message
    assert g.fallback_action(ctx)[0] == "run_recommendation"


def test_recheck_limit(tmp_path):
    d = diagnostic(SID, {"MCL": 0.7})
    ctx = _ctx(tmp_path, [d], [recommendation(recheck=["MCL"])], reviewed=0, max_recheck_rounds=2)
    ctx.state.recheck_rounds = 2
    verdict = g.check_call(ctx, "run_diagnostic", _recheck("MCL"))
    assert not verdict.allowed and "limit" in verdict.message
    assert g.fallback_action(ctx)[0] == "finalize_report"


def test_finalize_over_open_request_is_a_soft_refusal_given_once(tmp_path):
    ctx = _ctx(tmp_path, [diagnostic(SID, {"ACL": 0.7})], [recommendation(recheck=["ACL"])], reviewed=0)
    first = g.check_call(ctx, "finalize_report", FinalizeReportArgs())
    second = g.check_call(ctx, "finalize_report", FinalizeReportArgs())
    assert (first.allowed, first.soft) == (False, True)
    assert second.allowed and "after warning" in second.note


# --- Failures ---

def test_diagnostic_failures_allow_one_retry_then_stop(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.diagnostic_failures = 1
    assert g.fallback_action(ctx)[0] == "run_diagnostic"
    assert g.check_call(ctx, "request_reupload", RequestReuploadArgs(user_action="again")).allowed
    ctx.diagnostic_failures = 2
    assert not g.check_call(ctx, "run_diagnostic", RunDiagnosticArgs()).allowed
    assert g.fallback_action(ctx)[0] == "finalize_report"


def test_recommendation_failures_allow_one_retry_then_finalize(tmp_path):
    ctx = _ctx(tmp_path, [diagnostic(SID, {"ACL": 0.9})])
    ctx.recommendation_failures = 2
    assert not g.check_call(ctx, "run_recommendation", RunRecommendationArgs()).allowed
    assert g.fallback_action(ctx)[0] == "finalize_report"
    assert g.check_call(ctx, "finalize_report", FinalizeReportArgs()).allowed  # no soft warning


def test_forced_stop_never_calls_another_agent(tmp_path):
    assert g.forced_stop_action(_ctx(tmp_path, [diagnostic(SID)]))[0] == "finalize_report"
    bad = diagnostic(SID, decision=DiagnosticDecision.ESCALATE, flags=["low_slice_count"])
    assert g.forced_stop_action(_ctx(tmp_path, [bad]))[0] == "request_reupload"