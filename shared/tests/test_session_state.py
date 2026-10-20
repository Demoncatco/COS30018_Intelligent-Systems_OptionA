import pytest
from pydantic import BaseModel, ValidationError

from shared import schema
from shared.schema import (
    FINDINGS,
    AgentEvent,
    AgentName,
    CoordinatorRequest,
    DiagnosticAgentResponse,
    PatientMetadata,
    PredictionResponse,
    RecommendationResponse,
    ReplayPredictionsFile,
    SessionState,
    SessionStatus,
)

SCORES = {name: 0.1 for name in FINDINGS}


def _diagnostic(**overrides) -> DiagnosticAgentResponse:
    fields = {"study_id": "x", "findings": SCORES, "per_finding_confidence": SCORES, "overall_confidence": 0.8,
              "decision": "finalize", "retry_count": 0, "laterality": "left"}
    return DiagnosticAgentResponse(**(fields | overrides))


def _prediction(study_id: str, model_version: str = "knee_v1") -> PredictionResponse:
    return PredictionResponse(study_id=study_id, findings=SCORES, per_plane=[], laterality="unknown",
                              laterality_source="unavailable", model_version=model_version, latency_ms=1.0)


# Regression test for the tuple bug: Field(description=("a", "b"),) silently became a tuple. 

def test_every_field_description_is_a_string():
    for name, obj in vars(schema).items():
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            for field_name, field in obj.model_fields.items():
                assert field.description is None or isinstance(field.description, str), f"{name}.{field_name}"


# Patient metadata reaches the Recommendation Agent 

def test_coordinator_request_carries_patient_metadata_and_stays_backwards_compatible():
    old_style = CoordinatorRequest(finding=SCORES, confidence=0.9, task_type="initial")
    assert old_style.patient_metadata == PatientMetadata()
    new_style = CoordinatorRequest(finding=SCORES, confidence=0.9, task_type="initial", impression="Likely ACL tear",
                                   patient_metadata={"age": 24, "sex": "female", "symptoms": "twisting injury"})
    assert new_style.patient_metadata.age == 24 and new_style.impression == "Likely ACL tear"
    with pytest.raises(ValidationError):
        CoordinatorRequest(finding=SCORES, confidence=0.9, task_type="initial", patient_metadata={"age": 200})


# --- SessionState ---

def test_events_are_numbered_in_order_and_update_the_clock():
    state = SessionState(session_id="s1", study_id="x")
    first = state.add_event(AgentName.COORDINATOR, "decision", "call run_diagnostic")
    second = state.add_event("diagnostic", "tool_call", "POST /predict", {"study_id": "x"}, latency_ms=12.0)
    assert [first.seq, second.seq] == [0, 1]
    assert second.session_id == "s1" and second.data == {"study_id": "x"}
    assert state.updated_at == second.timestamp


def test_totals_add_up_tokens_and_latency_over_events():
    state = SessionState(session_id="s1", study_id="x")
    state.add_event("coordinator", "decision", "a", tokens_in=100, tokens_out=20, latency_ms=50.0)
    state.add_event("recommendation", "message", "b", tokens_in=300, tokens_out=80, latency_ms=150.0)
    state.add_event("model_service", "tool_result", "c")
    assert state.total_tokens == 500
    assert state.total_latency_ms == 200.0


def test_latest_results_and_recheck_budget():
    state = SessionState(session_id="s1", study_id="x")
    assert state.latest_diagnostic is None and state.latest_recommendation is None
    state.diagnostic_results.append(_diagnostic(overall_confidence=0.5))
    state.diagnostic_results.append(_diagnostic(overall_confidence=0.9))
    state.recommendation_results.append(
        RecommendationResponse(recommendation_text="may indicate", urgency="routine", evidence_status="grounded_local"))
    assert state.latest_diagnostic.overall_confidence == 0.9
    assert state.latest_recommendation.recommendation_text == "may indicate"
    assert state.can_recheck(max_rounds=2)
    state.recheck_rounds = 2
    assert not state.can_recheck(max_rounds=2)


def test_errors_are_recorded_in_both_the_error_list_and_the_trace():
    state = SessionState(session_id="s1", study_id="x")
    event = state.add_error(AgentName.RECOMMENDATION, "Tavily timed out", {"attempt": 2})
    assert state.errors == ["recommendation: Tavily timed out"]
    assert event.kind == "error" and state.events[-1] is event


def test_status_changes_are_traced_and_a_finished_session_cannot_restart():
    state = SessionState(session_id="s1", study_id="x")
    state.set_status(SessionStatus.RUNNING)
    state.set_status(SessionStatus.PARTIAL, "no credible source found")
    assert state.is_finished
    assert [e.data["status"] for e in state.events if e.kind == "status"] == ["running", "partial"]
    with pytest.raises(ValueError):
        state.set_status(SessionStatus.RUNNING)


def test_session_state_round_trips_through_json():
    state = SessionState(session_id="s1", study_id="x", patient_metadata={"symptoms": "swelling"})
    state.diagnostic_results.append(_diagnostic(impression="Effusion likely"))
    state.add_event("diagnostic", "message", "done")
    again = SessionState.model_validate_json(state.model_dump_json())
    assert again == state


def test_events_from_another_session_are_rejected():
    stray = AgentEvent(session_id="other", seq=0, timestamp=0.0, agent="coordinator", kind="status", message="x")
    with pytest.raises(ValidationError):
        SessionState(session_id="s1", study_id="x", events=[stray])


# --- Replay file contract ---

def test_replay_file_accepts_real_model_output():
    replay = ReplayPredictionsFile(model_version="knee_v1", predictions=[_prediction("a"), _prediction("b")])
    assert replay.format_version == 1 and len(replay.predictions) == 2


@pytest.mark.parametrize("version, predictions", [
    ("knee_v1", [_prediction("a"), _prediction("a")]),             # duplicate study
    ("knee_v1", [_prediction("a"), _prediction("b", "knee_v0")]),  # mixed checkpoints
    ("stub-0.1.0", [_prediction("a", "stub-0.1.0")]),              # random scores passed off as a model
])
def test_replay_file_rejects_bad_files(version, predictions):
    with pytest.raises(ValidationError):
        ReplayPredictionsFile(model_version=version, predictions=predictions)