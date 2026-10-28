"""Backend API tests with scripted agents, a fake Ollama and a fake Model Service. No servers or models needed."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from agents.coordinator.fakes import diagnostic, recommendation
from backend.main import create_app
from backend.model_service_client import ModelServiceClient
from backend.tests.fakes import (
    SID,
    WAIT,
    FakeOllama,
    baseline_json,
    chat_reply,
    make_services,
)
from model_service.api.main import StubBackend
from shared.schema import DISCLAIMER, EvidenceStatus, SessionStatus, UrgencyLevel


@pytest.fixture
def client_for(tmp_path):
    def build(**kwargs) -> TestClient:
        services = make_services(tmp_path, **kwargs)
        client = TestClient(create_app(services=services, configure_logging=False))
        client.__enter__()  # runs the lifespan
        clients.append(client)
        return client

    clients: list[TestClient] = []
    yield build
    for client in clients:
        client.__exit__(None, None, None)


# --- Sessions ---

def test_fixed_session_runs_to_a_complete_report(client_for):
    client = client_for()
    response = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed",
                                                            "patient_metadata": {"age": 30, "symptoms": "twist"}})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == SessionStatus.COMPLETED.value and body["finished"]
    report = body["final_report"]
    assert report["produced_by"] == "multi_agent"
    assert [f["name"] for f in report["findings"]] == ["ACL"]
    assert DISCLAIMER in report["disclaimer"]
    assert response.headers["Location"] == f"/sessions/{body['session_id']}"
    assert body["stats"]["executed_tools"][-1] == "finalize_report"


def test_session_starts_pending_without_wait(client_for):
    client = client_for()
    response = client.post("/sessions", json={"study_id": SID, "driver": "fixed"})
    assert response.status_code in (200, 202)
    assert response.json()["links"]["stream"].endswith("/stream")


def test_get_session_and_report(client_for):
    client = client_for()
    sid = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["session_id"]
    detail = client.get(f"/sessions/{sid}", params={"include_events": True}).json()
    assert detail["events"][0]["message"] == "Session queued"
    assert [e["seq"] for e in detail["events"]] == list(range(len(detail["events"])))
    assert client.get(f"/sessions/{sid}/report").json()["status"] == "completed"
    assert client.get("/sessions").json()[0]["session_id"] == sid


def test_unknown_or_malformed_session_is_404(client_for):
    client = client_for()
    assert client.get("/sessions/doesnotexist").status_code == 404
    assert client.get("/sessions/..%2F..%2Fetc").status_code == 404


def test_events_can_be_polled_incrementally(client_for):
    client = client_for()
    sid = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["session_id"]
    first = client.get(f"/sessions/{sid}/events").json()
    assert first["finished"] and first["events"]
    rest = client.get(f"/sessions/{sid}/events", params={"after": first["next_after"]}).json()
    assert rest["events"] == [] and rest["next_after"] == first["next_after"]


def test_stream_replays_every_event_then_ends(client_for):
    client = client_for()
    sid = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["session_id"]
    n_events = len(client.get(f"/sessions/{sid}", params={"include_events": True}).json()["events"])
    with client.stream("GET", f"/sessions/{sid}/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    assert text.count("event: agent_event") == n_events
    assert text.rstrip().endswith('"status": "completed"}')


def test_stream_resumes_from_last_event_id(client_for):
    client = client_for()
    sid = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["session_id"]
    with client.stream("GET", f"/sessions/{sid}/stream", headers={"Last-Event-ID": "2"}) as response:
        text = "".join(response.iter_text())
    assert "id: 2\n" not in text and "id: 3\n" in text


def test_reupload_session_is_inconclusive(client_for):
    bad = diagnostic(SID, {"ACL": 0.7}, decision="escalate", flags=["missing_plane:axial"])
    client = client_for(diag=[bad])
    body = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()
    assert body["status"] == "inconclusive"
    assert "axial" in body["user_action"]


def test_failed_recommendation_gives_partial(client_for):
    client = client_for(rec=[RuntimeError("index missing")])
    body = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()
    assert body["status"] == "partial"
    assert any("index missing" in e for e in body["errors"])


def test_no_credible_source_is_shown_as_a_gap(client_for):
    client = client_for(rec=[recommendation(evidence=EvidenceStatus.NO_CREDIBLE_SOURCE_FOUND)])
    report = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["final_report"]
    assert report["evidence_status"] == "no_credible_source_found" and report["citations"] == []


def test_session_tokens_count_every_agent(client_for, tmp_path):
    # Qwen Coordinator via tool calls; tokens from the Coordinator must reach the report and the usage table.
    calls = [("run_diagnostic", {"mode": "initial", "reason": "start"}), ("run_recommendation", {"reason": "r"}),
             ("finalize_report", {"summary": "done"})]
    replies = [{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": n,
               "arguments": a}}]}, "prompt_eval_count": 100, "eval_count": 10, "done_reason": "stop"}
               for n, a in calls]
    client = client_for(ollama=FakeOllama(replies))
    body = client.post("/sessions", params=WAIT, json={"study_id": SID}).json()
    assert body["status"] == "completed"
    assert body["usage"]["coordinator"]["calls"] == 3
    assert body["final_report"]["total_tokens"] == body["usage"]["total"]["tokens_in"] + \
        body["usage"]["total"]["tokens_out"] == 330


def test_finished_session_is_read_back_after_restart(tmp_path):
    services = make_services(tmp_path)
    with TestClient(create_app(services=services, configure_logging=False)) as client:
        sid = client.post("/sessions", params=WAIT, json={"study_id": SID, "driver": "fixed"}).json()["session_id"]
    with TestClient(create_app(services=make_services(tmp_path), configure_logging=False)) as client:
        body = client.get(f"/sessions/{sid}", params={"include_events": True}).json()
    assert body["status"] == "completed" and body["final_report"]["findings"][0]["name"] == "ACL"
    assert body["events"][-1]["kind"] == "status"


# --- Baseline ---

def test_baseline_is_one_medgemma_call_with_the_same_report_shape(client_for):
    ollama = FakeOllama([baseline_json("urgent")])
    client = client_for(ollama=ollama)
    body = client.post("/baseline", params=WAIT, json={"study_id": SID}).json()
    report = body["final_report"]
    assert body["status"] == "completed" and report["produced_by"] == "baseline"
    assert report["urgency"] == UrgencyLevel.URGENT.value
    assert report["evidence_status"] is None and report["citations"] == []
    assert DISCLAIMER in report["disclaimer"]
    assert len(ollama.bodies) == 1 and ollama.bodies[0]["model"] == "medgemma:4b"
    assert "format" in ollama.bodies[0] and "tools" not in ollama.bodies[0]
    assert report["total_tokens"] == 380
    # Findings follow the same threshold rule as the multi-agent report
    scores = StubBackend().predict(SID).findings
    assert [f["name"] for f in report["findings"]] == sorted(
        [n for n, p in scores.items() if p >= 0.5], key=lambda n: -scores[n])


def test_baseline_invalid_json_gives_partial_with_findings(client_for):
    client = client_for(ollama=FakeOllama([chat_reply("not json")]))
    body = client.post("/baseline", params=WAIT, json={"study_id": SID}).json()
    assert body["status"] == "partial"
    assert body["final_report"]["recommendation_text"] == ""
    assert body["usage"]["baseline"]["calls"] == 2  # one validation retry


def test_baseline_unknown_study_fails(client_for):
    client = client_for(missing={SID})
    body = client.post("/baseline", params=WAIT, json={"study_id": SID}).json()
    assert body["status"] == "failed"
    assert any("not available" in e for e in body["errors"])


def test_baseline_prompt_carries_scores_flags_and_patient(client_for):
    ollama = FakeOllama([baseline_json()])
    client = client_for(ollama=ollama)
    client.post("/baseline", params=WAIT, json={"study_id": SID, "patient_metadata": {"symptoms": "locking"}})
    prompt = ollama.bodies[0]["messages"][1]["content"]
    assert "ACL:" in prompt and "Image quality flags" in prompt and "locking" in prompt


# --- Health and validation ---

def test_health_reports_wiring_and_dependencies(client_for):
    client = client_for()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agents"]["diagnostic"] == "agent"
    assert body["model_service"]["status"] == "ok" and body["ollama"]["ready"] is True


def test_health_is_degraded_when_model_service_is_down(tmp_path):
    services = make_services(tmp_path)
    down = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    services.model_service = ModelServiceClient(services.settings, transport=down)
    with TestClient(create_app(services=services, configure_logging=False)) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded" and body["model_service"]["status"] == "unreachable"


def test_bad_request_bodies_are_rejected(client_for):
    client = client_for()
    assert client.post("/sessions", json={"study_id": ""}).status_code == 422
    assert client.post("/sessions", json={"study_id": SID, "driver": "magic"}).status_code == 422
    assert client.post("/sessions", json={"study_id": SID, "patient_metadata": {"age": 400}}).status_code == 422