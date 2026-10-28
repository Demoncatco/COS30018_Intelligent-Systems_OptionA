"""Every tool call is bounded: slow agents, a hung Model Service or a slow LLM end the step, not the session."""

from __future__ import annotations

import time

import httpx

from agents.coordinator.fakes import Slow, diagnostic, recommendation
from backend.model_service_client import ModelServiceClient
from backend.tests.fakes import SID, WAIT, app_client, make_services
from shared.config import Settings

FAST = {"tool_timeout_s": 0.3}


def _run(services, path="/sessions", body=None):
    started = time.perf_counter()
    with app_client(services) as client:
        response = client.post(path, params=WAIT, json=body or {"study_id": SID, "driver": "fixed"}).json()
    return response, time.perf_counter() - started


def test_slow_diagnostic_times_out_and_the_session_fails_cleanly(tmp_path):
    services = make_services(tmp_path, diag=[Slow(3, diagnostic(SID, {"ACL": 0.9}))], **FAST)
    body, elapsed = _run(services)
    assert body["status"] == "failed"
    assert any("TOOL_TIMEOUT_S" in e for e in body["errors"])
    assert elapsed < 2.5  # two 0.3 s attempts, not two 3 s waits


def test_slow_recommendation_times_out_and_the_report_is_partial(tmp_path):
    services = make_services(tmp_path, rec=[Slow(3, recommendation())], **FAST)
    body, elapsed = _run(services)
    assert body["status"] == "partial" and body["final_report"]["findings"]
    assert any("TOOL_TIMEOUT_S" in e for e in body["errors"])
    assert elapsed < 2.5


def test_hung_model_service_fails_the_baseline_with_a_timeout(tmp_path):
    services = make_services(tmp_path)

    def hang(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    services.scorer = ModelServiceClient(services.settings, transport=httpx.MockTransport(hang))
    body, _ = _run(services, "/baseline", {"study_id": SID})
    assert body["status"] == "failed"
    assert any("ReadTimeout" in e for e in body["errors"])


def test_model_service_client_uses_the_tool_timeout(tmp_path):
    settings = Settings(_env_file=None, log_dir=str(tmp_path), tool_timeout_s=7)
    client = ModelServiceClient(settings)
    assert client._http.timeout.read == 7


def test_health_check_does_not_hang_on_a_dead_model_service(tmp_path):
    services = make_services(tmp_path)
    services.model_service = ModelServiceClient(services.settings, transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout("timed out", request=request))))
    with app_client(services) as client:
        body = client.get("/health").json()
    assert body["model_service"]["status"] == "unreachable" and body["status"] == "degraded"