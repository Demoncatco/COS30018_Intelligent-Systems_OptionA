"""Upload API tests: zip/loose-file uploads, slice previews, and a full session on an uploaded scan (stub scores)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from agents.coordinator.fakes import ScriptedRecommendation, recommendation
from agents.coordinator.runners import ModelServiceDiagnostic
from agents.coordinator.wiring import Runners
from backend.main import create_app
from backend.model_service_client import ModelServiceClient
from backend.sessions import Services
from backend.tests.fakes import WAIT, FakeOllama, baseline_json, make_settings
from model_service.api.main import create_app as create_model_service
from model_service.tests.fake_upload import fake_study, zipped
from shared.llm import LLMClient
from shared.schema import FINDINGS


def _model_service_transport(settings) -> httpx.MockTransport:
    # Routes httpx calls to a real stub-mode Model Service app reading the same UPLOAD_DIR.
    service = TestClient(create_model_service(settings))

    def handler(request: httpx.Request) -> httpx.Response:
        answer = service.request(request.method, request.url.path, content=request.content,
                                 headers={"content-type": "application/json"})
        return httpx.Response(answer.status_code, content=answer.content, headers=answer.headers)

    return httpx.MockTransport(handler)


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path, upload_dir=str(tmp_path / "uploads"), model_mode="stub")
    transport = _model_service_transport(settings)
    llm = LLMClient(settings, transport=httpx.MockTransport(FakeOllama([baseline_json()])))
    runners = Runners(ModelServiceDiagnostic(settings, transport=transport),
                      ScriptedRecommendation([recommendation()] * 3), "interim", "agent", ["test"])
    services = Services(settings, llm, runners, ModelServiceClient(settings, transport=transport))
    with TestClient(create_app(services=services, configure_logging=False)) as test_client:
        yield test_client


def _upload(client, files) -> httpx.Response:
    return client.post("/uploads", files=[("files", (name, data, "application/octet-stream")) for name, data in files])


def test_zip_upload_then_session_gives_scores_for_every_finding(client):
    response = _upload(client, [("knee.zip", zipped(fake_study()))])
    assert response.status_code == 201, response.text
    upload = response.json()
    assert all(p["present"] for p in upload["planes"])
    assert client.get(f"/uploads/{upload['upload_id']}").json() == upload

    session = client.post("/sessions", params=WAIT, json={"study_id": upload["upload_id"], "driver": "fixed"}).json()
    assert session["finished"] and session["status"] in {"completed", "partial"}
    assert set(session["scores"]) == set(FINDINGS)
    assert session["final_report"]["study_id"] == upload["upload_id"]


def test_baseline_on_an_upload_also_reports_scores(client):
    upload = _upload(client, fake_study()).json()
    session = client.post("/baseline", params=WAIT, json={"study_id": upload["upload_id"]}).json()
    assert session["finished"] and set(session["scores"]) == set(FINDINGS)


def test_slice_previews(client):
    upload = _upload(client, fake_study()).json()
    png = client.get(f"/uploads/{upload['upload_id']}/slices/coronal/0.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/uploads/{upload['upload_id']}/slices/coronal/20.png").status_code == 404
    assert client.get(f"/uploads/{upload['upload_id']}/slices/oblique/0.png").status_code == 404


def test_non_dicom_upload_is_refused_with_advice(client):
    response = _upload(client, [("knee.png", b"\x89PNG\r\n\x1a\n not really")])
    assert response.status_code == 422 and "DICOM" in response.json()["detail"]


def test_session_for_a_deleted_upload_is_404(client):
    upload = _upload(client, fake_study()).json()
    assert client.delete(f"/uploads/{upload['upload_id']}").status_code == 204
    response = client.post("/sessions", json={"study_id": upload["upload_id"]})
    assert response.status_code == 404 and "upload" in response.json()["detail"]


def test_unknown_upload_id_is_404(client):
    assert client.get("/uploads/upl-000000000000").status_code == 404
    assert client.get("/uploads/../etc").status_code == 404