"""Fake Ollama, Model Service and agent services for backend and engineering-requirement tests."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from agents.coordinator.fakes import (
    ScriptedDiagnostic,
    ScriptedRecommendation,
    diagnostic,
    recommendation,
)
from agents.coordinator.wiring import Runners
from backend.main import create_app
from backend.model_service_client import ModelServiceClient
from backend.sessions import Services
from model_service.api.main import StubBackend
from shared.config import Settings
from shared.llm import LLMClient

SID = "1.2.826.0.1.3680043.8.498.10001"
WAIT = {"wait_s": 10}


def make_settings(tmp_path, **overrides) -> Settings:
    values = {"log_dir": str(tmp_path), "llm_http_retries": 0, "tool_timeout_s": 2.0, "llm_json_retries": 1} | overrides
    return Settings(_env_file=None, **values)


class FakeOllama:
    """Routes Ollama endpoints; /api/chat answers come from `chat` in order (the last repeats)."""

    def __init__(self, chat: list | None = None) -> None:
        self.chat = chat or []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            names = ["medgemma:4b", "qwen3:8b"]
            return httpx.Response(200, json={"models": [{"name": n, "digest": "abc123def456789"} for n in names]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        self.bodies.append(json.loads(request.content))
        item = self.chat[min(len(self.bodies) - 1, len(self.chat) - 1)]
        return httpx.Response(200, json=item)


def chat_reply(content: str, tokens=(300, 80)) -> dict:
    return {"message": {"role": "assistant", "content": content}, "prompt_eval_count": tokens[0],
            "eval_count": tokens[1], "done_reason": "stop"}


def baseline_json(urgency: str = "see_doctor_soon") -> dict:
    return chat_reply(json.dumps({"impression": "Scores may indicate an ACL injury.",
                                  "recommendation_text": "This may indicate a ligament injury; see a clinician.",
                                  "urgency": urgency}))


def model_service(missing: set[str] = frozenset()) -> httpx.MockTransport:
    stub = StubBackend()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "mode": "stub", "model_version": stub.model_version})
        if request.url.path == "/studies":
            return httpx.Response(200, json={"mode": "stub", "any_study_id_accepted": True, "study_ids": []})
        study_id = json.loads(request.content)["study_id"]
        if study_id in missing:
            return httpx.Response(404, json={"detail": "not available"})
        return httpx.Response(200, json=stub.predict(study_id).model_dump(mode="json"))

    return httpx.MockTransport(handler)


def make_services(tmp_path, diag=None, rec=None, ollama: FakeOllama | None = None, missing=frozenset(),
                  **overrides) -> Services:
    settings = make_settings(tmp_path, **overrides)
    llm = LLMClient(settings, transport=httpx.MockTransport(ollama or FakeOllama([baseline_json()])))
    runners = Runners(ScriptedDiagnostic(diag or [diagnostic(SID, {"ACL": 0.85})]),
                      ScriptedRecommendation(rec or [recommendation()]), "agent", "agent", ["fake agents"])
    return Services(settings, llm, runners, ModelServiceClient(settings, transport=model_service(missing)))


def app_client(services: Services) -> TestClient:
    return TestClient(create_app(services=services, configure_logging=False))