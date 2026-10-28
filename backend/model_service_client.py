"""Small HTTP client for the Model Service, used by the baseline and the backend's health check."""

from __future__ import annotations

import httpx

from agents.coordinator.runners import StudyNotAvailableError
from shared.config import Settings, get_settings
from shared.schema import PredictionRequest, PredictionResponse

HEALTH_TIMEOUT_S = 5.0  # health checks should answer fast, even when the service is down


class ModelServiceClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings or get_settings()
        self._http = httpx.Client(base_url=self.settings.model_service_url, timeout=self.settings.tool_timeout_s,
                                  transport=transport)

    def predict(self, study_id: str) -> PredictionResponse:
        response = self._http.post("/predict", json=PredictionRequest(study_id=study_id).model_dump())
        if response.status_code == 404:
            raise StudyNotAvailableError(f"study {study_id} is not available in the Model Service")
        response.raise_for_status()
        return PredictionResponse.model_validate(response.json())

    def health(self) -> dict:
        try:
            response = self._http.get("/health", timeout=HEALTH_TIMEOUT_S)
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            return {"status": "unreachable", "url": self.settings.model_service_url, "error": str(error)}
        return {"url": self.settings.model_service_url, **body}

    def studies(self) -> dict:
        response = self._http.get("/studies", timeout=HEALTH_TIMEOUT_S)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._http.close()