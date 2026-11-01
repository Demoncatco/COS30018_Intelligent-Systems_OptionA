"""
How the Diagnostic Agent asks the Model Service for a prediction. Same call the interim adapter in
agents/coordinator/runners.py makes (POST {settings.model_service_url}/predict), with two additions:
a per-call timeout, and the Coordinator's own StudyNotAvailableError for HTTP 404.

Any callable `(study_id, timeout=None) -> PredictionResponse` can be passed to DiagnosticAgent instead.
"""

from __future__ import annotations

import httpx

from agents.coordinator.runners import StudyNotAvailableError
from shared.config import Settings
from shared.schema import PredictionRequest, PredictionResponse


class ModelServiceClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=settings.model_service_url, timeout=settings.tool_timeout_s,
                                  transport=transport)

    def __call__(self, study_id: str, timeout: float | None = None) -> PredictionResponse:
        extra = {} if timeout is None else {"timeout": timeout}
        response = self._http.post("/predict", json=PredictionRequest(study_id=study_id).model_dump(), **extra)
        if response.status_code == 404:           # the Model Service has no usable data for this study
            raise StudyNotAvailableError(f"study {study_id} is not available in the Model Service")
        response.raise_for_status()               # 5xx / network problems: the caller may retry
        return PredictionResponse.model_validate(response.json())
