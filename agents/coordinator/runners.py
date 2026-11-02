"""How the Coordinator calls the other agents, with timeouts.

Agents must match DiagnosticRunner / RecommendationRunner and raise on failure. The Model Service adapter is interim.
"""

from __future__ import annotations

import atexit
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Protocol, TypeVar

import httpx

from shared.config import Settings, get_settings
from shared.schema import (
    FINDINGS,
    CoordinatorRequest,
    DiagnosticAgentRequest,
    DiagnosticAgentResponse,
    DiagnosticDecision,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)

R = TypeVar("R")


class DiagnosticRunner(Protocol):
    def __call__(self, request: DiagnosticAgentRequest, *, session_id: str) -> DiagnosticAgentResponse: ...


class RecommendationRunner(Protocol):
    def __call__(self, request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse: ...


class ToolTimeoutError(TimeoutError):
    """An agent did not answer within TOOL_TIMEOUT_S."""


# Worker threads allow a timeout; a timed-out call finishes in the background and is ignored.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="coordinator-tool")
atexit.register(_EXECUTOR.shutdown, wait=False, cancel_futures=True)


def call_with_timeout(fn: Callable[..., R], timeout_s: float, *args, **kwargs) -> R:
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        future.cancel()
        raise ToolTimeoutError(f"no answer within {timeout_s:g}s (TOOL_TIMEOUT_S)") from None


# --- Interim stand-ins -------------------------------------------------------------------------------------

class StudyNotAvailableError(LookupError):
    """The Model Service has no data for this study (HTTP 404)."""


class ModelServiceDiagnostic:
    """INTERIM: Model Service /predict -> DiagnosticAgentResponse. No LLM."""

    def __init__(self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings or get_settings()
        self._http = httpx.Client(base_url=self.settings.model_service_url, timeout=self.settings.tool_timeout_s,
                                  transport=transport)

    def __call__(self, request: DiagnosticAgentRequest, *, session_id: str) -> DiagnosticAgentResponse:
        response = self._http.post("/predict", json=PredictionRequest(study_id=request.study_id).model_dump())
        if response.status_code == 404:
            raise StudyNotAvailableError(f"study {request.study_id} is not available in the Model Service")
        response.raise_for_status()
        return prediction_to_diagnostic(PredictionResponse.model_validate(response.json()), request,
                                        self.settings.report_threshold)


def prediction_to_diagnostic(prediction: PredictionResponse, request: DiagnosticAgentRequest,
                             report_threshold: float = 0.5) -> DiagnosticAgentResponse:
    planes = {p.plane: p.scores for p in prediction.per_plane}
    confidence: dict[str, float] = {}
    variance: dict[str, float] = {}
    for name in FINDINGS:
        values = [scores[name] for scores in planes.values()]
        spread = (max(values) - min(values)) if len(values) > 1 else 0.0
        if spread >= 0.3:
            variance[name] = round(spread, 3)
        decisiveness = abs(prediction.findings[name] - 0.5) * 2  # 0..1
        confidence[name] = round(min(1.0, max(0.0, 0.5 * decisiveness + 0.5 * (1 - spread))), 3)

    reported = [n for n in FINDINGS if prediction.findings[n] >= report_threshold]
    overall = statistics.fmean(confidence[n] for n in reported) if reported else statistics.fmean(confidence.values())
    missing_plane = any(flag.startswith("missing_plane:") for flag in prediction.quality_flags)
    decision = DiagnosticDecision.ESCALATE if (missing_plane or not planes) else DiagnosticDecision.FINALIZE
    impression = ("Possible " + ", ".join(reported) + "." if reported else "No finding above the reporting threshold.")
    return DiagnosticAgentResponse(
        study_id=prediction.study_id, findings=prediction.findings, per_finding_confidence=confidence,
        overall_confidence=round(overall, 3), decision=decision, retry_count=0, cross_plane_variance=variance,
        per_plane={plane: dict(scores) for plane, scores in planes.items()},
        quality_flags=prediction.quality_flags, laterality=prediction.laterality, impression=impression,
        trace=[f"INTERIM adapter (no LLM): called Model Service /predict ({prediction.model_version}).",
               f"task_type={request.task_type}; planes={sorted(planes)}; decision={decision.value}."],
    )


class NotConnectedRecommendation:
    """INTERIM: always fails until the Recommendation Agent is merged."""

    def __call__(self, request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse:
        raise RuntimeError("Recommendation Agent is not connected yet (interim placeholder)")