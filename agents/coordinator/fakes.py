"""Scripted Diagnostic and Recommendation stand-ins for tests and validation.

Each call returns the next item (response, exception, Slow or callable); the last item repeats.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from shared.schema import (
    FINDINGS,
    CoordinatorRequest,
    DiagnosticAgentRequest,
    DiagnosticAgentResponse,
    DiagnosticDecision,
    EvidenceStatus,
    RecommendationResponse,
    UrgencyLevel,
)


@dataclass(frozen=True)
class Slow:
    seconds: float
    then: Any


class _Scripted:
    def __init__(self, script: Sequence[Any]) -> None:
        if not script:
            raise ValueError("script needs at least one item")
        self.script = list(script)
        self.requests: list[Any] = []

    def _next(self, request: Any) -> Any:
        index = min(len(self.requests), len(self.script) - 1)
        self.requests.append(request)
        item = self.script[index]
        if isinstance(item, Slow):
            time.sleep(item.seconds)
            item = item.then
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, (DiagnosticAgentResponse, RecommendationResponse)):
            return item(request)
        return item

    @property
    def calls(self) -> int:
        return len(self.requests)


class ScriptedDiagnostic(_Scripted):
    def __call__(self, request: DiagnosticAgentRequest, *, session_id: str) -> DiagnosticAgentResponse:
        return self._next(request)


class ScriptedRecommendation(_Scripted):
    def __call__(self, request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse:
        return self._next(request)


# --- Builders -----------------------------------------------------------------------------------------------

def diagnostic(study_id: str, positives: dict[str, float] | None = None, *,
               confidence: dict[str, float] | None = None, overall: float = 0.85,
               decision: DiagnosticDecision | str = DiagnosticDecision.FINALIZE, flags: Sequence[str] = (),
               variance: dict[str, float] | None = None, impression: str | None = None,
               laterality: str = "left", base: float = 0.05, base_confidence: float = 0.9,
               retry_count: int = 0) -> DiagnosticAgentResponse:
    """All findings at `base` except `positives`."""
    findings = {name: base for name in FINDINGS} | (positives or {})
    conf = {name: base_confidence for name in FINDINGS} | (confidence or {})
    if impression is None:
        named = [n for n in FINDINGS if findings[n] >= 0.5]
        impression = ("Possible " + ", ".join(named) + ".") if named else "No finding above threshold."
    return DiagnosticAgentResponse(
        study_id=study_id, findings=findings, per_finding_confidence=conf, overall_confidence=overall,
        decision=DiagnosticDecision(decision), retry_count=retry_count, cross_plane_variance=variance or {},
        quality_flags=list(flags), laterality=laterality, impression=impression,
        trace=[f"scripted diagnostic: decision={DiagnosticDecision(decision).value}"],
    )


def recommendation(text: str = "Findings may indicate a ligament injury. Consult a qualified clinician.", *,
                   urgency: UrgencyLevel | str = UrgencyLevel.SEE_DOCTOR_SOON,
                   evidence: EvidenceStatus | str = EvidenceStatus.GROUNDED_LOCAL,
                   citations: Sequence[str] = ("https://www.nhs.uk/conditions/knee-ligament-injuries/",),
                   concerns: Sequence[str] = (), recheck: Sequence[str] = ()) -> RecommendationResponse:
    return RecommendationResponse(
        recommendation_text=text, citations=list(citations), urgency=UrgencyLevel(urgency),
        evidence_status=EvidenceStatus(evidence), concerns=list(concerns), suggested_recheck=list(recheck),
        trace=["scripted recommendation"],
    )


def echo_study(build: Callable[[str], DiagnosticAgentResponse]) -> Callable[[DiagnosticAgentRequest],
                                                                              DiagnosticAgentResponse]:
    """Builds the response for the requested study."""
    return lambda request: build(request.study_id)