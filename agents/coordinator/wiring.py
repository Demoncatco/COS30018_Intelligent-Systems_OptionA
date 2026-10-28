"""Builds the Diagnostic and Recommendation runners the Coordinator calls. Used by the CLI and the backend.

DIAGNOSTIC_IMPL: "agent" = agents.diagnostic_agent.agent.DiagnosticAgent, "interim" = Model Service adapter,
"auto" = the real agent if that class exists, otherwise the interim adapter.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

from agents.coordinator.runners import (
    DiagnosticRunner,
    ModelServiceDiagnostic,
    RecommendationRunner,
)
from shared.config import Settings
from shared.llm import LLMClient
from shared.schema import CoordinatorRequest, RecommendationResponse

logger = logging.getLogger(__name__)

DIAGNOSTIC_MODULE = "agents.diagnostic_agent.agent"
DIAGNOSTIC_CLASS = "DiagnosticAgent"


class AgentUnavailableError(RuntimeError):
    """A required agent could not be imported or built."""


@dataclass
class Runners:
    diagnostic: DiagnosticRunner
    recommendation: RecommendationRunner
    diagnostic_impl: str  # "agent" or "interim"
    recommendation_impl: str  # "agent" or "unavailable"
    notes: list[str]


class UnavailableRecommendation:
    """Stands in when the Recommendation Agent cannot be imported, so sessions end as partial, not crash."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __call__(self, request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse:
        raise AgentUnavailableError(f"Recommendation Agent unavailable: {self.reason}")


def load_diagnostic(settings: Settings, llm: LLMClient) -> tuple[DiagnosticRunner, str, str]:
    # Returns (runner, impl, note).
    if settings.diagnostic_impl == "interim":
        return ModelServiceDiagnostic(settings), "interim", "DIAGNOSTIC_IMPL=interim"
    try:
        module = importlib.import_module(DIAGNOSTIC_MODULE)
        agent_cls = getattr(module, DIAGNOSTIC_CLASS)
    except (ImportError, AttributeError) as error:
        reason = f"{DIAGNOSTIC_MODULE}.{DIAGNOSTIC_CLASS} not found ({type(error).__name__}: {error})"
        if settings.diagnostic_impl == "agent":
            raise AgentUnavailableError(reason) from error
        logger.warning("using the interim Diagnostic adapter: %s", reason)
        return ModelServiceDiagnostic(settings), "interim", reason
    return agent_cls(llm=llm, settings=settings), "agent", f"{DIAGNOSTIC_MODULE}.{DIAGNOSTIC_CLASS}"


def load_recommendation(settings: Settings, llm: LLMClient) -> tuple[RecommendationRunner, str, str]:
    try:
        from agents.recommendation_agent.runner import RecommendationAgent
    except Exception as error:  # noqa: BLE001 - missing optional packages should not stop the backend
        reason = f"{type(error).__name__}: {error}"
        logger.error("Recommendation Agent could not be imported: %s", reason)
        return UnavailableRecommendation(reason), "unavailable", reason
    return RecommendationAgent(llm=llm, settings=settings), "agent", "agents.recommendation_agent.runner"


def build_runners(settings: Settings, llm: LLMClient) -> Runners:
    diagnostic, diag_impl, diag_note = load_diagnostic(settings, llm)
    recommendation, rec_impl, rec_note = load_recommendation(settings, llm)
    return Runners(diagnostic, recommendation, diag_impl, rec_impl,
                   [f"diagnostic: {diag_note}", f"recommendation: {rec_note}"])