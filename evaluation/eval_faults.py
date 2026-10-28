# Controlled failures used to exercise retries, recovery and evidence-gap handling
# Some faults use real HTTP failure paths; wrappers simulate transient failures without waiting
# Injected error messages identify the fault so results can distinguish it from an unexpected failure

from __future__ import annotations

from typing import Any

from agents.coordinator.runners import ToolTimeoutError

# Port 9 is used as the unavailable local endpoint for connection-failure tasks
UNREACHABLE_URL = "http://127.0.0.1:9"
WRONG_STUDY_ID = "eval-wrong-study"


class InjectedFault(ConnectionError):
    """Identify a simulated connection failure raised by the evaluation wrappers"""


class FailFirst:
    """Fail a runner for the first count calls, then pass later calls to the real runner"""

    def __init__(self, runner: Any, count: int, kind: str, agent_name: str) -> None:
        """Keep the original runner and fault configuration with an independent call counter"""
        self.runner = runner
        self.count = count
        self.kind = kind  # "connection", "timeout" or "wrong_study"
        self.agent_name = agent_name
        self.calls = 0

    def __call__(self, request, *, session_id: str):
        """Count attempts and inject a timeout, changed study identifier or connection error
        The wrong-study fault still calls the real runner before relabelling its response
        """
        self.calls += 1
        if self.calls > self.count:
            return self.runner(request, session_id=session_id)
        if self.kind == "timeout":
            raise ToolTimeoutError(f"simulated timeout (injected fault, call {self.calls})")
        if self.kind == "wrong_study":
            # The real agent runs, but its answer is relabelled as another study, like a mixed-up result
            response = self.runner(request, session_id=session_id)
            return response.model_copy(update={"study_id": WRONG_STUDY_ID})
        raise InjectedFault(f"{self.agent_name} connection dropped (injected fault, call {self.calls})")


def failing_retrieve(query: str, k: int = 3, finding: str | None = None) -> list[dict]:
    """Simulate an unavailable local document store by raising an identifiable injected error"""
    raise InjectedFault("document store unavailable (injected fault)")


def failing_search(query: str, max_results: int) -> list[dict]:
    """Simulate an unavailable web search service by raising an identifiable injected error"""
    raise InjectedFault("web search unavailable (injected fault)")


def no_web_search(query: str, max_results: int) -> list[dict]:
    """Disable web results with an empty successful response rather than a service failure"""
    return []
