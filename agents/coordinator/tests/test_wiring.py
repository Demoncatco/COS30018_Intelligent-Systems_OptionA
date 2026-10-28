# Tests which Diagnostic/Recommendation implementations the Coordinator and backend get.

import sys
import types

import pytest

from agents.coordinator import wiring
from agents.coordinator.runners import ModelServiceDiagnostic
from shared.config import Settings
from shared.llm import LLMClient


def _settings(tmp_path, impl):
    return Settings(_env_file=None, log_dir=str(tmp_path), diagnostic_impl=impl)


@pytest.fixture
def fake_diagnostic_module(monkeypatch):
    module = types.ModuleType(wiring.DIAGNOSTIC_MODULE)

    class DiagnosticAgent:
        def __init__(self, llm, settings):
            self.llm, self.settings = llm, settings

        def __call__(self, request, *, session_id):
            raise NotImplementedError

    module.DiagnosticAgent = DiagnosticAgent
    monkeypatch.setitem(sys.modules, wiring.DIAGNOSTIC_MODULE, module)
    return DiagnosticAgent


@pytest.fixture
def no_diagnostic_class(monkeypatch):
    monkeypatch.setitem(sys.modules, wiring.DIAGNOSTIC_MODULE, types.ModuleType(wiring.DIAGNOSTIC_MODULE))


def test_auto_uses_the_real_agent_when_it_exists(tmp_path, fake_diagnostic_module):
    settings = _settings(tmp_path, "auto")
    runner, impl, _ = wiring.load_diagnostic(settings, LLMClient(settings))
    assert impl == "agent" and isinstance(runner, fake_diagnostic_module)


def test_auto_falls_back_to_the_interim_adapter(tmp_path, no_diagnostic_class):
    settings = _settings(tmp_path, "auto")
    runner, impl, note = wiring.load_diagnostic(settings, LLMClient(settings))
    assert impl == "interim" and isinstance(runner, ModelServiceDiagnostic) and "DiagnosticAgent" in note


def test_agent_mode_refuses_to_fall_back(tmp_path, no_diagnostic_class):
    settings = _settings(tmp_path, "agent")
    with pytest.raises(wiring.AgentUnavailableError):
        wiring.load_diagnostic(settings, LLMClient(settings))


def test_interim_mode_ignores_the_real_agent(tmp_path, fake_diagnostic_module):
    settings = _settings(tmp_path, "interim")
    _, impl, _ = wiring.load_diagnostic(settings, LLMClient(settings))
    assert impl == "interim"


def test_unavailable_recommendation_raises_so_the_session_ends_partial():
    runner = wiring.UnavailableRecommendation("langgraph not installed")
    with pytest.raises(wiring.AgentUnavailableError, match="langgraph"):
        runner(None, session_id="x")


def test_build_runners_wires_the_real_recommendation_agent(tmp_path, no_diagnostic_class):
    settings = _settings(tmp_path, "auto")
    runners = wiring.build_runners(settings, LLMClient(settings))
    assert runners.recommendation_impl == "agent" and runners.diagnostic_impl == "interim"