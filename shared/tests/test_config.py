# Tests for shared/config.py: pinned model tags, secrets, and value checks.

import pytest
from pydantic import ValidationError

from shared.config import Settings


def _settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_defaults_match_the_agreed_models():
    s = _settings()
    assert s.model_for("diagnostic").startswith("qwen") and s.model_for("coordinator").startswith("qwen")
    assert s.model_for("recommendation").startswith("medgemma") and s.model_for("baseline") == s.recommendation_model


@pytest.mark.parametrize("tag", ["qwen3", "qwen3:latest", "qwen3:", "registry:5000/qwen3"])
def test_model_tags_must_be_pinned(tag):
    with pytest.raises(ValidationError):
        _settings(coordinator_model=tag)


@pytest.mark.parametrize("tag", ["qwen3:8b", "medgemma:4b", "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"])
def test_pinned_tags_are_accepted(tag):
    assert _settings(recommendation_model=tag).recommendation_model == tag


def test_models_can_come_from_environment_variables(monkeypatch):
    monkeypatch.setenv("COORDINATOR_MODEL", "qwen3:14b")
    monkeypatch.setenv("COORDINATOR_THINK", "true")
    s = _settings()
    assert s.coordinator_model == "qwen3:14b" and s.think_for("coordinator") is True


def test_think_is_never_sent_to_medgemma():
    s = _settings(diagnostic_think=True, coordinator_think=True)
    assert s.think_for("recommendation") is None and s.think_for("baseline") is None
    with pytest.raises(ValueError):
        s.think_for("janitor")


@pytest.mark.parametrize("value", [None, "", "your-tavily-api-key-here", "  "])
def test_placeholder_secret_counts_as_unset(value):
    assert _settings(tavily_api_key=value).has_tavily_key is False


def test_real_secret_is_kept_but_never_printed():
    s = _settings(tavily_api_key="tvly-real-key")
    assert s.tavily_api_key.get_secret_value() == "tvly-real-key"
    assert "tvly-real-key" not in repr(s) and "tvly-real-key" not in str(s.safe_summary())
    assert s.safe_summary()["tavily_api_key"] == "set"


@pytest.mark.parametrize("value", ["30m", "1h", "90s", "0", "-1", "300"])
def test_keep_alive_accepts_ollama_durations(value):
    assert _settings(ollama_keep_alive=value).ollama_keep_alive == value


def test_bad_values_are_rejected():
    for bad in ({"ollama_keep_alive": "forever"}, {"report_threshold": 1.5}, {"llm_num_ctx": 512},
                {"log_level": "LOUD"}, {"max_coordinator_steps": 0}):
        with pytest.raises(ValidationError):
            _settings(**bad)


def test_urls_lose_trailing_slashes_and_log_level_is_case_insensitive():
    s = _settings(ollama_host="http://localhost:11434/", log_level="debug")
    assert s.ollama_host == "http://localhost:11434" and s.log_level == "DEBUG"


def test_empty_seed_means_random_sampling(monkeypatch):
    monkeypatch.setenv("LLM_SEED", "")
    assert _settings().llm_seed is None