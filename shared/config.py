"""All runtime settings, read from environment variables or .env. Use get_settings() everywhere.

Environment variables win over .env, so a single run can be changed without editing the file:
    MODEL_MODE=replay uvicorn model_service.api.main:app --port 8001
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AgentName = Literal["diagnostic", "coordinator", "recommendation", "baseline"]

# Values copied straight from .env.example that mean "no key set".
_PLACEHOLDER_SECRETS = {"", "your-tavily-api-key-here", "changeme", "none", "null"}
# Ollama keep_alive: a duration ("30m", "1h", "90s"), a number of seconds, "0" (unload) or "-1" (never unload).
_KEEP_ALIVE = re.compile(r"^-?\d+(\.\d+)?(ms|s|m|h)?$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Local LLMs (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    # Pinned tags (name:tag, never untagged or :latest) so every run and the report use the same weights.
    # MedGemma reads the imaging findings (Diagnostic); Qwen handles tool calling and synthesis.
    diagnostic_model: str = "medgemma:4b"
    coordinator_model: str = "qwen3:8b"
    recommendation_model: str = "qwen3:8b"  # same tag as the Coordinator, so Ollama keeps one Qwen in memory
    baseline_model: str = "medgemma:4b"  # single-call baseline, independent of the agents' models
    # Qwen3 "thinking": slower and longer, sometimes better decisions. Never sent to non-Qwen models.
    coordinator_think: bool = False
    recommendation_think: bool = False
    ollama_keep_alive: str = "30m"
    llm_timeout_s: float = Field(default=180.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_seed: int | None = 30018  # fixed seed so evaluation runs are repeatable; unset for random sampling
    llm_num_ctx: int = Field(default=8192, ge=2048, le=131072)  # Ollama's default window silently truncates prompts
    llm_json_retries: int = Field(default=1, ge=0, le=3)
    llm_http_retries: int = Field(default=1, ge=0, le=3)  # retries when Ollama is unreachable or briefly unavailable

    # --- Model service ---
    model_service_url: str = "http://localhost:8001"
    model_mode: Literal["stub", "replay", "live"] = "stub"
    model_checkpoint: str = "model_service/checkpoints/knee_v1.pt"
    replay_predictions: str = "model_service/checkpoints/predictions.json"
    # live mode only: where the Kaggle DICOMs and train_series.csv were downloaded to
    live_dataset_root: str = "data/rsna-knee"
    live_series_csv: str = "data/rsna-knee/train_series.csv"
    live_split_folder: str = "train_series"
    model_device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Coordinator ---
    tool_timeout_s: float = Field(default=60.0, gt=0)
    max_recheck_rounds: int = Field(default=2, ge=0, le=5)
    max_coordinator_steps: int = Field(default=12, ge=1)
    report_threshold: float = Field(default=0.5, ge=0, le=1)  # probability at or above which a finding is reported
    low_confidence_threshold: float = Field(default=0.6, ge=0, le=1)  # confidence below which a finding is flagged

    # --- Recommendation Agent ---
    tavily_api_key: SecretStr | None = None

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: str = "logs"

    # --- Validation ---

    @field_validator("diagnostic_model", "coordinator_model", "recommendation_model", "baseline_model")
    @classmethod
    def _model_tag_is_pinned(cls, value: str) -> str:
        value = value.strip()
        name, _, tag = value.rpartition(":")
        if not name or not tag or "/" in tag:
            raise ValueError(f"model {value!r} has no tag; pin one, e.g. 'qwen3:8b' (see `ollama list`)")
        if tag == "latest":
            raise ValueError(f"model {value!r} uses ':latest', which can change under you; pin an explicit tag")
        return value

    @field_validator("ollama_host", "model_service_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("ollama_keep_alive")
    @classmethod
    def _keep_alive_format(cls, value: str) -> str:
        value = value.strip()
        if not _KEEP_ALIVE.match(value):
            raise ValueError(f"OLLAMA_KEEP_ALIVE={value!r} is not a duration like '30m', '1h', '0' or '-1'")
        return value

    @field_validator("llm_seed", mode="before")
    @classmethod
    def _empty_seed_means_random(cls, value: object) -> object:
        return None if isinstance(value, str) and value.strip().lower() in {"", "none", "null"} else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("tavily_api_key", mode="before")
    @classmethod
    def _placeholder_means_unset(cls, value: object) -> object:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in _PLACEHOLDER_SECRETS):
            return None
        return value

    # --- Helpers ---

    def model_for(self, agent: str) -> str:
        models = {"diagnostic": self.diagnostic_model, "coordinator": self.coordinator_model,
                  "recommendation": self.recommendation_model, "baseline": self.baseline_model}
        if agent not in models:
            raise ValueError(f"no model configured for agent {agent!r}")
        return models[agent]

    def think_for(self, agent: str) -> bool | None:
        # None means "do not send the think option": only Qwen has a thinking mode; Ollama rejects it elsewhere.
        model = self.model_for(agent)  # same unknown-agent error as model_for
        if not uses_qwen(model):
            return None
        return {"coordinator": self.coordinator_think, "recommendation": self.recommendation_think}.get(agent, False)

    def uses_tool_calls(self, agent: str) -> bool:
        # Qwen gets native tool calls; other models (MedGemma) use JSON-schema output via LLMClient.decide.
        return uses_qwen(self.model_for(agent))

    @property
    def has_tavily_key(self) -> bool:
        return self.tavily_api_key is not None

    def safe_summary(self) -> dict:
        # Every setting with secrets redacted. Safe to log at startup and to paste into the report.
        summary = self.model_dump(mode="json", exclude={"tavily_api_key"})
        summary["tavily_api_key"] = "set" if self.has_tavily_key else "not set"
        return summary


def uses_qwen(model: str) -> bool:
    # Matches "qwen3:8b" and Hugging Face paths like "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M".
    return "qwen" in model.lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()