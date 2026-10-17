"""All runtime settings, read from environment variables or .env. Use get_settings() everywhere."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Local LLMs (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    # Coordinator uses Qwen, not MedGemma, so it is independent of the agent it adjudicates.
    diagnostic_model: str = "qwen3:8b"
    coordinator_model: str = "qwen3:8b"
    recommendation_model: str = "medgemma:4b"
    ollama_keep_alive: str = "30m"
    llm_timeout_s: float = Field(default=180.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_json_retries: int = Field(default=1, ge=0, le=3)

    # --- Model service ---
    model_service_url: str = "http://localhost:8001"
    model_mode: Literal["stub", "replay", "live"] = "stub"
    model_checkpoint: str = "model_service/checkpoints/knee_v1.pt"
    replay_predictions: str = "model_service/checkpoints/predictions.json"

    # --- Coordinator ---
    tool_timeout_s: float = Field(default=60.0, gt=0)
    max_recheck_rounds: int = Field(default=2, ge=0, le=5)
    max_coordinator_steps: int = Field(default=12, ge=1)
    report_threshold: float = Field(default=0.5, ge=0, le=1)
    low_confidence_threshold: float = Field(default=0.6, ge=0, le=1)

    # --- Recommendation Agent ---
    tavily_api_key: SecretStr | None = None

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"

    def model_for(self, agent: str) -> str:
        models = {"diagnostic": self.diagnostic_model, "coordinator": self.coordinator_model,
                  "recommendation": self.recommendation_model, "baseline": self.recommendation_model}
        if agent not in models:
            raise ValueError(f"no model configured for agent {agent!r}")
        return models[agent]


@lru_cache
def get_settings() -> Settings:
    return Settings()