"""JSON logging to logs/app.jsonl (rotated, secrets redacted), plus one trace file per session in logs/traces/."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path

from shared.config import get_settings
from shared.schema import AgentEvent


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": round(record.created, 3), "level": record.levelname, "logger": record.name,
                 "message": record.getMessage()}
        for key in ("session_id", "agent", "data"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_text:  # already formatted (and redacted) by RedactSecrets
            entry["exception"] = record.exc_text
        elif record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


REDACTED = "***redacted***"


class RedactSecrets(logging.Filter):
    """Replaces secret values (e.g. the Tavily key) wherever they appear in a log record."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self.secrets = [s for s in secrets if len(s) >= 8]  # short values would mangle normal text

    def _clean(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, REDACTED)
            return value
        if isinstance(value, dict):
            return {k: self._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._clean(v) for v in value)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secrets:
            record.msg = self._clean(record.getMessage())
            record.args = None
            if getattr(record, "data", None) is not None:
                record.data = self._clean(record.data)
            if record.exc_info and not record.exc_text:
                record.exc_text = self._clean(logging.Formatter().formatException(record.exc_info))
        return True


def setup_logging(log_dir: str | None = None, level: str | None = None) -> None:
    settings = get_settings()
    directory = Path(log_dir or settings.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level or settings.log_level)
    formatter = JsonFormatter()
    redact = RedactSecrets([settings.tavily_api_key.get_secret_value()] if settings.has_tavily_key else [])
    file_handler = logging.handlers.RotatingFileHandler(
        directory / "app.jsonl", maxBytes=settings.log_max_bytes, backupCount=settings.log_backup_count,
        encoding="utf-8")
    for handler in (logging.StreamHandler(sys.stdout), file_handler):
        handler.setFormatter(formatter)
        handler.addFilter(redact)
        root.addHandler(handler)
    for noisy in ("httpx", "httpcore"):  # one line per HTTP call is too much at INFO
        logging.getLogger(noisy).setLevel(logging.WARNING)


class TraceWriter:
    """Appends AgentEvents for one session to logs/traces/<session_id>.jsonl."""

    def __init__(self, session_id: str, log_dir: str | None = None) -> None:
        self.path = Path(log_dir or get_settings().log_dir) / "traces" / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: AgentEvent) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(event.model_dump_json() + "\n")

    def read(self) -> list[AgentEvent]:
        if not self.path.exists():
            return []
        return [AgentEvent.model_validate_json(line) for line in self.path.read_text().splitlines() if line.strip()]


def now() -> float:
    return time.time()