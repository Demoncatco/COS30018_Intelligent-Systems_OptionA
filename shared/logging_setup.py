"""JSON logging to logs/app.jsonl, plus one trace file per session in logs/traces/."""

from __future__ import annotations

import json
import logging
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
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(log_dir: str | None = None, level: str | None = None) -> None:
    settings = get_settings()
    directory = Path(log_dir or settings.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level or settings.log_level)
    formatter = JsonFormatter()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "app.jsonl")):
        handler.setFormatter(formatter)
        root.addHandler(handler)


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