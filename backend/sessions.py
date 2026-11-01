"""Runs multi-agent and baseline sessions in worker threads and keeps their state for the API.

Finished sessions are also saved to logs/reports/<session_id>.json, so they can be read back after a restart.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from agents.coordinator.agent import CoordinatorAgent
from agents.coordinator.wiring import Runners, build_runners
from backend.baseline import BaselineAgent, StudyScorer
from backend.model_service_client import ModelServiceClient
from backend.models import Driver, SessionKind
from shared.config import Settings, get_settings
from shared.llm import LLMClient, LLMResult
from shared.logging_setup import TraceWriter
from shared.schema import (
    AgentName,
    FinalReport,
    PatientMetadata,
    SessionState,
    SessionStatus,
)

logger = logging.getLogger(__name__)

class QueueFullError(RuntimeError):
    """Too many sessions are waiting; the API answers 429."""


SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # also stops path tricks when reading from disk


@dataclass
class Services:
    """Everything a session needs. Built once per process; tests pass fakes."""

    settings: Settings
    llm: LLMClient
    runners: Runners
    model_service: ModelServiceClient
    scorer: StudyScorer | None = None  # baseline's scorer; defaults to model_service

    @classmethod
    def build(cls, settings: Settings | None = None) -> Services:
        settings = settings or get_settings()
        llm = LLMClient(settings)
        runners = build_runners(settings, llm)
        # Baseline scores with the Diagnostic Agent's local model when it has one (fair comparison)
        scorer = getattr(runners.diagnostic, "scorer", None)
        return cls(settings, llm, runners, ModelServiceClient(settings), scorer)

    def coordinator(self, driver: Driver) -> CoordinatorAgent:
        return CoordinatorAgent(self.runners.diagnostic, self.runners.recommendation,
                                llm=None if driver == "fixed" else self.llm, settings=self.settings, driver=driver)

    def baseline(self) -> BaselineAgent:
        return BaselineAgent(self.scorer or self.model_service, self.llm, self.settings)

    def close(self) -> None:
        self.llm.close()
        self.model_service.close()


@dataclass
class SessionEntry:
    state: SessionState
    kind: SessionKind
    driver: str
    done: threading.Event = field(default_factory=threading.Event)
    usage: dict[str, dict] = field(default_factory=dict)
    stats: dict | None = None

    @property
    def session_id(self) -> str:
        return self.state.session_id


class SessionManager:
    def __init__(self, services: Services) -> None:
        self.services = services
        self.settings = services.settings
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.settings.backend_max_workers,
                                            thread_name_prefix="session")
        self._reports_dir = Path(self.settings.log_dir) / "reports"
        services.llm.on_call = self.record_llm_call  # per-session token counts for every agent

    # --- Starting ---

    def start(self, study_id: str, patient: PatientMetadata, kind: SessionKind = "multi_agent",
              driver: Driver = "tools") -> SessionEntry:
        pending = self.counts()["pending"]
        if pending >= self.settings.backend_max_pending:
            raise QueueFullError(f"{pending} sessions are already waiting (BACKEND_MAX_PENDING); try again shortly")
        state = SessionState(session_id=uuid.uuid4().hex[:12], study_id=study_id.strip(), patient_metadata=patient,
                             produced_by=kind)
        agent = AgentName.BASELINE if kind == "baseline" else AgentName.COORDINATOR
        event = state.add_event(agent, "status", "Session queued", {"status": "pending", "driver": driver})
        self._write_trace(state.session_id, event)
        entry = SessionEntry(state=state, kind=kind, driver="baseline" if kind == "baseline" else driver)
        with self._lock:
            self._sessions[entry.session_id] = entry
            self._evict()
        self._executor.submit(self._run, entry)
        logger.info("session started", extra={"session_id": entry.session_id,
                                              "data": {"study_id": study_id, "kind": kind, "driver": entry.driver}})
        return entry

    def _run(self, entry: SessionEntry) -> None:
        state = entry.state
        try:
            if entry.kind == "baseline":
                self.services.baseline().run_session(state)
            else:
                result = self.services.coordinator(entry.driver).run_session(state)
                entry.stats = result.stats.to_dict() | {"override_rate": result.override_rate,
                                                        "executed_tools": result.executed_tools}
        except Exception as error:
            logger.exception("session crashed", extra={"session_id": state.session_id})
            self._fail(entry, f"internal error: {type(error).__name__}: {error}")
        finally:
            self._total_tokens(entry)
            self._save(entry)
            entry.done.set()

    def _fail(self, entry: SessionEntry, message: str) -> None:
        state = entry.state
        agent = AgentName.BASELINE if entry.kind == "baseline" else AgentName.COORDINATOR
        self._write_trace(state.session_id, state.add_error(agent, message))
        if state.is_finished:
            return
        state.final_report = FinalReport(session_id=state.session_id, study_id=state.study_id,
                                         status=SessionStatus.FAILED, produced_by=entry.kind,
                                         errors=list(state.errors))
        self._write_trace(state.session_id, state.set_status(SessionStatus.FAILED, "Session failed"))

    # --- Usage ---

    def record_llm_call(self, result: LLMResult) -> None:
        # LLMClient.on_call hook. Runs in worker threads.
        with self._lock:
            entry = self._sessions.get(result.session_id or "")
            if entry is None:
                return
            totals = entry.usage.setdefault(result.agent, {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                                                           "latency_ms": 0.0})
            totals["calls"] += 1
            totals["tokens_in"] += result.tokens_in
            totals["tokens_out"] += result.tokens_out
            totals["latency_ms"] = round(totals["latency_ms"] + result.latency_ms, 1)

    def _total_tokens(self, entry: SessionEntry) -> None:
        # The Coordinator's report only counts its own tokens; count every agent's for fair baseline comparison.
        report = entry.state.final_report
        if report is None or not entry.usage:
            return
        with self._lock:
            total = sum(u["tokens_in"] + u["tokens_out"] for u in entry.usage.values())
        if total > report.total_tokens:
            entry.state.final_report = report.model_copy(update={"total_tokens": total})

    # --- Reading ---

    def get(self, session_id: str) -> SessionEntry | None:
        if not SESSION_ID.match(session_id):
            return None
        with self._lock:
            entry = self._sessions.get(session_id)
        return entry or self._load(session_id)

    def list(self, limit: int = 50) -> list[SessionEntry]:
        with self._lock:
            entries = list(self._sessions.values())
        return sorted(entries, key=lambda e: -e.state.created_at)[:limit]

    def counts(self) -> dict[str, int]:
        with self._lock:
            statuses = [e.state.status for e in self._sessions.values()]
        return {"in_memory": len(statuses), "pending": statuses.count(SessionStatus.PENDING),
                "running": statuses.count(SessionStatus.RUNNING)}

    def wait(self, entry: SessionEntry, timeout_s: float) -> bool:
        return entry.done.wait(timeout_s) if timeout_s > 0 else entry.done.is_set()

    # --- Disk ---

    def _save(self, entry: SessionEntry) -> None:
        try:
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            body = {"kind": entry.kind, "driver": entry.driver, "usage": entry.usage, "stats": entry.stats,
                    "state": entry.state.model_dump(mode="json", exclude={"events"})}
            (self._reports_dir / f"{entry.session_id}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        except OSError as error:
            logger.warning("could not save session: %s", error, extra={"session_id": entry.session_id})

    def _load(self, session_id: str) -> SessionEntry | None:
        path = self._reports_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            events = TraceWriter(session_id, self.settings.log_dir).read()
            state = SessionState.model_validate(body["state"] | {"events": [e.model_dump() for e in events]})
        except (OSError, ValueError, KeyError) as error:
            logger.warning("could not read saved session: %s", error, extra={"session_id": session_id})
            return None
        entry = SessionEntry(state=state, kind=body.get("kind", state.produced_by), driver=body.get("driver", ""),
                             usage=body.get("usage") or {}, stats=body.get("stats"))
        entry.done.set()
        return entry

    def _write_trace(self, session_id: str, event) -> None:
        try:
            TraceWriter(session_id, self.settings.log_dir).write(event)
        except OSError as error:
            logger.warning("could not write trace: %s", error, extra={"session_id": session_id})

    def _evict(self) -> None:
        # Called with the lock held. Drops the oldest finished sessions; they stay readable from disk.
        excess = len(self._sessions) - self.settings.backend_max_sessions
        if excess <= 0:
            return
        finished = sorted((e for e in self._sessions.values() if e.done.is_set()), key=lambda e: e.state.created_at)
        for entry in finished[:excess]:
            del self._sessions[entry.session_id]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)