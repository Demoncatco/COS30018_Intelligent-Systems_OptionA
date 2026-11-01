"""Request and response bodies of the backend API. Agent contracts stay in shared/schema.py."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from shared.schema import AgentEvent, FinalReport, PatientMetadata, SessionStatus

Driver = Literal["tools", "json", "fixed"]
SessionKind = Literal["multi_agent", "baseline"]


class SessionCreate(BaseModel):
    # DICOM UIDs are digits and dots; letters, "_" and "-" are allowed for test ids. Nothing that could be a path.
    study_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", examples=["1.2.826.0.1.3680043.8.498.10001"])
    patient_metadata: PatientMetadata = Field(default_factory=PatientMetadata)
    driver: Driver = Field(default="tools", description="Coordinator driver; ignored by POST /baseline.")


class SessionSummary(BaseModel):
    session_id: str
    study_id: str
    produced_by: SessionKind
    driver: str
    status: SessionStatus
    finished: bool
    created_at: float
    updated_at: float
    n_events: int
    links: dict[str, str] = Field(default_factory=dict)


class SessionDetail(SessionSummary):
    patient_metadata: PatientMetadata
    final_report: FinalReport | None = None
    errors: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    recheck_rounds: int = 0
    user_action: str = ""
    usage: dict = Field(default_factory=dict, description="LLM calls, tokens and latency per agent for this session.")
    stats: dict | None = Field(default=None, description="Coordinator stats (valid-call rate, overrides, ...).")
    events: list[AgentEvent] | None = Field(default=None, description="Only with ?include_events=true.")
    scores: dict[str, float] | None = Field(
        default=None, description="All 12 study probabilities from the latest scoring, reported or not (for the UI).")


class EventsPage(BaseModel):
    session_id: str
    status: SessionStatus
    finished: bool
    events: list[AgentEvent]
    next_after: int = Field(description="Pass as ?after= to get only newer events.")