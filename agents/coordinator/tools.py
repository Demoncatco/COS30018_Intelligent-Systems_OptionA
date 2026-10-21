"""The Coordinator's four tools as Pydantic argument models and Ollama tool schemas.

study_id is not a tool argument, so the LLM cannot mistype it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from shared.llm import tool_schema
from shared.schema import FINDINGS

RUN_DIAGNOSTIC = "run_diagnostic"
RUN_RECOMMENDATION = "run_recommendation"
REQUEST_REUPLOAD = "request_reupload"
FINALIZE_REPORT = "finalize_report"

TOOL_NAMES: tuple[str, ...] = (RUN_DIAGNOSTIC, RUN_RECOMMENDATION, REQUEST_REUPLOAD, FINALIZE_REPORT)
TERMINAL_TOOLS: frozenset[str] = frozenset({REQUEST_REUPLOAD, FINALIZE_REPORT})

_FINDING_LOOKUP = {name.lower(): name for name in FINDINGS}


def _truncate(limit: int):
    # Cut long text instead of rejecting the call.
    def check(value: object) -> object:
        return value[:limit] if isinstance(value, str) else value
    return check


def canonical_findings(names: list[str]) -> list[str]:
    """Case-insensitive match to FINDINGS; raises ValueError on unknown names."""
    unknown = [n for n in names if n.strip().lower() not in _FINDING_LOOKUP]
    if unknown:
        raise ValueError(f"unknown finding names {unknown}; use exactly: {', '.join(FINDINGS)}")
    wanted = {_FINDING_LOOKUP[n.strip().lower()] for n in names}
    return [name for name in FINDINGS if name in wanted]


class RunDiagnosticArgs(BaseModel):
    mode: Literal["initial", "recheck"] = Field(
        default="initial",
        description='"initial" for the first run. "recheck" to have the Diagnostic Agent re-examine specific findings.',
    )
    focus_findings: list[str] = Field(
        default_factory=list,
        description=f"Recheck only: the findings to re-examine, spelled exactly as one of: {', '.join(FINDINGS)}.",
    )
    reason: str = Field(
        default="",
        description="Why you are calling this. For a recheck, the specific question for the Diagnostic Agent.",
    )

    _short_reason = field_validator("reason", mode="before")(_truncate(500))

    @field_validator("focus_findings")
    @classmethod
    def _known_findings(cls, names: list[str]) -> list[str]:
        return canonical_findings(names)

    @model_validator(mode="after")
    def _recheck_needs_focus(self) -> RunDiagnosticArgs:
        if self.mode == "recheck" and not self.focus_findings:
            raise ValueError('mode "recheck" needs at least one name in focus_findings')
        if self.mode == "initial" and self.focus_findings:
            self.focus_findings = []  # ignored on an initial run
        return self


class RunRecommendationArgs(BaseModel):
    reason: str = Field(default="", description="Why you are asking the Recommendation Agent now.")

    _short_reason = field_validator("reason", mode="before")(_truncate(500))


class RequestReuploadArgs(BaseModel):
    user_action: str = Field(
        min_length=1,
        description="Plain instruction for the user, e.g. 'Please re-upload the study including an axial series.'",
    )
    reason: str = Field(default="", description="Why the current images cannot be used.")

    _short_text = field_validator("user_action", "reason", mode="before")(_truncate(500))


class FinalizeReportArgs(BaseModel):
    summary: str = Field(default="", description="One or two sentences on how the agents' results were combined.")
    unresolved: list[str] = Field(
        default_factory=list,
        description="Disagreements between the Diagnostic and Recommendation Agents that you could not resolve.",
    )

    _short_summary = field_validator("summary", mode="before")(_truncate(1000))

    @field_validator("unresolved", mode="before")
    @classmethod
    def _short_items(cls, items: object) -> object:
        if isinstance(items, str):  # accept a single string
            items = [items] if items.strip() else []
        return [str(i)[:300] for i in items][:10] if isinstance(items, list) else items


TOOL_ARGS: dict[str, type[BaseModel]] = {
    RUN_DIAGNOSTIC: RunDiagnosticArgs,
    RUN_RECOMMENDATION: RunRecommendationArgs,
    REQUEST_REUPLOAD: RequestReuploadArgs,
    FINALIZE_REPORT: FinalizeReportArgs,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    RUN_DIAGNOSTIC: (
        "Run the Diagnostic Agent (junior doctor) on this session's MRI study. It returns probabilities for the "
        "12 findings, its confidence, image quality flags and a short impression. Use mode 'initial' first; use "
        "mode 'recheck' with focus_findings when the Recommendation Agent questions specific findings."
    ),
    RUN_RECOMMENDATION: (
        "Send the latest diagnostic result, the impression and the patient details to the Recommendation Agent "
        "(senior doctor). It returns recommendations, an urgency level, the evidence status, and any concerns or "
        "findings it wants rechecked."
    ),
    REQUEST_REUPLOAD: (
        "End the session and ask the user for better images. Only when the images are not usable "
        "(needs_reupload is true) or the study could not be processed."
    ),
    FINALIZE_REPORT: "End the session and produce the final report from the results gathered so far.",
}

TOOLS: list[dict] = [tool_schema(name, TOOL_DESCRIPTIONS[name], TOOL_ARGS[name]) for name in TOOL_NAMES]


# JSON driver for models without tool calling (MedGemma).
class NextAction(BaseModel):
    tool: Literal["run_diagnostic", "run_recommendation", "request_reupload", "finalize_report"]
    arguments: dict = Field(default_factory=dict)
    reason: str = ""