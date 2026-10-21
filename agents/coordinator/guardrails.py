"""Coordinator guardrails: which actions may run next, and the fixed fallback policy.

Hard refusals are never allowed; soft refusals pass if the LLM insists, and open items go to disagreements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from agents.coordinator.tools import (
    FINALIZE_REPORT,
    REQUEST_REUPLOAD,
    RUN_DIAGNOSTIC,
    RUN_RECOMMENDATION,
    FinalizeReportArgs,
    RequestReuploadArgs,
    RunDiagnosticArgs,
    RunRecommendationArgs,
)
from shared.config import Settings
from shared.schema import (
    FINDINGS,
    DiagnosticAgentResponse,
    DiagnosticDecision,
    RecommendationResponse,
    SessionState,
)

MAX_DIAGNOSTIC_ATTEMPTS = 2  # try + one retry
MAX_RECOMMENDATION_ATTEMPTS = 2  # per diagnostic result


@dataclass
class RunContext:
    """Coordinator bookkeeping on top of SessionState."""

    state: SessionState
    settings: Settings
    diagnostic_failures: int = 0  # before any result
    recommendation_failures: int = 0  # for the current result
    reviewed_index: int | None = None  # diagnostic the senior last reviewed
    reupload_requested: bool = False
    finished: bool = False
    warned: set[str] = field(default_factory=set)  # soft warnings given

    @property
    def latest(self) -> DiagnosticAgentResponse | None:
        return self.state.latest_diagnostic

    @property
    def latest_recommendation(self) -> RecommendationResponse | None:
        return self.state.latest_recommendation

    @property
    def diagnostic_index(self) -> int:
        return len(self.state.diagnostic_results) - 1

    @property
    def has_current_review(self) -> bool:
        return (self.reviewed_index is not None and self.reviewed_index == self.diagnostic_index
                and self.latest_recommendation is not None)

    @property
    def rechecks_remaining(self) -> int:
        return max(0, self.settings.max_recheck_rounds - self.state.recheck_rounds)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    message: str = ""
    soft: bool = False  # LLM may insist
    args: BaseModel | None = None  # args to run with
    note: str = ""  # for the trace


# --- State predicates ---

def needs_reupload(result: DiagnosticAgentResponse | None) -> bool:
    # Not finalised and quality flags set.
    return result is not None and result.decision != DiagnosticDecision.FINALIZE and bool(result.quality_flags)


def reported_findings(result: DiagnosticAgentResponse, settings: Settings) -> list[str]:
    return [name for name in FINDINGS if result.findings[name] >= settings.report_threshold]


def low_confidence_findings(result: DiagnosticAgentResponse, settings: Settings) -> list[str]:
    return [name for name in reported_findings(result, settings)
            if result.per_finding_confidence[name] < settings.low_confidence_threshold]


def open_recheck_requests(ctx: RunContext) -> list[str]:
    # Senior's requests for the current result.
    if not ctx.has_current_review:
        return []
    wanted = set(ctx.latest_recommendation.suggested_recheck)
    return [name for name in FINDINGS if name in wanted]


def recheck_candidates(ctx: RunContext) -> list[str]:
    """Findings a recheck may target: the senior's requests, or low-confidence ones before any review."""
    result = ctx.latest
    if result is None or needs_reupload(result):
        return []
    if ctx.has_current_review:
        return open_recheck_requests(ctx)
    if ctx.reviewed_index is None:
        return low_confidence_findings(result, ctx.settings)
    return []


# --- Refusal reasons (None = allowed) ---

def _initial_diagnostic_refusal(ctx: RunContext) -> str | None:
    if ctx.state.diagnostic_results:
        return ("The initial diagnostic is already done. Use mode 'recheck' with focus_findings if the senior "
                "asked for one, otherwise move on.")
    if ctx.diagnostic_failures >= MAX_DIAGNOSTIC_ATTEMPTS:
        return (f"The Diagnostic Agent failed {ctx.diagnostic_failures} times. Call request_reupload or "
                "finalize_report.")
    return None


def _recheck_refusal(ctx: RunContext) -> str | None:
    result = ctx.latest
    if result is None:
        return "There is no diagnostic result to recheck yet. Call run_diagnostic with mode 'initial' first."
    if needs_reupload(result):
        return "The images are not usable (needs_reupload). Call request_reupload instead of a recheck."
    if not ctx.state.can_recheck(ctx.settings.max_recheck_rounds):
        return (f"The recheck limit ({ctx.settings.max_recheck_rounds} rounds) is reached. Call finalize_report; "
                "anything still disputed will be listed as unresolved.")
    if not recheck_candidates(ctx):
        if ctx.reviewed_index is not None and not ctx.has_current_review:
            return "The senior has not reviewed the recheck result yet. Call run_recommendation first."
        return ("Nothing to recheck: the senior raised no recheck requests and no reported finding is "
                "low-confidence.")
    return None


def _recommendation_refusal(ctx: RunContext) -> str | None:
    result = ctx.latest
    if result is None:
        return "There is no diagnostic result yet. Call run_diagnostic first."
    if needs_reupload(result):
        return "The images are not usable (needs_reupload). Call request_reupload, not run_recommendation."
    if ctx.has_current_review:
        return ("The senior has already reviewed the current diagnostic result. Recheck the findings it "
                "questioned, or call finalize_report.")
    if ctx.recommendation_failures >= MAX_RECOMMENDATION_ATTEMPTS:
        return (f"The Recommendation Agent failed {ctx.recommendation_failures} times. Call finalize_report; the "
                "report will be marked partial.")
    return None


def _reupload_refusal(ctx: RunContext) -> str | None:
    if needs_reupload(ctx.latest):
        return None
    if ctx.latest is None and ctx.diagnostic_failures > 0:
        return None  # study could not be processed
    if ctx.latest is None:
        return "Call run_diagnostic first; there is no evidence yet that the images are unusable."
    return ("The images were usable (no quality problem stopped the Diagnostic Agent). Do not ask the user to "
            "re-upload; continue with the results you have.")


def _finalize_hard_refusal(ctx: RunContext) -> str | None:
    if ctx.latest is None and ctx.diagnostic_failures == 0:
        return "Nothing has been done yet. Call run_diagnostic first."
    if needs_reupload(ctx.latest):
        return "The images are not usable (needs_reupload). Call request_reupload so the user knows what to do."
    return None


def _finalize_soft_refusal(ctx: RunContext) -> tuple[str, str] | None:
    # (key, message); each warning given once.
    if ctx.latest is None:
        if ctx.diagnostic_failures < MAX_DIAGNOSTIC_ATTEMPTS:
            return "retry_diagnostic", "The Diagnostic Agent failed once. You may retry run_diagnostic before giving up."
        return None
    if (not ctx.has_current_review and ctx.recommendation_failures < MAX_RECOMMENDATION_ATTEMPTS):
        return (f"no_review:{ctx.diagnostic_index}",
                ("The senior Recommendation Agent has not reviewed the current diagnostic result. Call "
                 "run_recommendation, or call finalize_report again to finish without it (the report will be "
                 "partial)."))
    requests = open_recheck_requests(ctx)
    if requests and ctx.rechecks_remaining > 0:
        return (f"open_recheck:{ctx.diagnostic_index}",
                (f"The senior asked for a recheck of {requests} and {ctx.rechecks_remaining} recheck round(s) "
                 "remain. The senior outranks the junior: recheck, or call finalize_report again to finish and "
                 "list it as unresolved."))
    return None


# --- Public API ---

def allowed_tools(ctx: RunContext) -> list[str]:
    """Tools not hard-refused right now."""
    if ctx.finished:
        return []
    tools = []
    if _initial_diagnostic_refusal(ctx) is None or _recheck_refusal(ctx) is None:
        tools.append(RUN_DIAGNOSTIC)
    if _recommendation_refusal(ctx) is None:
        tools.append(RUN_RECOMMENDATION)
    if _reupload_refusal(ctx) is None:
        tools.append(REQUEST_REUPLOAD)
    if _finalize_hard_refusal(ctx) is None:
        tools.append(FINALIZE_REPORT)
    return tools


def check_call(ctx: RunContext, name: str, args: BaseModel) -> Verdict:
    """Accepts, narrows or refuses a validated tool call."""
    if ctx.finished:
        return Verdict(False, "The session is already finished.")

    if name == RUN_DIAGNOSTIC:
        assert isinstance(args, RunDiagnosticArgs)
        if args.mode == "initial":
            refusal = _initial_diagnostic_refusal(ctx)
            return Verdict(False, refusal) if refusal else Verdict(True, args=args)
        refusal = _recheck_refusal(ctx)
        if refusal:
            return Verdict(False, refusal)
        candidates = recheck_candidates(ctx)
        focus = [n for n in args.focus_findings if n in candidates]
        if not focus:
            return Verdict(False, f"focus_findings must include at least one of {candidates}.")
        dropped = [n for n in args.focus_findings if n not in candidates]
        note = f"dropped {dropped} from focus (not under dispute)" if dropped else ""
        return Verdict(True, args=args.model_copy(update={"focus_findings": focus}), note=note)

    if name == RUN_RECOMMENDATION:
        refusal = _recommendation_refusal(ctx)
        return Verdict(False, refusal) if refusal else Verdict(True, args=args)

    if name == REQUEST_REUPLOAD:
        refusal = _reupload_refusal(ctx)
        return Verdict(False, refusal) if refusal else Verdict(True, args=args)

    if name == FINALIZE_REPORT:
        refusal = _finalize_hard_refusal(ctx)
        if refusal:
            return Verdict(False, refusal)
        soft = _finalize_soft_refusal(ctx)
        if soft:
            key, message = soft
            if key not in ctx.warned:
                ctx.warned.add(key)
                return Verdict(False, message, soft=True)
            return Verdict(True, args=args, note=f"finalised after warning ({key})")
        return Verdict(True, args=args)

    return Verdict(False, f"Unknown tool {name!r}.")


def reupload_instruction(result: DiagnosticAgentResponse | None) -> str:
    if result is None:
        return "The study could not be processed. Please check the study and upload it again."
    missing = [flag.split(":", 1)[1] for flag in result.quality_flags if flag.startswith("missing_plane:")]
    parts = []
    if missing:
        parts.append(f"including a {' and '.join(missing)} series")
    if "low_slice_count" in result.quality_flags:
        parts.append("with complete series (at least 15 slices per plane)")
    detail = " ".join(parts) if parts else "with complete, good-quality series for all three planes"
    return f"Please re-upload the knee MRI study {detail}."


def fallback_action(ctx: RunContext) -> tuple[str, BaseModel]:
    """Reference policy: the next action without an LLM."""
    result = ctx.latest
    if result is None:
        if ctx.diagnostic_failures < MAX_DIAGNOSTIC_ATTEMPTS:
            return RUN_DIAGNOSTIC, RunDiagnosticArgs(mode="initial", reason="start the session")
        return FINALIZE_REPORT, FinalizeReportArgs(summary="The Diagnostic Agent could not process the study.")
    if needs_reupload(result):
        return REQUEST_REUPLOAD, RequestReuploadArgs(user_action=reupload_instruction(result),
                                                     reason=f"quality flags: {result.quality_flags}")
    if not ctx.has_current_review and ctx.recommendation_failures < MAX_RECOMMENDATION_ATTEMPTS:
        return RUN_RECOMMENDATION, RunRecommendationArgs(reason="senior review of the current diagnostic result")
    requests = open_recheck_requests(ctx)
    if requests and ctx.state.can_recheck(ctx.settings.max_recheck_rounds):
        concerns = "; ".join(ctx.latest_recommendation.concerns) or "senior review questioned these findings"
        return RUN_DIAGNOSTIC, RunDiagnosticArgs(mode="recheck", focus_findings=requests, reason=concerns)
    return FINALIZE_REPORT, FinalizeReportArgs(summary="")


def forced_stop_action(ctx: RunContext) -> tuple[str, BaseModel]:
    """End action when the step limit is hit."""
    if needs_reupload(ctx.latest):
        return REQUEST_REUPLOAD, RequestReuploadArgs(user_action=reupload_instruction(ctx.latest),
                                                     reason="step limit reached")
    return FINALIZE_REPORT, FinalizeReportArgs(summary="Stopped at the Coordinator step limit.")