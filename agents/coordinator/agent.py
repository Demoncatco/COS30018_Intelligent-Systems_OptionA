"""Coordinator Agent: Qwen tool-calling loop over the Diagnostic and Recommendation Agents, with guardrails.

Drivers: "tools" (Qwen), "json" (models without tool calling), "fixed" (no LLM, fixed-workflow baseline).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel

from agents.coordinator import guardrails
from agents.coordinator.guardrails import RunContext
from agents.coordinator.prompts import (
    SYSTEM_PROMPT,
    json_driver_system_prompt,
    nudge,
    opening_message,
)
from agents.coordinator.report import build_final_report
from agents.coordinator.runners import (
    DiagnosticRunner,
    RecommendationRunner,
    call_with_timeout,
)
from agents.coordinator.tools import (
    FINALIZE_REPORT,
    REQUEST_REUPLOAD,
    RUN_DIAGNOSTIC,
    RUN_RECOMMENDATION,
    TOOL_ARGS,
    TOOLS,
    FinalizeReportArgs,
    NextAction,
    RequestReuploadArgs,
    RunDiagnosticArgs,
)
from shared.config import Settings, get_settings
from shared.llm import (
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMResult,
    ToolArgumentsError,
    ToolCall,
    tool_message,
)
from shared.logging_setup import TraceWriter
from shared.schema import (
    FINDINGS,
    AgentEvent,
    AgentName,
    CoordinatorRequest,
    DiagnosticAgentRequest,
    DiagnosticAgentResponse,
    FinalReport,
    PatientMetadata,
    RecommendationResponse,
    SessionState,
    SessionStatus,
)

logger = logging.getLogger(__name__)

Driver = Literal["tools", "json", "fixed"]
Source = Literal["llm", "fallback", "guardrail"]

MAX_NO_TOOL_REPLIES = 2  # then the fixed policy takes a step
MAX_CONSECUTIVE_REFUSALS = 3  # then the fixed policy takes a step


@dataclass
class Decision:
    step: int
    source: Source  # who decided
    tool: str
    args: dict
    outcome: Literal["executed", "refused", "invalid"]
    note: str = ""


@dataclass
class CoordinatorStats:
    llm_calls: int = 0
    proposed_calls: int = 0
    valid_calls: int = 0  # known tool, valid args
    refused_calls: int = 0  # refused by a guardrail
    fallback_steps: int = 0
    forced_stops: int = 0  # step limit hit
    no_tool_replies: int = 0
    llm_errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_latency_ms: float = 0.0
    wall_ms: float = 0.0

    @property
    def valid_call_rate(self) -> float | None:
        return self.valid_calls / self.proposed_calls if self.proposed_calls else None

    def to_dict(self) -> dict:
        return asdict(self) | {"valid_call_rate": self.valid_call_rate}


@dataclass
class CoordinatorResult:
    state: SessionState
    report: FinalReport
    stats: CoordinatorStats
    decisions: list[Decision] = field(default_factory=list)

    @property
    def executed_tools(self) -> list[str]:
        return [d.tool for d in self.decisions if d.outcome == "executed"]

    @property
    def override_rate(self) -> float | None:
        # Share of decisions made by code, not the LLM.
        counted = [d for d in self.decisions if d.outcome != "invalid"]
        if not counted:
            return None
        overridden = sum(1 for d in counted if d.source != "llm" or d.outcome == "refused")
        return overridden / len(counted)


@dataclass
class _Run:
    ctx: RunContext
    stats: CoordinatorStats
    history: list[dict]
    trace: TraceWriter | None
    decisions: list[Decision] = field(default_factory=list)
    step: int = 0
    no_tool_streak: int = 0
    refusal_streak: int = 0
    llm_down: bool = False
    llm_summary: str = ""
    llm_unresolved: list[str] = field(default_factory=list)


class CoordinatorAgent:
    def __init__(self, diagnostic: DiagnosticRunner, recommendation: RecommendationRunner,
                 llm: LLMClient | None = None, settings: Settings | None = None, driver: Driver = "tools",
                 on_event: Callable[[AgentEvent], None] | None = None, write_traces: bool = True) -> None:
        if driver != "fixed" and llm is None:
            raise ValueError(f"driver {driver!r} needs an LLMClient; use driver='fixed' to run without an LLM")
        self.diagnostic = diagnostic
        self.recommendation = recommendation
        self.llm = llm
        self.settings = settings or (llm.settings if llm is not None else get_settings())
        self.driver: Driver = driver
        self.on_event = on_event
        self.write_traces = write_traces

    # --- Entry points ---

    def run(self, study_id: str, patient_metadata: PatientMetadata | None = None,
            session_id: str | None = None) -> CoordinatorResult:
        state = SessionState(session_id=session_id or uuid.uuid4().hex[:12], study_id=study_id,
                             patient_metadata=patient_metadata or PatientMetadata())
        return self.run_session(state)

    def run_session(self, state: SessionState) -> CoordinatorResult:
        """Runs one session to the end, updating `state` in place."""
        start = time.perf_counter()
        run = _Run(ctx=RunContext(state=state, settings=self.settings), stats=CoordinatorStats(),
                   history=self._opening_history(state),
                   trace=TraceWriter(state.session_id, self.settings.log_dir) if self.write_traces else None)
        self._emit(run, state.set_status(SessionStatus.RUNNING, f"Coordinator started ({self.driver} driver)"))
        try:
            self._loop(run, start)
        except Exception as error:  # end the session cleanly even on a bug
            logger.exception("coordinator crashed", extra={"session_id": state.session_id})
            self._emit(run, state.add_error(AgentName.COORDINATOR, f"internal error: {type(error).__name__}: {error}"))
            if not run.ctx.finished:
                self._finish(run, start, force_status=SessionStatus.FAILED)
        run.stats.wall_ms = (time.perf_counter() - start) * 1000
        return CoordinatorResult(state=state, report=state.final_report, stats=run.stats, decisions=run.decisions)

    # --- Main loop ---

    def _loop(self, run: _Run, start: float) -> None:
        ctx, state = run.ctx, run.ctx.state
        while not ctx.finished:
            if state.coordinator_steps >= self.settings.max_coordinator_steps:
                name, args = guardrails.forced_stop_action(ctx)
                run.stats.forced_stops += 1
                self._emit(run, state.add_event(AgentName.COORDINATOR, "decision",
                                                f"Step limit ({self.settings.max_coordinator_steps}) reached: {name}",
                                                {"guardrail": "max_coordinator_steps"}))
                self._run_tool(run, name, args, "guardrail", start)
                break
            state.coordinator_steps += 1
            run.step = state.coordinator_steps

            if self._code_should_decide(run):
                name, args = guardrails.fallback_action(ctx)
                if self.driver != "fixed":
                    run.stats.fallback_steps += 1
                run.no_tool_streak = run.refusal_streak = 0
                payload = self._run_tool(run, name, args, "fallback", start)
                if self.driver != "fixed" and not ctx.finished:
                    run.history.append({"role": "user", "content": (
                        f"[Guardrail] The coordinator ran {name} automatically. Result: {json.dumps(payload)}")})
                continue

            calls = self._propose(run)
            if calls is None:  # LLM down: fall back next step
                continue
            for call in calls:
                if ctx.finished:
                    self._emit(run, state.add_event(AgentName.COORDINATOR, "message",
                                                    f"Ignored extra tool call {call.name} after the session ended"))
                    break
                self._handle_call(run, call, start)

    def _code_should_decide(self, run: _Run) -> bool:
        return (self.driver == "fixed" or run.llm_down or run.no_tool_streak >= MAX_NO_TOOL_REPLIES
                or run.refusal_streak >= MAX_CONSECUTIVE_REFUSALS)

    # --- LLM proposals ---

    def _propose(self, run: _Run) -> list[ToolCall] | None:
        ctx = run.ctx
        try:
            if self.driver == "tools":
                result = self.llm.chat("coordinator", run.history, tools=TOOLS, session_id=ctx.state.session_id)
                self._count_llm(run, [result])
                run.history.append(result.assistant_message())
                self._emit_llm_decision(run, result, [c.name for c in result.tool_calls])
                if not result.tool_calls:
                    run.no_tool_streak += 1
                    run.stats.no_tool_replies += 1
                    run.history.append({"role": "user", "content": nudge(guardrails.allowed_tools(ctx))})
                    return []
                return result.tool_calls

            action, results = self.llm.decide("coordinator", run.history, NextAction, session_id=ctx.state.session_id)
            self._count_llm(run, results)
            run.history.append({"role": "assistant", "content": action.model_dump_json()})
            self._emit_llm_decision(run, results[-1], [action.tool], reason=action.reason)
            call = ToolCall(name=action.tool, arguments=action.arguments)
            if action.reason and "reason" not in call.arguments and action.tool != REQUEST_REUPLOAD:
                call.arguments = {**call.arguments, "reason": action.reason}
            return [call]
        except LLMOutputError as error:  # JSON driver: no valid JSON
            run.stats.llm_calls += self.settings.llm_json_retries + 1
            run.stats.proposed_calls += 1
            run.no_tool_streak += 1
            self._emit(run, ctx.state.add_error(AgentName.COORDINATOR, f"invalid action JSON: {error}",
                                                {"last_output": error.last_output[:300]}))
            run.decisions.append(Decision(run.step, "llm", "?", {}, "invalid", "unparseable JSON"))
            run.history.append({"role": "user", "content": self._nudge_text(ctx)})
            return []
        except LLMError as error:
            run.stats.llm_errors += 1
            run.llm_down = True
            self._emit(run, ctx.state.add_error(AgentName.COORDINATOR,
                                                f"LLM unavailable, switching to the fixed policy: {error}"))
            return None

    def _handle_call(self, run: _Run, call: ToolCall, start: float) -> None:
        ctx, state = run.ctx, run.ctx.state
        run.stats.proposed_calls += 1
        schema = TOOL_ARGS.get(call.name)
        try:
            if schema is None:
                raise ToolArgumentsError(call.name, f"Unknown tool {call.name!r}. Use one of {list(TOOL_ARGS)}.")
            args = call.parse(schema)
        except ToolArgumentsError as error:
            run.no_tool_streak += 1
            run.decisions.append(Decision(run.step, "llm", call.name, call.arguments, "invalid", str(error)))
            self._emit(run, state.add_event(AgentName.COORDINATOR, "error", f"Invalid tool call: {error}",
                                            {"tool": call.name, "arguments": call.arguments}))
            self._reply(run, call, {"ok": False, "error": str(error), "allowed_tools": guardrails.allowed_tools(ctx)})
            return
        run.stats.valid_calls += 1
        run.no_tool_streak = 0

        verdict = guardrails.check_call(ctx, call.name, args)
        if not verdict.allowed:
            run.stats.refused_calls += 1
            run.refusal_streak += 1
            run.decisions.append(Decision(run.step, "llm", call.name, args.model_dump(), "refused", verdict.message))
            self._emit(run, state.add_event(AgentName.COORDINATOR, "decision",
                                            f"Guardrail refused {call.name}: {verdict.message}",
                                            {"tool": call.name, "arguments": args.model_dump(), "soft": verdict.soft}))
            self._reply(run, call, {"ok": False, "refused": verdict.message,
                                    "allowed_tools": guardrails.allowed_tools(ctx)})
            return
        run.refusal_streak = 0
        payload = self._run_tool(run, call.name, verdict.args, "llm", start, note=verdict.note)
        if not ctx.finished:
            self._reply(run, call, payload)

    def _reply(self, run: _Run, call: ToolCall, payload: dict) -> None:
        if self.driver == "tools":
            run.history.append(tool_message(call, payload))
        else:
            run.history.append({"role": "user", "content": f"Result of {call.name}: {json.dumps(payload)}"})

    # --- Tool execution ---

    def _run_tool(self, run: _Run, name: str, args: BaseModel, source: Source, start: float,
                  note: str = "") -> dict:
        run.decisions.append(Decision(run.step, source, name, args.model_dump(), "executed", note))
        self._emit(run, run.ctx.state.add_event(AgentName.COORDINATOR, "tool_call", f"{name} ({source})",
                                                {"tool": name, "arguments": args.model_dump(), "note": note}))
        if name == RUN_DIAGNOSTIC:
            return self._run_diagnostic(run, args)
        if name == RUN_RECOMMENDATION:
            return self._run_recommendation(run)
        if name == REQUEST_REUPLOAD:
            assert isinstance(args, RequestReuploadArgs)
            run.ctx.reupload_requested = True
            run.ctx.state.user_action = args.user_action
            self._finish(run, start)
            return {"ok": True}
        assert name == FINALIZE_REPORT and isinstance(args, FinalizeReportArgs)
        run.llm_summary = args.summary if source == "llm" else ""
        run.llm_unresolved = list(args.unresolved) if source == "llm" else []
        self._finish(run, start)
        return {"ok": True}

    def _run_diagnostic(self, run: _Run, args: RunDiagnosticArgs) -> dict:
        ctx, state = run.ctx, run.ctx.state
        recheck = args.mode == "recheck"
        request = DiagnosticAgentRequest(
            study_id=state.study_id, focus_findings=args.focus_findings,
            task_type=f"recheck: {args.reason or 'senior review questioned ' + ', '.join(args.focus_findings)}"
            if recheck else "initial")
        if recheck:
            state.recheck_rounds += 1  # counts even if it fails
        previous = ctx.latest
        started = time.perf_counter()
        try:
            response = call_with_timeout(self.diagnostic, self.settings.tool_timeout_s, request,
                                         session_id=state.session_id)
            response = DiagnosticAgentResponse.model_validate(response, from_attributes=True)
            if response.study_id != state.study_id:
                raise ValueError(f"returned a result for study {response.study_id!r}, not {state.study_id!r}")
        except Exception as error:  # noqa: BLE001 - any agent failure is handled the same way
            if previous is None:
                ctx.diagnostic_failures += 1
            message = f"Diagnostic Agent failed: {type(error).__name__}: {error}"
            self._emit(run, state.add_error(AgentName.DIAGNOSTIC, message,
                                            {"latency_ms": round((time.perf_counter() - started) * 1000, 1)}))
            return {"ok": False, "error": message, "rechecks_remaining": ctx.rechecks_remaining,
                    "suggested_recheck": guardrails.open_recheck_requests(ctx),
                    "allowed_tools": guardrails.allowed_tools(ctx)}
        latency = (time.perf_counter() - started) * 1000
        state.diagnostic_results.append(response)
        ctx.recommendation_failures = 0
        for line in response.trace:
            self._emit(run, state.add_event(AgentName.DIAGNOSTIC, "message", line))
        payload = self._diagnostic_payload(ctx, response, previous if recheck else None, args.focus_findings)
        self._emit(run, state.add_event(AgentName.DIAGNOSTIC, "tool_result",
                                        f"decision={response.decision.value}, "
                                        f"overall_confidence={response.overall_confidence:.2f}",
                                        payload, latency_ms=latency))
        return payload

    def _run_recommendation(self, run: _Run) -> dict:
        ctx, state = run.ctx, run.ctx.state
        result = ctx.latest
        first_review = not state.recommendation_results
        task_type = ("initial" if first_review else
                     f"recheck: review the Diagnostic Agent's result after recheck round {state.recheck_rounds}")
        request = CoordinatorRequest(
            finding=result.findings, confidence=result.overall_confidence,
            study_context={"laterality": result.laterality, "quality_flags": result.quality_flags,
                           "diagnostic_decision": result.decision.value, "recheck_rounds": state.recheck_rounds},
            task_type=task_type, impression=result.impression, patient_metadata=state.patient_metadata)
        started = time.perf_counter()
        try:
            response = call_with_timeout(self.recommendation, self.settings.tool_timeout_s, request,
                                         session_id=state.session_id)
            response = RecommendationResponse.model_validate(response, from_attributes=True)
        except Exception as error:  # noqa: BLE001 - any agent failure is handled the same way
            ctx.recommendation_failures += 1
            message = f"Recommendation Agent failed: {type(error).__name__}: {error}"
            self._emit(run, state.add_error(AgentName.RECOMMENDATION, message,
                                            {"latency_ms": round((time.perf_counter() - started) * 1000, 1)}))
            return {"ok": False, "error": message, "allowed_tools": guardrails.allowed_tools(ctx)}
        latency = (time.perf_counter() - started) * 1000
        state.recommendation_results.append(response)
        ctx.reviewed_index = ctx.diagnostic_index
        for line in response.trace:
            self._emit(run, state.add_event(AgentName.RECOMMENDATION, "message", line))
        payload = {
            "ok": True, "urgency": response.urgency.value, "evidence_status": response.evidence_status.value,
            "citations": len(response.citations), "concerns": response.concerns,
            "suggested_recheck": guardrails.open_recheck_requests(ctx),
            "rechecks_remaining": ctx.rechecks_remaining, "allowed_tools": guardrails.allowed_tools(ctx),
        }
        self._emit(run, state.add_event(AgentName.RECOMMENDATION, "tool_result",
                                        f"urgency={response.urgency.value}, evidence={response.evidence_status.value}, "
                                        f"recheck requests={payload['suggested_recheck']}",
                                        payload, latency_ms=latency))
        return payload

    def _diagnostic_payload(self, ctx: RunContext, result: DiagnosticAgentResponse,
                            previous: DiagnosticAgentResponse | None, focus: list[str]) -> dict:
        reported = guardrails.reported_findings(result, self.settings)
        payload = {
            "ok": True, "result_of": "recheck" if previous is not None else "initial",
            "decision": result.decision.value, "overall_confidence": round(result.overall_confidence, 3),
            "reported_findings": [{"name": n, "probability": round(result.findings[n], 3),
                                   "confidence": round(result.per_finding_confidence[n], 3)} for n in reported],
            "low_confidence_findings": guardrails.low_confidence_findings(result, self.settings),
            "planes_disagree_on": [n for n in FINDINGS if n in result.cross_plane_variance],
            "quality_flags": result.quality_flags, "needs_reupload": guardrails.needs_reupload(result),
            "impression": result.impression, "rechecks_remaining": ctx.rechecks_remaining,
        }
        if previous is not None:
            payload["changed_since_previous"] = {
                n: {"before": round(previous.findings[n], 3), "after": round(result.findings[n], 3)} for n in focus}
        payload["allowed_tools"] = guardrails.allowed_tools(ctx)
        return payload

    # --- Ending ---

    def _finish(self, run: _Run, start: float, force_status: SessionStatus | None = None) -> None:
        ctx, state = run.ctx, run.ctx.state
        ctx.finished = True
        path = [d.tool for d in run.decisions if d.outcome == "executed"]
        report = build_final_report(ctx, llm_summary=run.llm_summary, llm_unresolved=run.llm_unresolved,
                                    decision_path=path, wall_ms=(time.perf_counter() - start) * 1000)
        if force_status is not None:
            report = report.model_copy(update={"status": force_status})
        state.final_report = report
        state.disagreements = list(report.disagreements)
        self._emit(run, state.add_event(AgentName.COORDINATOR, "message", "Final report assembled",
                                        {"status": report.status.value, "findings": [f.name for f in report.findings],
                                         "disagreements": len(report.disagreements),
                                         "recheck_rounds": report.recheck_rounds}))
        self._emit(run, state.set_status(report.status, f"Session {report.status.value}"))

    # --- Helpers ---

    def _opening_history(self, state: SessionState) -> list[dict]:
        if self.driver == "fixed":
            return []
        system = SYSTEM_PROMPT if self.driver == "tools" else json_driver_system_prompt()
        return [{"role": "system", "content": system},
                {"role": "user", "content": opening_message(state.patient_metadata, self.settings.max_recheck_rounds,
                                                            self.settings.max_coordinator_steps)}]

    def _nudge_text(self, ctx: RunContext) -> str:
        allowed = guardrails.allowed_tools(ctx)
        if self.driver == "tools":
            return nudge(allowed)
        return (f"Reply with ONLY a JSON object {{\"tool\": ..., \"arguments\": {{...}}, \"reason\": ...}}. "
                f"allowed_tools: {json.dumps(allowed)}")

    def _count_llm(self, run: _Run, results: list[LLMResult]) -> None:
        for result in results:
            run.stats.llm_calls += 1
            run.stats.tokens_in += result.tokens_in
            run.stats.tokens_out += result.tokens_out
            run.stats.llm_latency_ms += result.latency_ms

    def _emit_llm_decision(self, run: _Run, result: LLMResult, tools: list[str], reason: str = "") -> None:
        data = {"tool_calls": tools, "model": result.model}
        if result.thinking:
            data["thinking"] = result.thinking[:1000]
        if result.content and self.driver == "tools":
            data["content"] = result.content[:500]
        if reason:
            data["reason"] = reason[:500]
        message = f"LLM chose {', '.join(tools)}" if tools else "LLM replied without a tool call"
        self._emit(run, run.ctx.state.add_event(AgentName.COORDINATOR, "decision", message, data,
                                                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                                                latency_ms=result.latency_ms))

    def _emit(self, run: _Run, event: AgentEvent) -> None:
        if run.trace is not None:
            try:
                run.trace.write(event)
            except OSError as error:
                logger.warning("could not write trace: %s", error, extra={"session_id": event.session_id})
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # UI errors must not stop the session
                logger.exception("on_event callback failed", extra={"session_id": event.session_id})


__all__ = ["CoordinatorAgent", "CoordinatorResult", "CoordinatorStats", "Decision"]