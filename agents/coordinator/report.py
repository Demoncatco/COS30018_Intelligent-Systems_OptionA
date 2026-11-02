"""Builds the FinalReport. Status, findings, disagreements and the disclaimer are set by code, not the LLM.
"""

from __future__ import annotations

from agents.coordinator.guardrails import RunContext, open_recheck_requests
from agents.recommendation_agent.trusted_domains import is_trusted_domain
from shared.schema import (
    DISCLAIMER,
    DiagnosticDecision,
    EvidenceStatus,
    FinalReport,
    ReportedFinding,
    SessionStatus,
)

NO_EVIDENCE_TEXT = (
    "No credible source could be found for these findings, so no recommendation is given. "
    "Please discuss the findings with a qualified clinician."
)


def split_citations(urls: list[str]) -> tuple[list[str], list[str]]:
    # (kept, dropped): only http(s) links on the allowlist reach the report, once each.
    kept = [u for u in dict.fromkeys(urls) if is_trusted_domain(u)]
    return kept, [u for u in urls if u not in kept]


def decide_status(ctx: RunContext) -> SessionStatus:
    if ctx.reupload_requested:
        return SessionStatus.INCONCLUSIVE
    result = ctx.latest
    if result is None:
        return SessionStatus.FAILED
    review = ctx.latest_recommendation if ctx.reviewed_index is not None else None
    if (review is None
            or review.evidence_status == EvidenceStatus.NO_CREDIBLE_SOURCE_FOUND
            or result.decision != DiagnosticDecision.FINALIZE):
        return SessionStatus.PARTIAL
    return SessionStatus.COMPLETED


def unresolved_disagreements(ctx: RunContext, llm_notes: list[str]) -> list[str]:
    items: list[str] = []
    review = ctx.latest_recommendation
    if review is not None and ctx.has_current_review:
        rounds = ctx.state.recheck_rounds
        for name in open_recheck_requests(ctx):
            items.append(f"Senior review still questions {name} after {rounds} recheck round(s).")
        items.extend(f"Senior concern: {concern}" for concern in review.concerns)
    elif review is not None and ctx.latest is not None:
        items.append("The latest Diagnostic recheck was not reviewed again by the Recommendation Agent.")
    items.extend(f"Coordinator note: {note}" for note in llm_notes if note.strip())
    return list(dict.fromkeys(items))  # drop duplicates, keep order


def build_final_report(ctx: RunContext, *, llm_summary: str = "", llm_unresolved: list[str] | None = None,
                       decision_path: list[str] | None = None, wall_ms: float = 0.0) -> FinalReport:
    state, settings = ctx.state, ctx.settings
    status = decide_status(ctx)
    result = ctx.latest
    review = ctx.latest_recommendation if ctx.reviewed_index is not None else None
    disagreements = [] if status == SessionStatus.INCONCLUSIVE else unresolved_disagreements(ctx, llm_unresolved or [])

    findings: list[ReportedFinding] = []
    if result is not None and status != SessionStatus.INCONCLUSIVE:
        disputed = set(open_recheck_requests(ctx))  # senior still disputes these
        for name, probability in sorted(result.findings.items(), key=lambda item: -item[1]):
            if probability >= settings.report_threshold:
                findings.append(ReportedFinding(
                    name=name, probability=round(probability, 4),
                    per_plane={plane: round(scores[name], 4) for plane, scores in result.per_plane.items()},
                    low_confidence=(result.per_finding_confidence[name] < settings.low_confidence_threshold
                                    or name in disputed)))

    recommendation_text, citations, dropped = "", [], []
    if review is not None and status != SessionStatus.INCONCLUSIVE:
        if review.evidence_status == EvidenceStatus.NO_CREDIBLE_SOURCE_FOUND:
            recommendation_text = NO_EVIDENCE_TEXT  # show the gap, never invent
        else:
            recommendation_text = review.recommendation_text
            citations, dropped = split_citations(review.citations)  # second allowlist check before the UI

    path = " -> ".join(decision_path or [])
    summary = (f"Diagnostic runs: {len(state.diagnostic_results)} ({state.recheck_rounds} recheck round(s)); "
               f"senior reviews: {len(state.recommendation_results)}; errors: {len(state.errors)}."
               + (f" Path: {path}." if path else ""))
    if dropped:
        summary += f" Dropped {len(dropped)} citation(s) outside the trusted-domain allowlist."
    if llm_summary.strip():
        summary += f" Coordinator note: {llm_summary.strip()}"

    return FinalReport(
        session_id=state.session_id, study_id=state.study_id, status=status, produced_by="multi_agent",
        findings=findings,
        impression=result.impression if result is not None else "",
        recommendation_text=recommendation_text, citations=citations,
        urgency=review.urgency if review is not None and status != SessionStatus.INCONCLUSIVE else None,
        evidence_status=review.evidence_status if review is not None else None,
        overall_confidence=result.overall_confidence if result is not None else None,
        disagreements=disagreements, errors=list(state.errors), user_action=state.user_action,
        coordination_summary=summary, disclaimer=DISCLAIMER, recheck_rounds=state.recheck_rounds,
        total_tokens=state.total_tokens, total_latency_ms=round(wall_ms, 1),
    )