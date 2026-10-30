"""
The short plain-language impression passed on to the Recommendation Agent.

MedGemma words it (llm.decide with a JSON schema), from facts the code has already worked out. The code, not the model,
decides which findings are reported, which are uncertain and which planes are missing; the model only phrases them.
If the LLM fails or its text breaks a guardrail, the rule-based text below is used instead, so an impression is always
produced and never states a finding the numbers do not support.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from shared.schema import FINDINGS

if TYPE_CHECKING:
    from shared.llm import LLMClient

log = logging.getLogger(__name__)

STRONG_PROBABILITY = 0.75   # at or above this (and the report threshold) a finding is "likely", below it "possible"
MAX_IMPRESSION_CHARS = 300
STUB_NOTE = "These are placeholder scores from a stub model, not a real analysis."

SYSTEM_PROMPT = (
    "You are the junior diagnostic assistant for a knee MRI screening tool. You are given findings that software has "
    "already decided. Write a short plain-language impression of them, at most 2 sentences and 300 characters. "
    "Mention only the findings listed as reported, uncertain or rechecked. Do not add findings, numbers, causes, "
    "severities or advice. Reply with JSON: {\"impression\": \"...\"}."
)


class ImpressionDraft(BaseModel):
    impression: str = Field(min_length=1, max_length=MAX_IMPRESSION_CHARS)


def rule_impression(findings: dict[str, float], report_threshold: float, uncertain: list[str],
                    quality_flags: list[str], focus: list[str] | None = None,
                    per_plane: dict[str, dict[str, float]] | None = None) -> str:
    strong = max(STRONG_PROBABILITY, report_threshold)
    ranked = sorted(findings, key=lambda n: -findings[n])
    likely = [n for n in ranked if findings[n] >= strong]
    possible = [n for n in ranked if report_threshold <= findings[n] < strong]

    parts = []
    if likely:
        parts.append("Likely " + ", ".join(likely) + ".")
    if possible:
        parts.append("Possible " + ", ".join(possible) + ".")
    if not likely and not possible:
        parts.append("No finding above the reporting threshold.")
    if uncertain:   # ordered by plane spread, largest first; the full list is in cross_plane_variance
        shown = ", ".join(uncertain[:3]) + (f" and {len(uncertain) - 3} more" if len(uncertain) > 3 else "")
        parts.append(f"The imaging planes disagree most on {shown}, so these are uncertain.")
    missing = _missing_planes(quality_flags)
    if missing:
        parts.append("Not assessed in the " + ", ".join(missing) + " plane" + ("s" if len(missing) > 1 else "") + ".")
    if focus:
        detail = []
        for n in focus:
            planes = ", ".join(f"{pl} {s[n]:.2f}" for pl, s in (per_plane or {}).items())
            detail.append(f"{n} {findings[n]:.2f}" + (f" ({planes})" if planes else ""))
        parts.append("Recheck of " + "; ".join(detail) + ".")
    return " ".join(parts)


def _missing_planes(quality_flags: list[str]) -> list[str]:
    return [f.split(":", 1)[1] for f in quality_flags if f.startswith("missing_plane:")]


def _mentioned(text: str) -> set[str]:
    # Finding names are matched as whole words, so "ACL" is not found inside another word.
    return {n for n in FINDINGS if re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", text, re.I)}


def _passes_guardrails(text: str, allowed: set[str]) -> bool:
    return 0 < len(text) <= MAX_IMPRESSION_CHARS and _mentioned(text) <= allowed


def build_impression(llm: LLMClient | None, session_id: str, findings: dict[str, float], report_threshold: float,
                     uncertain: list[str], quality_flags: list[str], focus: list[str] | None = None,
                     per_plane: dict[str, dict[str, float]] | None = None) -> tuple[str, str]:
    """Returns (impression, source) where source is 'llm' or 'rules'. Never raises."""
    rules = rule_impression(findings, report_threshold, uncertain, quality_flags, focus, per_plane)
    stub = STUB_NOTE if "stub_model" in quality_flags else ""
    if llm is None:
        return _with_stub(rules, stub), "rules"

    reported = [n for n in sorted(findings, key=lambda n: -findings[n]) if findings[n] >= report_threshold]
    facts = (f"Reported (likely or possible): {', '.join(reported) or 'none'}\n"
             f"Uncertain (imaging planes disagree): {', '.join(uncertain) or 'none'}\n"
             f"Planes not assessed: {', '.join(_missing_planes(quality_flags)) or 'none'}\n"
             f"Being rechecked: {', '.join(focus or []) or 'none'}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": facts}]
    try:
        draft, _ = llm.decide("diagnostic", messages, ImpressionDraft, session_id=session_id)
    except Exception as error:  # noqa: BLE001 - any LLM problem falls back to the rule-based text
        log.warning("impression LLM failed (%s: %s); using rule-based text", type(error).__name__, error)
        return _with_stub(rules, stub), "rules"

    text = draft.impression.strip()
    if not _passes_guardrails(text, set(reported) | set(uncertain) | set(focus or [])):
        log.warning("impression LLM output broke a guardrail; using rule-based text")
        return _with_stub(rules, stub), "rules"
    return _with_stub(text, stub), "llm"


def _with_stub(text: str, stub: str) -> str:
    return f"{text} {stub}".strip() if stub else text
