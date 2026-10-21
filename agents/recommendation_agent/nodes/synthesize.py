# Write a recommendation from the retrieved sources and check its disclaimer

import re
from typing import Callable, cast

from ..llm_client import call_qwen
from ..state import RecommendationState, RetrievedChunk

SYSTEM_PROMPT = (
    "You are the recommendation step of a knee-injury assistant. Using "
    "ONLY the provided source excerpts, write a short, plain-language "
    "recommendation for someone with this finding. Cite each source you "
    "use by its title in parentheses, e.g. (MCL Sprain: Symptoms, "
    "Treatment, Exercises & Rehab). Do not invent information that isn't "
    "in the excerpts. End with a sentence noting this may indicate a "
    "condition requiring professional assessment, and to consult a "
    "clinician."
)


# Accept a few common phrasings for each part of the disclaimer.
MAY_INDICATE_PATTERNS = [
    r"may indicate",
    r"could (be a sign of|indicate)",
    r"possible sign of",
]
CONSULT_PATTERNS = [
    r"consult a (clinician|doctor|physician|healthcare professional)",
    r"see a (clinician|doctor|physician|healthcare professional)",
    r"speak (with|to) a (clinician|doctor|physician|healthcare professional)",
]

REQUIRED_DISCLAIMER = (
    " This may indicate a condition that needs clinical attention - please "
    "consult a clinician for a full assessment."
)


def _has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def has_required_disclaimer(text: str) -> bool:
    """Check for both tentative wording and advice to consult a clinician."""
    return _has_pattern(text, MAY_INDICATE_PATTERNS) and _has_pattern(text, CONSULT_PATTERNS)


def enforce_disclaimer(text: str) -> tuple[str, bool]:
    """Return the text and whether the required disclaimer was added."""
    if has_required_disclaimer(text):
        return text, False
    return text.rstrip() + REQUIRED_DISCLAIMER, True


def _build_prompt(finding: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(
        f"Source: {chunk['source_title']}\nExcerpt: {chunk['text']}"
        for chunk in chunks
    )
    return f"Finding: {finding}\n\nSource excerpts:\n\n{excerpts}\n\nWrite the recommendation now."


def synthesize(
    state: RecommendationState,
    llm: Callable[[str, str | None], str] = call_qwen,
) -> RecommendationState:
    """Generate a recommendation from local and web sources, then check the disclaimer."""
    chunks = (state.get("retrieved_chunks") or []) + (state.get("web_results") or [])
    prompt = _build_prompt(state["finding"], chunks)  # pyright: ignore[reportTypedDictNotRequiredAccess]
    raw_recommendation = llm(prompt, SYSTEM_PROMPT)
    recommendation, disclaimer_added = enforce_disclaimer(raw_recommendation)

    new_state = cast(RecommendationState, dict(state))
    new_state["recommendation"] = recommendation
    new_state["disclaimer_added"] = disclaimer_added
    return new_state
