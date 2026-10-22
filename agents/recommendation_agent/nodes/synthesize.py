# Write a recommendation from retrieved sources and check its disclaimer

import re
from typing import Callable

from ..llm_client import call_qwen
from ..state import RecommendationState, RetrievedChunk

SYSTEM_PROMPT = (
    "You are the recommendation step of a knee-injury assistant. Using "
    "ONLY the provided source excerpts, write a short, plain-language "
    "recommendation for someone with this finding. Cite each source you "
    "use by its title in parentheses, e.g. (MCL Sprain: Symptoms, "
    "Treatment, Exercises & Rehab). Do not invent information that isn't "
    "in the excerpts. Finish by copying this sentence exactly, word for "
    'word: "This may indicate a condition requiring professional '
    'assessment, so please consult a clinician."'
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
    """Check for tentative wording and advice to consult a clinician."""
    return _has_pattern(text, MAY_INDICATE_PATTERNS) and _has_pattern(text, CONSULT_PATTERNS)


def enforce_disclaimer(text: str) -> tuple[str, bool]:
    """Return the text and whether the required disclaimer was added."""
    if has_required_disclaimer(text):
        return text, False
    return text.rstrip() + REQUIRED_DISCLAIMER, True


NO_SOURCES_TEXT = (
    "No trusted sources could be retrieved for this finding, so no specific "
    "guidance can be given here."
)


def _url_key(url: str) -> str:
    """Normalize URL spelling so the same page is only listed once."""
    key = url.strip().lower().split("#")[0]
    key = re.sub(r"^https?://(www\.)?", "", key)
    return key.rstrip("/")


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """List each source once, keeping its original title and URL."""
    seen: set[str] = set()
    lines = []
    for chunk in chunks:
        url = chunk["source_url"]
        key = _url_key(url)
        if key not in seen:
            seen.add(key)
            lines.append(f"- {chunk['source_title']} - {url}")
    return "Sources:\n" + "\n".join(lines)


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
    """Write a recommendation from the available sources and add its disclaimer and source list."""
    chunks = (state.get("retrieved_chunks") or []) + (state.get("web_results") or [])
    if chunks:
        raw_recommendation = llm(_build_prompt(state["finding"], chunks), SYSTEM_PROMPT)
    else:
        # Avoid generating advice without source excerpts.
        raw_recommendation = NO_SOURCES_TEXT
    recommendation, disclaimer_added = enforce_disclaimer(raw_recommendation)
    if chunks:
        recommendation = f"{recommendation}\n\n{format_sources(chunks)}"

    new_state = state.copy()
    new_state["recommendation"] = recommendation
    new_state["disclaimer_added"] = disclaimer_added
    return new_state
