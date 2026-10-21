# Choose the next action, retrying invalid model responses once before using a default

import json
import re
from typing import Callable, cast

from ..llm_client import call_qwen
from ..state import RecommendationState

VALID_ACTIONS = {"query_doc_store", "web_search", "synthesize"}
DEFAULT_ACTION = "query_doc_store"  # Check local sources first if parsing fails.
MAX_ATTEMPTS = 2  # Initial prompt and one stricter retry

JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are the decision step of a knee-injury recommendation agent. "
    "Given the current state, decide the single next action. Respond with "
    "ONLY a JSON object of the form "
    '{"action": "query_doc_store"} or {"action": "web_search"} or '
    '{"action": "synthesize"}. No other text.'
)


def _build_prompt(state: RecommendationState, strict: bool) -> str:
    context = (
        f"Finding: {state.get('finding')}\n"
        f"Doc-store chunks retrieved so far: {len(state.get('retrieved_chunks', []))}\n"
        f"Web results retrieved so far: {len(state.get('web_results', []))}\n"
    )
    guidance = (
        "Rules: if no doc-store chunks have been retrieved yet, choose "
        "query_doc_store. If doc-store chunks exist but are thin (fewer "
        "than 2) and no web results exist yet, choose web_search. Once you "
        "have enough chunks (doc-store and/or web), choose synthesize."
    )
    if strict:
        guidance += " Respond with ONLY the JSON object - no explanation, no markdown."
    return f"{context}\n{guidance}"


def _extract_action(raw_response: str) -> str | None:
    """Read a valid action from the JSON in a model response, or return None."""
    match = JSON_BLOCK_PATTERN.search(raw_response)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action = parsed.get("action")
    if action not in VALID_ACTIONS:
        return None
    return action


def decide_action(
    state: RecommendationState,
    llm: Callable[[str, str | None], str] = call_qwen,
) -> RecommendationState:
    """Ask the model for an action, with a stricter retry if parsing fails."""
    action = None

    for attempt in range(MAX_ATTEMPTS):
        strict = attempt > 0
        prompt = _build_prompt(state, strict=strict)
        raw_response = llm(prompt, SYSTEM_PROMPT)
        action = _extract_action(raw_response)
        if action is not None:
            break

    new_state = cast(RecommendationState, dict(state))
    new_state["action"] = action or DEFAULT_ACTION
    new_state["action_used_fallback"] = action is None
    return new_state
