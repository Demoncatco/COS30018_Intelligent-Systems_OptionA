# Choose the next action and prevent repeated tools or an early stop

import json
import re
from typing import Callable

from ..llm_client import call_qwen
from ..state import RecommendationState

VALID_ACTIONS = {"query_doc_store", "web_search", "synthesize", "stop"}
TOOL_ACTIONS = ("query_doc_store", "web_search")
DEFAULT_ACTION = "query_doc_store"  # Check local sources first.
DEFAULT_MAX_STEPS = 6  # Force synthesis after this many decisions.
MAX_ATTEMPTS = 2  # Initial prompt and one stricter retry

JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are the decision step of a knee-injury recommendation agent. "
    "Given the current state, decide the single next action. Respond with "
    "ONLY a JSON object of the form "
    '{"action": "query_doc_store"} or {"action": "web_search"} or '
    '{"action": "synthesize"} or {"action": "stop"}. No other text.'
)


def _build_prompt(state: RecommendationState, strict: bool) -> str:
    context = (
        f"Finding: {state.get('finding')}\n"
        f"Actions already taken: {state.get('actions_taken', []) or 'none'}\n"
        f"Doc-store chunks retrieved so far: {len(state.get('retrieved_chunks', []))}\n"
        f"Web results retrieved so far: {len(state.get('web_results', []))}\n"
        f"Recommendation written yet: {'yes' if state.get('recommendation') else 'no'}\n"
    )
    guidance = (
        "Rules: if no doc-store chunks have been retrieved yet, choose "
        "query_doc_store. If you have fewer than 2 chunks and haven't tried "
        "web_search yet, choose web_search. Once you have enough chunks and "
        "no recommendation is written, choose synthesize. Once a "
        "recommendation is written, choose stop."
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
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")
    if action not in VALID_ACTIONS:
        return None
    return action


def _total_chunks(state: RecommendationState) -> int:
    return len(state.get("retrieved_chunks") or []) + len(state.get("web_results") or [])


def _fallback_action(state: RecommendationState) -> str:
    """Use the available sources, or search locally if there are none."""
    return "synthesize" if _total_chunks(state) else DEFAULT_ACTION


def _apply_guards(action: str, state: RecommendationState) -> tuple[str, str | None]:
    """Replace repeated tools or an early stop. Return the action and an optional note."""
    taken = state.get("actions_taken", [])

    # Gather sources or write a recommendation before stopping.
    if action == "stop":
        if _total_chunks(state) == 0 and "query_doc_store" not in taken:
            return "query_doc_store", "model chose stop with nothing retrieved; checking the doc store first"
        return "synthesize", "model chose stop before a recommendation existed"

    # Repeating a tool that already ran returns the same results again.
    if action in TOOL_ACTIONS and action in taken:
        other = "web_search" if action == "query_doc_store" else "query_doc_store"
        if _total_chunks(state) == 0 and other not in taken:
            return other, f"{action} already ran; trying {other} instead"
        return "synthesize", f"{action} already ran; moving on to synthesize"

    return action, None


def decide_action(
    state: RecommendationState,
    llm: Callable[[str, str | None], str] = call_qwen,
) -> RecommendationState:
    """Choose the next action, applying retries, fallbacks, and the step limit."""
    steps = state.get("step_count", 0)
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)

    new_state = state.copy()
    new_state["step_count"] = steps + 1
    new_state["action_used_fallback"] = False
    new_state["decision_note"] = ""

    if state.get("recommendation"):
        new_state["action"] = "stop"
        new_state["decision_note"] = "recommendation already written"
        return new_state

    if steps >= max_steps:
        new_state["action"] = "synthesize"
        new_state["decision_note"] = f"step cap of {max_steps} reached; forcing synthesize"
        return new_state

    action = None
    for attempt in range(MAX_ATTEMPTS):
        prompt = _build_prompt(state, strict=attempt > 0)
        action = _extract_action(llm(prompt, SYSTEM_PROMPT))
        if action is not None:
            break

    if action is None:
        new_state["action_used_fallback"] = True
        action = _fallback_action(state)

    action, note = _apply_guards(action, state)
    new_state["action"] = action
    if note:
        new_state["decision_note"] = note
    return new_state
