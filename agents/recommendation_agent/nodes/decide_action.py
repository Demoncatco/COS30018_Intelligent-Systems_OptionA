# Choose the next graph action using Qwen, then enforce workflow rules in code
# Parsing retries handle invalid replies; guards enforce the evidence rule, prevent repeated tools and premature completion

import json
import re
from typing import Callable

from ..llm_client import call_qwen
from ..state import RecommendationState
from .query_doc_store import DEFAULT_K

VALID_ACTIONS = {"query_doc_store", "web_search", "synthesize", "stop"}
TOOL_ACTIONS = ("query_doc_store", "web_search")
DEFAULT_MAX_STEPS = 6  # Force synthesis after this many decisions
MAX_ATTEMPTS = 2  # Initial prompt and one stricter retry
# Local chunks needed before synthesis; with fewer, web search must be tried first.
# Equal to the number of chunks query_doc_store returns, so a full local result never triggers web search
MIN_LOCAL_CHUNKS = DEFAULT_K

JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are the decision step of a knee-injury recommendation agent. "
    "Given the current state, decide the single next action. Respond with "
    "ONLY a JSON object of the form "
    '{"action": "query_doc_store"} or {"action": "web_search"} or '
    '{"action": "synthesize"} or {"action": "stop"}. No other text.'
)


def _build_prompt(state: RecommendationState, strict: bool) -> str:
    """Describe available evidence, completed actions and whether a recommendation exists
    The retry adds a stricter formatting instruction without changing the requested action rules
    """
    context = (
        f"Finding: {state.get('finding')}\n"
        f"Actions already taken: {state.get('actions_taken', []) or 'none'}\n"
        f"Doc-store chunks retrieved so far: {len(state.get('retrieved_chunks', []))}\n"
        f"Web results retrieved so far: {len(state.get('web_results', []))}\n"
        f"Recommendation written yet: {'yes' if state.get('recommendation') else 'no'}\n"
    )
    guidance = (
        "Rules: if no doc-store chunks have been retrieved yet, choose "
        f"query_doc_store. If you have fewer than {MIN_LOCAL_CHUNKS} doc-store "
        "chunks and haven't tried web_search yet, choose web_search. Otherwise, "
        "if no recommendation is written, choose synthesize. Once a "
        "recommendation is written, choose stop."
    )
    if strict:
        guidance += " Respond with ONLY the JSON object - no explanation, no markdown."
    return f"{context}\n{guidance}"


def _extract_action(raw_response: str) -> str | None:
    """Parse the matched brace-delimited JSON object and accept only known action values
    The greedy pattern spans the first opening brace through the last closing brace
    """
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


def _required_tool(state: RecommendationState) -> str | None:
    """Return the tool the evidence rule requires next, or None once synthesis may go ahead
    The local store always runs first; web search runs only when it returned fewer than
    MIN_LOCAL_CHUNKS chunks. A tool that has already run (even if it failed) is not required again
    """
    taken = state.get("actions_taken", [])
    if "query_doc_store" not in taken:
        return "query_doc_store"
    local = len(state.get("retrieved_chunks") or [])
    if local < MIN_LOCAL_CHUNKS and "web_search" not in taken:
        return "web_search"
    return None


def _fallback_action(state: RecommendationState) -> str:
    """Follow the evidence rule when the model reply cannot be parsed"""
    return _required_tool(state) or "synthesize"


def _apply_guards(action: str, state: RecommendationState) -> tuple[str, str | None]:
    """Enforce the evidence rule on the model's choice and redirect premature stops and repeated tools
    Return a note whenever code overrides the model so the trace explains the decision
    """
    required = _required_tool(state)
    local = len(state.get("retrieved_chunks") or [])

    # Gather the required evidence or write a recommendation before stopping
    if action == "stop":
        if required:
            return required, f"model chose stop before the evidence rule was met; running {required}"
        return "synthesize", "model chose stop before a recommendation existed"

    # The evidence rule: local store first, then web search only if it returned too few chunks
    if required and action != required:
        if required == "query_doc_store":
            reason = "the doc store has not been searched yet"
        else:
            reason = f"only {local} of {MIN_LOCAL_CHUNKS} required doc-store chunks were found"
        return required, f"model chose {action}, but {reason}; running {required}"

    # Once the rule is met, another search is not needed: either the local store found enough
    # chunks or both tools have already run
    if not required and action in TOOL_ACTIONS:
        if action == "web_search" and local >= MIN_LOCAL_CHUNKS:
            return "synthesize", f"{local} doc-store chunks found, so web search is not needed; moving on to synthesize"
        return "synthesize", f"{action} already ran; moving on to synthesize"

    return action, None


def decide_action(
    state: RecommendationState,
    llm: Callable[[str, str | None], str] = call_qwen,
) -> RecommendationState:
    """Increment the decision count, then stop, force synthesis or ask the model for an action
    Invalid replies get one stricter retry before a fallback action is checked by the guards
    """
    steps = state.get("step_count", 0)
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)

    new_state = state.copy()
    new_state["step_count"] = steps + 1
    new_state["action_used_fallback"] = False
    new_state["decision_note"] = ""

    # Completed text and the step limit take priority over another model call
    if state.get("recommendation"):
        new_state["action"] = "stop"
        new_state["decision_note"] = "recommendation already written"
        return new_state

    if steps >= max_steps:
        new_state["action"] = "synthesize"
        new_state["decision_note"] = f"step cap of {max_steps} reached; forcing synthesize"
        return new_state

    # Only formatting failures are retried here; service failures propagate to the graph caller
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
