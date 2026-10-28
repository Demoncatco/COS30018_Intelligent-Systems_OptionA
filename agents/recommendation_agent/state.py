# Data passed between nodes during one finding
# TypedDict describes the expected keys for type checking, without adding runtime validation

from typing import Required, TypedDict


class RetrievedChunk(TypedDict):
    """Common evidence fields consumed by synthesis for both local chunks and web results"""
    source_title: str
    source_url: str
    findings: list[str]
    text: str
    score: float


class RecommendationState(TypedDict, total=False):
    # The finding is required; other fields are filled in as the graph runs
    """Require the finding as input and allow other keys to be added as nodes complete"""
    finding: Required[str]
    user_query: str

    # Coordinator context is labelled as data in the synthesis prompt, not as model instructions
    impression: str           # the Diagnostic Agent's plain-language initial impression
    patient_context: str      # age, sex, symptoms and injury mechanism, as one labelled line

    # Loop control - set by graph.initial_state(), updated as the graph runs
    max_steps: int            # hard cap on decide_action calls before synthesis is forced
    step_count: int           # how many times decide_action has run
    actions_taken: list[str]  # nodes executed so far (decide_action excluded)
    trace: list[str]          # one human-readable line per node transition
    errors: list[str]         # tool failures that were caught instead of crashing the loop

    # Written by decide_action
    action: str               # query_doc_store | web_search | synthesize | stop
    action_used_fallback: bool  # True if Qwen's reply couldn't be parsed
    decision_note: str        # why code overrode the model's choice, if it did

    # Written by query_doc_store / web_search
    retrieved_chunks: list[RetrievedChunk]
    web_results: list[RetrievedChunk]
    web_results_filtered_count: int

    # Written by synthesize, including its disclaimer check
    recommendation: str
    disclaimer_added: bool
