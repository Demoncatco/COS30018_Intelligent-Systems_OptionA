# State shared by the recommendation graph nodes

from typing import Required, TypedDict


class RetrievedChunk(TypedDict):
    source_title: str
    source_url: str
    findings: list[str]
    text: str
    score: float


class RecommendationState(TypedDict, total=False):
    # The finding is required; other fields are filled in as the graph runs.
    finding: Required[str]
    user_query: str

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

    # Written by synthesize / disclaimer_check
    recommendation: str
    disclaimer_added: bool
