# State shared by the recommendation graph nodes

from typing import TypedDict


class RetrievedChunk(TypedDict):
    source_title: str
    source_url: str
    findings: list[str]
    text: str
    score: float


class RecommendationState(TypedDict, total=False):
    # Inputs from the Coordinator. total=False also allows partial test states.
    finding: str
    user_query: str

    # Written by decide_action
    action: str
    action_used_fallback: bool

    # Written by query_doc_store / web_search
    retrieved_chunks: list[RetrievedChunk]
    web_results: list[RetrievedChunk]
    web_results_filtered_count: int

    # Written by synthesize / disclaimer_check
    recommendation: str
    disclaimer_added: bool
