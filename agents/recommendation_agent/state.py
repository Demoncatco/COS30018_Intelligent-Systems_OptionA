"""
Internal working state for the Recommendation Agent's LangGraph graph.

This is NOT shared with the Coordinator or Diagnostic Agent - it only exists to pass information between this agent's own nodes as the graph runs
"""

from typing import TypedDict, Optional


class RecommendationGraphState(TypedDict):
    # Copied in from the CoordinatorRequest when the graph starts
    finding: dict[str, float]
    confidence: float
    study_context: dict
    task_type: str

    # Working memory while the graph runs
    next_action: Optional[str]        # parsed from the LLM: query_doc_store / web_search / synthesize / stop
    retrieved_chunks: list[dict]      # e.g. [{"text": "...", "source_url": "..."}, ...]
    search_attempts: int              # how many times web_search has been tried this run
    retry_count: int                  # how many malformed-output retries have happened
    trace: list[str]                  # human-readable log of every step, for the UI

    # Filled in right before the graph returns
    recommendation_text: Optional[str]
    citations: list[str]
    urgency: Optional[str]
    evidence_status: Optional[str]