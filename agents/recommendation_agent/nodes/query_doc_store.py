# Retrieve local document chunks for the current finding

from typing import Callable

from ..rag.retrieval import retrieve
from ..state import RecommendationState, RetrievedChunk

DEFAULT_K = 3
CANDIDATE_POOL = 12  # Retrieve extra chunks to allow a mix of sources.
MAX_CHUNKS_PER_SOURCE = 1


def _diversify(results: list[RetrievedChunk], k: int, max_per_source: int) -> list[RetrievedChunk]:
    """Pick varied sources first, then fill any remaining slots with the best chunks."""
    chosen: list[RetrievedChunk] = []
    per_source: dict[str, int] = {}
    for chunk in results:
        url = chunk.get("source_url", "")
        if per_source.get(url, 0) < max_per_source:
            chosen.append(chunk)
            per_source[url] = per_source.get(url, 0) + 1
        if len(chosen) == k:
            return chosen
    for chunk in results:  # backfill with the best chunks not already chosen
        if len(chosen) == k:
            break
        if chunk not in chosen:
            chosen.append(chunk)
    return chosen


def query_doc_store(
    state: RecommendationState,
    retrieve_fn: Callable = retrieve,
) -> RecommendationState:
    """Retrieve chunks for the query and keep a mix of sources."""
    query = state.get("user_query") or f"{state['finding']} treatment"
    candidates = retrieve_fn(query, k=CANDIDATE_POOL, finding=state.get("finding"))
    results = _diversify(candidates, DEFAULT_K, MAX_CHUNKS_PER_SOURCE)

    new_state = state.copy()
    new_state["retrieved_chunks"] = results
    return new_state
