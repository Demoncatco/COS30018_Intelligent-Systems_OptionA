# Retrieve local document chunks for the current finding

from typing import Callable, cast

from ..rag.retrieval import retrieve
from ..state import RecommendationState

DEFAULT_K = 3


def query_doc_store(
    state: RecommendationState,
    retrieve_fn: Callable = retrieve,
) -> RecommendationState:
    """Look up the query and store the retrieved chunks in a copy of the state."""
    query = state.get("user_query") or f"{state['finding']} treatment"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    results = retrieve_fn(query, k=DEFAULT_K, finding=state.get("finding"))

    new_state = cast(RecommendationState, dict(state))
    new_state["retrieved_chunks"] = results
    return new_state
