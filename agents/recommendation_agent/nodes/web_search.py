# Search Tavily and keep results from trusted domains

import os
from typing import Callable, cast

from ..state import RecommendationState, RetrievedChunk
from ..trusted_domains import is_trusted_domain

DEFAULT_MAX_RESULTS = 5


def _tavily_search(query: str, max_results: int) -> list[dict]:
    """Load Tavily only when a live search is needed."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=max_results)
    return response.get("results", [])


def _filter_results(raw_results: list[dict]) -> list[dict]:
    """Keep results whose URLs belong to trusted domains."""
    return [result for result in raw_results if is_trusted_domain(result.get("url", ""))]


def web_search(
    state: RecommendationState,
    search_fn: Callable[[str, int], list[dict]] = _tavily_search,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> RecommendationState:
    """Search for the finding and store the trusted results in a copy of the state."""
    query = state.get("user_query") or f"{state['finding']} treatment"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    raw_results = search_fn(query, max_results)
    trusted_results = _filter_results(raw_results)

    new_state = cast(RecommendationState, dict(state))
    new_state["web_results"] = cast(list[RetrievedChunk], trusted_results)
    new_state["web_results_filtered_count"] = len(raw_results) - len(trusted_results)
    return new_state
