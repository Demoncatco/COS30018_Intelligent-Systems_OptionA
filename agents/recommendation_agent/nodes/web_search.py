# Search Tavily and keep results from trusted domains

import inspect
import logging
from collections.abc import Callable

from shared.config import get_settings

from ..state import RecommendationState, RetrievedChunk
from ..trusted_domains import TRUSTED_DOMAINS, is_trusted_domain

logger = logging.getLogger("recommendation_agent.web_search")

DEFAULT_MAX_RESULTS = 5
MAX_WEB_CONTENT_CHARS = 1000  # Limit snippet length for the model.
WEB_SEARCH_TIMEOUT_S = 20  # one search must not use up the whole TOOL_TIMEOUT_S budget


def _tavily_search(query: str, max_results: int) -> list[dict]:
    """Load Tavily only when a live search is needed."""
    from tavily import TavilyClient

    # The key comes from .env through shared/config.py; the .env.example placeholder counts as unset.
    settings = get_settings()
    if not settings.has_tavily_key:
        raise RuntimeError("TAVILY_API_KEY is not set in .env, so web search is unavailable")
    client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
    # Limit the search to trusted sites, then check each result below.
    kwargs = {"query": query, "max_results": max_results, "include_domains": sorted(TRUSTED_DOMAINS)}
    if "timeout" in inspect.signature(client.search).parameters:  # older tavily-python has no timeout
        kwargs["timeout"] = max(1, int(min(WEB_SEARCH_TIMEOUT_S, settings.tool_timeout_s)))
    response = client.search(**kwargs)
    return response.get("results", [])


def _filter_results(raw_results: list[dict]) -> list[dict]:
    """Keep results from trusted domains."""
    return [result for result in raw_results if is_trusted_domain(result.get("url", ""))]


def _to_chunk(result: dict, finding: str) -> RetrievedChunk | None:
    """Convert a search result to a chunk, or return None if it has no text."""
    text = (result.get("content") or result.get("snippet") or "").strip()
    if not text:
        return None
    return {
        "source_title": result.get("title") or result["url"],
        "source_url": result["url"],
        "findings": [finding],
        "text": text[:MAX_WEB_CONTENT_CHARS],
        "score": float(result.get("score") or 0.0),
    }


def web_search(
    state: RecommendationState,
    search_fn: Callable[[str, int], list[dict]] = _tavily_search,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> RecommendationState:
    """Search for the finding and store trusted results as chunks."""
    query = state.get("user_query") or f"{state['finding']} treatment"
    raw_results = search_fn(query, max_results)
    trusted_results = _filter_results(raw_results)
    dropped = [r.get("url", "?") for r in raw_results if r not in trusted_results]
    if dropped:
        logger.info("web_search dropped untrusted results: %s", dropped)
    chunks = [c for c in (_to_chunk(r, state["finding"]) for r in trusted_results) if c]

    new_state = state.copy()
    new_state["web_results"] = chunks
    new_state["web_results_filtered_count"] = len(raw_results) - len(trusted_results)
    return new_state