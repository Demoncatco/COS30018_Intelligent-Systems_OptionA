# Call Qwen for the Recommendation Agent through the shared Ollama client (shared/llm.py)
#
# The graph nodes only need a plain function: llm(prompt, system) -> reply text.
# Everything else (model tag, Ollama host, timeout, temperature, seed, thinking mode) comes from .env
# via get_settings(), and every call is counted under the "recommendation" agent so its tokens and
# latency show up in LLMClient.usage_summary() and the session trace.

from __future__ import annotations

from collections.abc import Callable

from shared.llm import LLMClient, LLMError

AGENT_NAME = "recommendation"

# Kept so older code that catches OllamaError still works: it is the shared client's error.
OllamaError = LLMError

LLMFunction = Callable[[str, "str | None"], str]

_default_client: LLMClient | None = None


def _get_default_client() -> LLMClient:
    # One client per process, created on first use, so importing this module never touches Ollama.
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def make_qwen_caller(client: LLMClient | None = None, session_id: str | None = None) -> LLMFunction:
    """Return llm(prompt, system) bound to one LLMClient and one session, for the graph nodes."""

    def call(prompt: str, system: str | None = None) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        result = (client or _get_default_client()).chat(AGENT_NAME, messages, session_id=session_id)
        # chat() already moves any <think>...</think> text into result.thinking.
        return result.content.strip()

    return call


def call_qwen(prompt: str, system: str | None = None) -> str:
    """Send one prompt to the Recommendation Agent's model (RECOMMENDATION_MODEL in .env).

    Raises LLMError (alias OllamaError) with a readable reason if Ollama is down, the model is
    not pulled, or the call times out."""
    return make_qwen_caller()(prompt, system)
