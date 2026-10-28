# Connect graph prompts to the shared Ollama client used by the application
# Model settings come from shared configuration and calls are attributed to the recommendation agent
# The shared client records token usage and latency, while nodes receive only the reply text

from __future__ import annotations

from collections.abc import Callable

from shared.llm import LLMClient, LLMError

AGENT_NAME = "recommendation"

# OllamaError and LLMError name the same shared exception type
OllamaError = LLMError

LLMFunction = Callable[[str, "str | None"], str]

_default_client: LLMClient | None = None


def _get_default_client() -> LLMClient:
    """Reuse one lazily created client when the caller has not supplied its own"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def make_qwen_caller(client: LLMClient | None = None, session_id: str | None = None) -> LLMFunction:
    """Return a prompt function bound to the supplied client and session identifier"""

    def call(prompt: str, system: str | None = None) -> str:
        """Send system and user messages through the shared client and return trimmed answer text"""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        result = (client or _get_default_client()).chat(AGENT_NAME, messages, session_id=session_id)
        # chat() already moves any <think>...</think> text into result.thinking
        return result.content.strip()

    return call


def call_qwen(prompt: str, system: str | None = None) -> str:
    """Call the configured recommendation model through the default shared client
    LLMError propagates when the service, model or request fails
    """
    return make_qwen_caller()(prompt, system)
