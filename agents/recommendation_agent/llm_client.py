# Call Qwen3 through a local Ollama server

import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:1.7b"
REQUEST_TIMEOUT_SECONDS = 120  # CPU-only inference on a small GPU can be slow

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaError(RuntimeError):
    """Raised when Ollama can't be reached or returns an error. The message
    includes Ollama's own explanation, which a bare 'HTTP 500' hides."""


def call_qwen(prompt: str, system: str | None = None) -> str:
    """Sends one prompt to the local Qwen3-1.7B instance and returns the
    reply text. Does NOT parse or validate the content - that's each node's
    job - but it does turn transport failures into OllamaError with a
    readable reason."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False,  # Qwen3 otherwise writes out its reasoning first
    }
    if system:
        payload["system"] = system

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Is it running? (check the tray icon, or run `ollama serve`)"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            f"Ollama did not reply within {REQUEST_TIMEOUT_SECONDS}s "
            "(model may still be loading, or inference is very slow)."
        ) from exc

    if not response.ok:
        # Ollama puts the real reason in the body, e.g. {"error": "..."}
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        raise OllamaError(f"Ollama returned HTTP {response.status_code} for model '{MODEL_NAME}': {detail}")

    text = response.json().get("response", "")
    # Some Ollama versions still put <think>...</think> in the reply; strip it.
    return THINK_BLOCK_PATTERN.sub("", text).strip()