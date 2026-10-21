# Call Qwen3 through a local Ollama server

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:1.7b"
REQUEST_TIMEOUT_SECONDS = 60


def call_qwen(prompt: str, system: str | None = None) -> str:
    """Send a prompt to Qwen and return its response text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["response"]
