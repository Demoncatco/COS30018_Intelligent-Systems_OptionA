"""Fake Ollama servers for Coordinator tests. No model needs to be running."""

from __future__ import annotations

import json

import httpx

from shared.config import Settings
from shared.llm import LLMClient


def make_settings(tmp_path, **overrides) -> Settings:
    values = {"log_dir": str(tmp_path), "llm_http_retries": 0, "tool_timeout_s": 1.0} | overrides
    return Settings(_env_file=None, **values)


def reply(tool_calls: list[tuple[str, dict]] | None = None, content: str = "", tokens=(120, 20)) -> dict:
    calls = [{"function": {"name": name, "arguments": args}} for name, args in (tool_calls or [])]
    return {"message": {"role": "assistant", "content": content, "tool_calls": calls or None},
            "prompt_eval_count": tokens[0], "eval_count": tokens[1], "done_reason": "stop"}


class ScriptedOllama:
    """Scripted replies in order; the last repeats."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        item = self.replies[min(len(self.bodies) - 1, len(self.replies) - 1)]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    def client(self, settings: Settings) -> LLMClient:
        return LLMClient(settings, transport=httpx.MockTransport(self))


class PolicyOllama:
    """Fake Qwen that follows the system prompt's policy from the latest tool result."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        return httpx.Response(200, json=reply([self.decide(body["messages"])]))

    @staticmethod
    def decide(messages: list[dict]) -> tuple[str, dict]:
        payloads = [_payload(m) for m in messages if m["role"] in ("tool", "user")]
        payloads = [p for p in payloads if p is not None]
        if not payloads:
            return "run_diagnostic", {"mode": "initial", "reason": "start"}
        last = payloads[-1]
        allowed = last.get("allowed_tools", [])
        have_result = any("result_of" in p for p in payloads)
        if last.get("needs_reupload"):
            flags = ", ".join(last.get("quality_flags", []))
            text = "Please re-upload the study"
            if "missing_plane:axial" in flags:
                text += " including an axial series"
            if "low_slice_count" in flags:
                text += " with complete series (enough slices)"
            return "request_reupload", {"user_action": text + ".", "reason": flags}
        if last.get("suggested_recheck") and "run_diagnostic" in allowed:
            return "run_diagnostic", {"mode": "recheck", "focus_findings": last["suggested_recheck"],
                                      "reason": "; ".join(last.get("concerns", []))}
        if "run_recommendation" in allowed:
            return "run_recommendation", {"reason": "senior review"}
        if not have_result and "run_diagnostic" in allowed:
            return "run_diagnostic", {"mode": "initial", "reason": "retry"}
        if not have_result and "request_reupload" in allowed and "finalize_report" not in allowed:
            return "request_reupload", {"user_action": "Please upload the study again."}
        return "finalize_report", {"summary": "Combined the junior and senior results."}

    def client(self, settings: Settings) -> LLMClient:
        return LLMClient(settings, transport=httpx.MockTransport(self))


def _payload(message: dict) -> dict | None:
    content = message.get("content") or ""
    if message["role"] == "user" and "Result: " in content:
        content = content.split("Result: ", 1)[1]
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None