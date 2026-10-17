"""Shared Ollama client for all agents: native tool calls (Qwen), validated JSON output (MedGemma), token and latency tracking."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from shared.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Ollama could not be reached or returned an error."""


class LLMOutputError(LLMError):
    """The model kept returning output that did not match the requested schema."""

    def __init__(self, message: str, last_output: str) -> None:
        super().__init__(message)
        self.last_output = last_output


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResult:
    agent: str
    model: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass
class UsageTotals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    json_retries: int = 0
    json_failures: int = 0


class LLMClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None,
                 on_call: Callable[[LLMResult], None] | None = None) -> None:
        # transport is for tests (httpx.MockTransport).
        self.settings = settings or get_settings()
        self._http = httpx.Client(base_url=self.settings.ollama_host, timeout=self.settings.llm_timeout_s,
                                  transport=transport)
        self.on_call = on_call
        self.usage: dict[str, UsageTotals] = defaultdict(UsageTotals)

    def chat(self, agent: str, messages: list[dict], tools: list[dict] | None = None,
             format: dict | str | None = None, temperature: float | None = None) -> LLMResult:
        model = self.settings.model_for(agent)
        body: dict = {
            "model": model, "messages": messages, "stream": False, "keep_alive": self.settings.ollama_keep_alive,
            "options": {"temperature": self.settings.llm_temperature if temperature is None else temperature},
        }
        if tools:
            body["tools"] = tools
        if format is not None:
            body["format"] = format

        start = time.perf_counter()
        try:
            response = self._http.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise LLMError(f"Ollama call for {agent} ({model}) failed: {error}") from error
        data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        message = data.get("message", {})
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args))

        result = LLMResult(agent=agent, model=model, content=message.get("content", "") or "", tool_calls=calls,
                           tokens_in=int(data.get("prompt_eval_count", 0) or 0),
                           tokens_out=int(data.get("eval_count", 0) or 0), latency_ms=latency_ms)
        totals = self.usage[agent]
        totals.calls += 1
        totals.tokens_in += result.tokens_in
        totals.tokens_out += result.tokens_out
        totals.latency_ms += latency_ms
        if self.on_call:
            self.on_call(result)
        return result

    def decide(self, agent: str, messages: list[dict], schema: type[T]) -> tuple[T, list[LLMResult]]:
        # JSON output validated against `schema`, retried with the error. Raises LLMOutputError if it never validates.
        json_schema = schema.model_json_schema()
        history = list(messages)
        calls: list[LLMResult] = []
        last = ""
        for attempt in range(self.settings.llm_json_retries + 1):
            result = self.chat(agent, history, format=json_schema, temperature=0.0)
            calls.append(result)
            last = result.content
            try:
                return schema.model_validate_json(_strip_fences(last)), calls
            except ValidationError as error:
                if attempt < self.settings.llm_json_retries:
                    self.usage[agent].json_retries += 1
                    history = history + [
                        {"role": "assistant", "content": last},
                        {"role": "user", "content": f"That JSON was invalid: {error.errors(include_url=False)}. "
                                                    "Reply again with ONLY corrected JSON matching the schema."},
                    ]
        self.usage[agent].json_failures += 1
        raise LLMOutputError(f"{agent} returned invalid JSON {self.settings.llm_json_retries + 1} times", last)

    def health(self) -> dict:
        try:
            tags = self._http.get("/api/tags").json().get("models", [])
        except httpx.HTTPError as error:
            return {"ollama": "unreachable", "error": str(error)}
        available = {m.get("name") for m in tags}
        wanted = {a: self.settings.model_for(a) for a in ("diagnostic", "coordinator", "recommendation")}
        return {"ollama": "ok", "models": {a: {"model": m, "available": m in available} for a, m in wanted.items()}}

    def usage_summary(self) -> dict:
        return {agent: vars(totals) for agent, totals in self.usage.items()}

    def close(self) -> None:
        self._http.close()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def tool_schema(name: str, description: str, parameters: type[BaseModel]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": parameters.model_json_schema()}}