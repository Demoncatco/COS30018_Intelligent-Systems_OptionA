"""Shared Ollama client for every agent and the baseline.

- Qwen agents (Coordinator, Diagnostic) use native tool calls:  chat(..., tools=[tool_schema(...)])
- MedGemma (Recommendation, baseline) uses JSON-schema output:   decide(..., schema=SomeModel)
- Every call is counted per agent (tokens, latency, retries) and logged, for the report and the UI trace.

A Qwen tool loop looks like this:

    result = client.chat("coordinator", history, tools=TOOLS, session_id=sid)
    history.append(result.assistant_message())
    for call in result.tool_calls:
        args = call.parse(RunDiagnosticArgs)          # raises ToolArgumentsError with feedback for the model
        output = run_diagnostic(**args.model_dump())
        history.append(tool_message(call, output))
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from shared.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Some Ollama/model versions put the reasoning inline instead of in message.thinking.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Ollama restarting or a proxy in between.
_RETRYABLE_STATUS = {502, 503, 504}


class LLMError(RuntimeError):
    """Ollama could not be reached or returned an error."""


class LLMOutputError(LLMError):
    """The model kept returning output that did not match the requested schema."""

    def __init__(self, message: str, last_output: str) -> None:
        super().__init__(message)
        self.last_output = last_output


class ToolArgumentsError(ValueError):
    """A tool call's arguments did not match the tool's schema. str(error) is written to be sent back to the model."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None  # set by newer Ollama versions; echoed back in the tool result

    def parse(self, schema: type[T]) -> T:
        # Validates the arguments against the tool's Pydantic model.
        if "_unparsed" in self.arguments:
            raise ToolArgumentsError(self.name, f"Arguments for {self.name} were not valid JSON: "
                                                f"{self.arguments['_unparsed']!r}. Call the tool again with a JSON object.")
        try:
            return schema.model_validate(self.arguments)
        except ValidationError as error:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
                                 for e in error.errors(include_url=False))
            raise ToolArgumentsError(self.name, f"Invalid arguments for {self.name}: {problems}. "
                                                "Call the tool again with corrected arguments.") from None


@dataclass
class LLMResult:
    agent: str
    model: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0  # wall clock for the whole HTTP call
    load_ms: float = 0.0  # part of latency spent loading the model (cold start); ~0 when the model is warm
    truncated: bool = False  # output hit the token limit, so JSON may be cut off
    session_id: str | None = None

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def assistant_message(self) -> dict:
        # The message to append to the history so the model sees its own turn, including tool calls.
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {**({"id": c.id} if c.id else {}), "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


def tool_message(call: ToolCall | str, content: Any) -> dict:
    # A tool result for the history. content may be a Pydantic model, a dict/list or text.
    if isinstance(content, BaseModel):
        text = content.model_dump_json()
    elif isinstance(content, (dict, list)):
        text = json.dumps(content, default=str)
    else:
        text = str(content)
    name = call.name if isinstance(call, ToolCall) else call
    message = {"role": "tool", "tool_name": name, "content": text}
    if isinstance(call, ToolCall) and call.id:
        message["tool_call_id"] = call.id
    return message


@dataclass
class UsageTotals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    load_ms: float = 0.0
    errors: int = 0
    json_retries: int = 0
    json_failures: int = 0
    truncated: int = 0


class LLMClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None,
                 on_call: Callable[[LLMResult], None] | None = None) -> None:
        # transport is for tests (httpx.MockTransport). on_call receives every LLMResult, e.g. to add an AgentEvent.
        self.settings = settings or get_settings()
        self._http = httpx.Client(base_url=self.settings.ollama_host, timeout=self.settings.llm_timeout_s,
                                  transport=transport)
        self.on_call = on_call
        self.usage: dict[str, UsageTotals] = defaultdict(UsageTotals)
        self._lock = threading.Lock()  # the backend may run several sessions at once on one client

    # --- Calls ---

    def chat(self, agent: str, messages: list[dict], tools: list[dict] | None = None,
             format: dict | str | None = None, temperature: float | None = None,
             session_id: str | None = None) -> LLMResult:
        model = self.settings.model_for(agent)
        options: dict = {"temperature": self.settings.llm_temperature if temperature is None else temperature,
                         "num_ctx": self.settings.llm_num_ctx}
        if self.settings.llm_seed is not None:
            options["seed"] = self.settings.llm_seed
        body: dict = {"model": model, "messages": messages, "stream": False,
                      "keep_alive": self.settings.ollama_keep_alive, "options": options}
        think = self.settings.think_for(agent)
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        if format is not None:
            body["format"] = format

        start = time.perf_counter()
        try:
            data = self._post("/api/chat", body, agent, model)
        except LLMError:
            with self._lock:
                self.usage[agent].errors += 1
            logger.warning("llm call failed", extra={"agent": agent, "session_id": session_id,
                                                     "data": {"model": model}})
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        message = data.get("message") or {}
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
        inline = _THINK_BLOCK.findall(content)
        if inline:
            thinking = "\n".join([thinking, *inline]).strip()
            content = _THINK_BLOCK.sub("", content).strip()

        result = LLMResult(
            agent=agent, model=model, content=content, tool_calls=_parse_tool_calls(message), thinking=thinking,
            tokens_in=int(data.get("prompt_eval_count") or 0), tokens_out=int(data.get("eval_count") or 0),
            latency_ms=latency_ms, load_ms=int(data.get("load_duration") or 0) / 1e6,
            truncated=data.get("done_reason") == "length", session_id=session_id,
        )
        self._record(result)
        if self.on_call:
            self.on_call(result)
        return result

    def decide(self, agent: str, messages: list[dict], schema: type[T],
               session_id: str | None = None) -> tuple[T, list[LLMResult]]:
        # JSON output constrained to `schema`, validated, and retried with the error message if it does not fit.
        # Returns the parsed object and every call made. Raises LLMOutputError if no attempt validates.
        json_schema = inline_schema(schema)
        history = list(messages)
        calls: list[LLMResult] = []
        last = ""
        attempts = self.settings.llm_json_retries + 1
        for attempt in range(attempts):
            result = self.chat(agent, history, format=json_schema, temperature=0.0, session_id=session_id)
            calls.append(result)
            last = result.content
            try:
                return schema.model_validate_json(_strip_fences(last)), calls
            except ValidationError as error:
                if attempt == attempts - 1:
                    break
                with self._lock:
                    self.usage[agent].json_retries += 1
                hint = ("Your reply was cut off. Reply with shorter, complete JSON."
                        if result.truncated else
                        f"That JSON was invalid: {_short_errors(error)}.")
                history = [*history, {"role": "assistant", "content": last},
                           {"role": "user", "content": f"{hint} Reply again with ONLY JSON matching the schema."}]
        with self._lock:
            self.usage[agent].json_failures += 1
        logger.warning("llm json failed", extra={"agent": agent, "session_id": session_id,
                                                 "data": {"attempts": attempts, "last_output": last[:500]}})
        raise LLMOutputError(f"{agent} returned invalid JSON {attempts} times", last)

    # --- Model management ---

    def health(self) -> dict:
        # Is Ollama up, is each configured model pulled, and can the Qwen models call tools?
        try:
            response = self._http.get("/api/tags")
            response.raise_for_status()
            tags = response.json().get("models", [])
        except (httpx.HTTPError, ValueError) as error:
            return {"ollama": "unreachable", "host": self.settings.ollama_host, "error": str(error)}
        available = {_normalise(m.get("name") or m.get("model") or ""): m for m in tags}

        models = {}
        for agent in ("diagnostic", "coordinator", "recommendation"):
            name = self.settings.model_for(agent)
            entry = available.get(_normalise(name))
            info: dict = {"model": name, "available": entry is not None}
            if entry is not None:
                info["digest"] = (entry.get("digest") or "")[:12]  # record this in the report
                capabilities = self._capabilities(name)
                if capabilities is not None:
                    info["capabilities"] = capabilities
                    if agent in ("diagnostic", "coordinator"):
                        info["tools_supported"] = "tools" in capabilities
            models[agent] = info
        ready = all(m["available"] and m.get("tools_supported", True) for m in models.values())
        return {"ollama": "ok", "host": self.settings.ollama_host, "ready": ready, "models": models}

    def preload(self, agents: tuple[str, ...] = ("coordinator", "diagnostic", "recommendation")) -> dict[str, float]:
        # Loads each model into memory (an empty chat) so the first real call is not a cold start.
        # Returns load time per model in ms. Models shared by several agents are loaded once.
        loaded: dict[str, float] = {}
        for agent in agents:
            model = self.settings.model_for(agent)
            if model in loaded:
                continue
            start = time.perf_counter()
            self._post("/api/chat", {"model": model, "messages": [], "keep_alive": self.settings.ollama_keep_alive},
                       agent, model)
            loaded[model] = (time.perf_counter() - start) * 1000
        return loaded

    # --- Usage ---

    def usage_summary(self) -> dict:
        with self._lock:
            per_agent = {agent: asdict(totals) for agent, totals in self.usage.items()}
        total = UsageTotals()
        for totals in per_agent.values():
            for key, value in totals.items():
                setattr(total, key, getattr(total, key) + value)
        return {**per_agent, "total": asdict(total)} if per_agent else {}

    def reset_usage(self) -> None:
        with self._lock:
            self.usage.clear()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- Internals ---

    def _post(self, path: str, body: dict, agent: str, model: str) -> dict:
        last_error = ""
        for attempt in range(self.settings.llm_http_retries + 1):
            try:
                response = self._http.post(path, json=body)
            except (httpx.ConnectError, httpx.RemoteProtocolError) as error:
                last_error = (f"cannot reach Ollama at {self.settings.ollama_host} ({error}); "
                              "is `ollama serve` running?")
            except httpx.TimeoutException:
                # Not retried: a second attempt would just wait another full timeout.
                raise LLMError(f"Ollama call for {agent} ({model}) timed out after {self.settings.llm_timeout_s}s; "
                               "raise LLM_TIMEOUT_S or use a smaller model") from None
            except httpx.HTTPError as error:
                raise LLMError(f"Ollama call for {agent} ({model}) failed: {error}") from error
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as error:
                        raise LLMError(f"Ollama returned non-JSON for {agent} ({model})") from error
                # Ollama explains the problem in the body, e.g. "model 'x' not found, try pulling it first".
                last_error = f"HTTP {response.status_code}: {_ollama_error(response)}"
                if response.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < self.settings.llm_http_retries:
                time.sleep(min(2.0 ** attempt, 5.0))
        raise LLMError(f"Ollama call for {agent} ({model}) failed: {last_error}")

    def _capabilities(self, model: str) -> list[str] | None:
        try:
            response = self._http.post("/api/show", json={"model": model})
            response.raise_for_status()
            capabilities = response.json().get("capabilities")
        except (httpx.HTTPError, ValueError):
            return None
        return list(capabilities) if isinstance(capabilities, list) else None

    def _record(self, result: LLMResult) -> None:
        with self._lock:
            totals = self.usage[result.agent]
            totals.calls += 1
            totals.tokens_in += result.tokens_in
            totals.tokens_out += result.tokens_out
            totals.latency_ms += result.latency_ms
            totals.load_ms += result.load_ms
            totals.truncated += int(result.truncated)
        logger.info(
            "llm call %s %s: %d in / %d out tokens, %.0f ms", result.agent, result.model, result.tokens_in,
            result.tokens_out, result.latency_ms,
            extra={"agent": result.agent, "session_id": result.session_id, "data": {
                "model": result.model, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "latency_ms": round(result.latency_ms, 1), "load_ms": round(result.load_ms, 1),
                "tool_calls": [c.name for c in result.tool_calls], "truncated": result.truncated}},
        )
        if result.truncated:
            logger.warning("llm output truncated (done_reason=length); consider raising LLM_NUM_CTX",
                           extra={"agent": result.agent, "session_id": result.session_id})


# --- Schemas ---

def inline_schema(model: type[BaseModel]) -> dict:
    # Pydantic's JSON schema with $refs expanded and titles removed: small models follow a flat schema
    # far more reliably, and the prompt gets shorter.
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                if name in seen:  # recursive model: leave the reference rather than loop forever
                    return {"type": "object"}
                merged = {**copy.deepcopy(definitions[name]), **{k: v for k, v in node.items() if k != "$ref"}}
                return expand(merged, (*seen, name))
            out = {}
            for key, value in node.items():
                if key == "title":
                    continue
                if key == "properties" and isinstance(value, dict):
                    # Keys here are field names (a field may itself be called "title"), not schema keywords.
                    out[key] = {prop: expand(sub, seen) for prop, sub in value.items()}
                else:
                    out[key] = expand(value, seen)
            return out
        if isinstance(node, list):
            return [expand(item, seen) for item in node]
        return node

    return expand(schema, ())


def tool_schema(name: str, description: str, parameters: type[BaseModel]) -> dict:
    # An Ollama/OpenAI-style tool definition built from a Pydantic model of the tool's arguments.
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise ValueError(f"tool name {name!r} must be 1-64 letters, digits, '_' or '-'")
    params = inline_schema(parameters)
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


# --- Helpers ---

def _parse_tool_calls(message: dict) -> list[ToolCall]:
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_unparsed": args}
        if not isinstance(args, dict):
            args = {} if args is None else {"_unparsed": args}
        calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=call.get("id")))
    return calls


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _short_errors(error: ValidationError, limit: int = 5) -> str:
    items = [f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in error.errors(include_url=False)]
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return "; ".join(items[:limit]) + more


def _ollama_error(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error") or response.text)[:300]
    except ValueError:
        return response.text[:300]


def _normalise(model: str) -> str:
    # Ollama treats "name" and "name:latest" as the same model.
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


# --- Smoke test against a real Ollama: python -m shared.llm ---

def _smoke_test() -> int:
    class _Echo(BaseModel):
        study_id: str

    class _Advice(BaseModel):
        urgency: str
        summary: str

    with LLMClient() as client:
        health = client.health()
        print(json.dumps(health, indent=2))
        if health.get("ollama") != "ok" or not health.get("ready"):
            print("\nNot ready: start Ollama and `ollama pull` the models above, then run this again.")
            return 1
        print("\npreload (ms):", {m: round(ms) for m, ms in client.preload().items()})

        result = client.chat("coordinator", [{"role": "user", "content": "Run the diagnostic for study s-001."}],
                             tools=[tool_schema("run_diagnostic", "Score one MRI study.", _Echo)])
        calls = [(c.name, c.arguments) for c in result.tool_calls]
        print(f"\ncoordinator tool calls: {calls or 'none'} ({result.latency_ms:.0f} ms)")

        advice, _ = client.decide("recommendation", [{"role": "user", "content":
                                  "A knee MRI may indicate an ACL tear. Give urgency and a one-line summary."}],
                                  _Advice)
        print(f"recommendation JSON: {advice.model_dump()}")
        print("\nusage:", json.dumps(client.usage_summary()["total"], indent=2))
        return 0 if calls else 1


if __name__ == "__main__":
    raise SystemExit(_smoke_test())