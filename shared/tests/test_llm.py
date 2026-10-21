# Tests the shared LLM client against a fake Ollama server, so no model needs to be running.

import json
import logging

import httpx
import pytest
from pydantic import BaseModel, Field

from shared.config import Settings
from shared.llm import (
    LLMClient,
    LLMError,
    LLMOutputError,
    ToolArgumentsError,
    ToolCall,
    inline_schema,
    tool_message,
    tool_schema,
)


class Decision(BaseModel):
    action: str
    reason: str


class Inner(BaseModel):
    title: str = Field(description="a field that happens to be called title")
    score: float


class Outer(BaseModel):
    items: list[Inner]
    best: Inner | None = None


class RunDiagnosticArgs(BaseModel):
    study_id: str = Field(min_length=1)
    focus_findings: list[str] = Field(default_factory=list)


def _client(replies: list, seen: list[dict] | None = None, **settings) -> LLMClient:
    # Each reply is a dict (HTTP 200 JSON), an httpx.Response, or an exception to raise.
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content) if request.content else {"_path": request.url.path})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    settings.setdefault("llm_http_retries", 0)
    return LLMClient(Settings(_env_file=None, **settings), transport=httpx.MockTransport(handler))


def _reply(content: str = "", tool_calls: list | None = None, tokens=(10, 5), **extra) -> dict:
    return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "prompt_eval_count": tokens[0], "eval_count": tokens[1], "done_reason": "stop", **extra}


USER = [{"role": "user", "content": "go"}]


# --- Request body ---

def test_each_agent_uses_its_configured_model():
    seen = []
    client = _client([_reply("a"), _reply("b"), _reply("c")], seen,
                     coordinator_model="qwen3:14b", recommendation_model="medgemma:4b")
    client.chat("coordinator", USER)
    client.chat("recommendation", USER)
    client.chat("baseline", USER)
    assert [body["model"] for body in seen] == ["qwen3:14b", "medgemma:4b", "medgemma:4b"]


def test_request_carries_keep_alive_context_seed_and_temperature():
    seen = []
    client = _client([_reply("a")], seen, ollama_keep_alive="1h", llm_num_ctx=16384, llm_seed=7, llm_temperature=0.2)
    client.chat("coordinator", USER)
    body = seen[0]
    assert body["keep_alive"] == "1h" and body["stream"] is False
    assert body["options"] == {"temperature": 0.2, "num_ctx": 16384, "seed": 7}


def test_think_is_only_sent_to_qwen_agents():
    seen = []
    client = _client([_reply("a"), _reply("b"), _reply("c")], seen, coordinator_think=True)
    client.chat("coordinator", USER)
    client.chat("diagnostic", USER)
    client.chat("recommendation", USER)
    assert seen[0]["think"] is True and seen[1]["think"] is False
    assert "think" not in seen[2]  # MedGemma has no thinking mode; Ollama would reject the option


# --- Qwen native tools ---

def test_native_tool_calls_are_parsed_and_tokens_counted():
    call = {"id": "call_1", "function": {"name": "run_diagnostic", "arguments": {"study_id": "s1"}}}
    client = _client([_reply(tool_calls=[call], tokens=(100, 20))])
    tools = [tool_schema("run_diagnostic", "Run it", RunDiagnosticArgs)]
    result = client.chat("coordinator", USER, tools=tools)
    assert result.tool_calls == [ToolCall(name="run_diagnostic", arguments={"study_id": "s1"}, id="call_1")]
    assert client.usage_summary()["coordinator"]["tokens_in"] == 100


def test_string_and_broken_arguments_are_handled():
    calls = [{"function": {"name": "a", "arguments": '{"study_id": "s1"}'}},
             {"function": {"name": "b", "arguments": "{not json"}}]
    result = _client([_reply(tool_calls=calls)]).chat("coordinator", USER)
    assert result.tool_calls[0].arguments == {"study_id": "s1"}
    with pytest.raises(ToolArgumentsError, match="not valid JSON"):
        result.tool_calls[1].parse(RunDiagnosticArgs)


def test_tool_arguments_are_validated_with_feedback_for_the_model():
    assert ToolCall("run_diagnostic", {"study_id": "s1"}).parse(RunDiagnosticArgs).study_id == "s1"
    with pytest.raises(ToolArgumentsError) as error:
        ToolCall("run_diagnostic", {"study_id": ""}).parse(RunDiagnosticArgs)
    assert error.value.tool_name == "run_diagnostic"
    assert "study_id" in str(error.value) and "Call the tool again" in str(error.value)


def test_tool_loop_history_messages():
    call = ToolCall("run_diagnostic", {"study_id": "s1"}, id="call_1")
    result = _client([_reply("checking", tool_calls=[
        {"id": "call_1", "function": {"name": "run_diagnostic", "arguments": {"study_id": "s1"}}}])]
    ).chat("coordinator", USER)
    assert result.assistant_message() == {"role": "assistant", "content": "checking", "tool_calls": [
        {"id": "call_1", "function": {"name": "run_diagnostic", "arguments": {"study_id": "s1"}}}]}

    assert tool_message(call, RunDiagnosticArgs(study_id="s1")) == {
        "role": "tool", "tool_name": "run_diagnostic", "tool_call_id": "call_1",
        "content": '{"study_id":"s1","focus_findings":[]}'}
    assert tool_message("request_reupload", {"ok": True})["content"] == '{"ok": true}'


def test_tool_schema_is_flat_and_keeps_fields_named_title():
    schema = tool_schema("pick", "Pick items", Outer)["function"]["parameters"]
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    assert schema["properties"]["items"]["items"]["properties"]["title"]["type"] == "string"
    assert "title" not in schema  # the schema keyword is dropped
    with pytest.raises(ValueError):
        tool_schema("bad name!", "x", Decision)


def test_inline_schema_handles_optional_nested_models():
    best = inline_schema(Outer)["properties"]["best"]
    assert any(option.get("properties", {}).get("score") for option in best["anyOf"])


# --- Thinking ---

def test_thinking_is_kept_separate_from_content():
    reply = _reply("answer")
    reply["message"]["thinking"] = "hmm"
    separate = _client([reply]).chat("coordinator", USER)
    assert separate.content == "answer" and separate.thinking == "hmm"
    inline = _client([_reply("<think>let me see</think>\n{\"a\": 1}")]).chat("coordinator", USER)
    assert inline.content == '{"a": 1}' and inline.thinking == "let me see"


# --- MedGemma JSON output ---

def test_decide_sends_the_flat_schema_and_retries_once_then_succeeds():
    seen = []
    client = _client([_reply("not json"), _reply('```json\n{"action": "finalize", "reason": "ok"}\n```')], seen)
    decision, calls = client.decide("recommendation", USER, Decision)
    assert decision.action == "finalize" and len(calls) == 2
    assert client.usage["recommendation"].json_retries == 1
    assert seen[0]["format"] == inline_schema(Decision) and seen[0]["options"]["temperature"] == 0.0
    assert "invalid" in seen[1]["messages"][-1]["content"]


def test_decide_tells_the_model_when_its_output_was_cut_off():
    seen = []
    client = _client([_reply('{"action": "fin', done_reason="length"),
                      _reply('{"action": "finalize", "reason": "ok"}')], seen)
    client.decide("recommendation", USER, Decision)
    assert "cut off" in seen[1]["messages"][-1]["content"]
    assert client.usage["recommendation"].truncated == 1


def test_decide_raises_after_all_attempts_fail():
    client = _client([_reply("nope"), _reply("still nope")])
    with pytest.raises(LLMOutputError) as error:
        client.decide("coordinator", USER, Decision)
    assert error.value.last_output == "still nope"
    assert client.usage["coordinator"].json_failures == 1


# --- Errors ---

def test_ollama_error_message_is_surfaced():
    missing = httpx.Response(404, json={"error": "model 'qwen3:8b' not found, try pulling it first"})
    client = _client([missing])
    with pytest.raises(LLMError, match="try pulling it first"):
        client.chat("coordinator", USER)
    assert client.usage["coordinator"].errors == 1


def test_connection_errors_are_retried(monkeypatch):
    monkeypatch.setattr("shared.llm.time.sleep", lambda _: None)
    refused = httpx.ConnectError("refused")
    assert _client([refused, _reply("ok")], llm_http_retries=1).chat("coordinator", USER).content == "ok"
    with pytest.raises(LLMError, match="ollama serve"):
        _client([refused, refused], llm_http_retries=1).chat("coordinator", USER)


def test_timeouts_are_not_retried():
    client = _client([httpx.ReadTimeout("slow"), _reply("never reached")], llm_http_retries=2)
    with pytest.raises(LLMError, match="timed out"):
        client.chat("coordinator", USER)


# --- Usage, logging, callbacks ---

def test_usage_has_per_agent_and_total_figures_and_can_be_reset():
    client = _client([_reply("a", tokens=(10, 2), load_duration=1_500_000_000), _reply("b", tokens=(30, 4))])
    client.chat("coordinator", USER)
    client.chat("recommendation", USER)
    summary = client.usage_summary()
    assert summary["total"]["calls"] == 2 and summary["total"]["tokens_in"] == 40
    assert summary["coordinator"]["load_ms"] == pytest.approx(1500.0)
    client.reset_usage()
    assert client.usage_summary() == {}


def test_every_call_reaches_on_call_and_the_log(caplog):
    seen_results = []
    client = _client([_reply("a", tokens=(12, 3))])
    client.on_call = seen_results.append
    with caplog.at_level(logging.INFO, logger="shared.llm"):
        client.chat("diagnostic", USER, session_id="sess-1")
    assert seen_results[0].session_id == "sess-1" and seen_results[0].tokens == 15
    record = next(r for r in caplog.records if r.message.startswith("llm call"))
    assert record.agent == "diagnostic" and record.session_id == "sess-1"
    assert record.data["tokens_in"] == 12 and "latency_ms" in record.data


# --- Health and preload ---

def test_health_reports_availability_digest_and_tool_support():
    tags = {"models": [{"name": "qwen3:8b", "digest": "abcdef1234567890"},
                       {"name": "medgemma:4b", "digest": "0123456789abcdef"}]}
    show_qwen = {"capabilities": ["completion", "tools", "thinking"]}
    show_med = {"capabilities": ["completion", "vision"]}
    # tags, then /api/show for diagnostic, coordinator, recommendation
    client = _client([tags, show_qwen, show_qwen, show_med])
    health = client.health()
    assert health["ollama"] == "ok" and health["ready"] is True
    assert health["models"]["coordinator"] == {"model": "qwen3:8b", "available": True, "digest": "abcdef123456",
                                               "capabilities": ["completion", "tools", "thinking"],
                                               "tools_supported": True}
    assert "tools_supported" not in health["models"]["recommendation"]


def test_health_flags_a_missing_model_and_an_unreachable_server():
    client = _client([{"models": [{"name": "qwen3:8b"}]}, {}, {}])
    health = client.health()
    assert health["ready"] is False and health["models"]["recommendation"]["available"] is False
    assert _client([httpx.ConnectError("refused")]).health()["ollama"] == "unreachable"


def test_preload_loads_each_distinct_model_once():
    seen = []
    loaded = _client([{"done": True}, {"done": True}], seen).preload()
    assert sorted(loaded) == ["medgemma:4b", "qwen3:8b"]
    assert [body["messages"] for body in seen] == [[], []]


def test_unknown_agent_is_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None).model_for("janitor")