# Tests the shared LLM client against a fake Ollama server, so no model needs to be running.

import json

import httpx
import pytest
from pydantic import BaseModel

from shared.config import Settings
from shared.llm import LLMClient, LLMOutputError, tool_schema


class Decision(BaseModel):
    action: str
    reason: str


def _client(replies: list[dict], seen: list[dict] | None = None, **settings) -> LLMClient:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    return LLMClient(Settings(_env_file=None, **settings), transport=httpx.MockTransport(handler))


def _reply(content: str = "", tool_calls: list | None = None, tokens=(10, 5)) -> dict:
    return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "prompt_eval_count": tokens[0], "eval_count": tokens[1]}


def test_each_agent_uses_its_configured_model():
    seen = []
    client = _client([_reply("a"), _reply("b")], seen, coordinator_model="qwen-x", recommendation_model="medgemma-y")
    client.chat("coordinator", [{"role": "user", "content": "hi"}])
    client.chat("recommendation", [{"role": "user", "content": "hi"}])
    assert [body["model"] for body in seen] == ["qwen-x", "medgemma-y"]


def test_native_tool_calls_are_parsed_and_tokens_counted():
    call = {"function": {"name": "run_diagnostic", "arguments": {"study_id": "s1"}}}
    client = _client([_reply(tool_calls=[call], tokens=(100, 20))])
    tools = [tool_schema("run_diagnostic", "Run it", Decision)]
    result = client.chat("coordinator", [{"role": "user", "content": "go"}], tools=tools)
    assert result.tool_calls[0].name == "run_diagnostic" and result.tool_calls[0].arguments == {"study_id": "s1"}
    assert client.usage_summary()["coordinator"]["tokens_in"] == 100


def test_decide_retries_once_then_succeeds():
    client = _client([_reply("not json"), _reply('```json\n{"action": "finalize", "reason": "ok"}\n```')])
    decision, calls = client.decide("recommendation", [{"role": "user", "content": "?"}], Decision)
    assert decision.action == "finalize" and len(calls) == 2
    assert client.usage["recommendation"].json_retries == 1


def test_decide_raises_after_all_attempts_fail():
    client = _client([_reply("nope"), _reply("still nope")])
    with pytest.raises(LLMOutputError):
        client.decide("coordinator", [{"role": "user", "content": "?"}], Decision)
    assert client.usage["coordinator"].json_failures == 1


def test_unknown_agent_is_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None).model_for("janitor")