"""Logging: JSON lines, rotation, secret redaction, request logs and per-session trace files."""

from __future__ import annotations

import json
import logging

import pytest

from backend import traces
from backend.tests.fakes import SID, WAIT, app_client, make_services
from shared import logging_setup
from shared.config import Settings
from shared.schema import DISCLAIMER

SECRET = "tvly-super-secret-key-42"


@pytest.fixture
def logs(tmp_path, monkeypatch):
    # Real setup_logging into tmp_path; restores the root logger afterwards.
    settings = Settings(_env_file=None, log_dir=str(tmp_path), tavily_api_key=SECRET, log_max_bytes=20_000,
                        log_backup_count=2)
    monkeypatch.setattr(logging_setup, "get_settings", lambda: settings)
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    logging_setup.setup_logging()
    yield tmp_path
    for handler in root.handlers:
        handler.close()
    root.handlers, root.level = saved[0], saved[1]


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_logs_are_json_lines_with_session_context(logs):
    logging.getLogger("t").info("hello", extra={"session_id": "abc", "data": {"k": 1}})
    entry = _lines(logs / "app.jsonl")[-1]
    assert entry["message"] == "hello" and entry["session_id"] == "abc" and entry["data"] == {"k": 1}


def test_secrets_never_reach_the_log_file(logs):
    log = logging.getLogger("t")
    log.info("key is %s", SECRET, extra={"data": {"headers": {"Authorization": f"Bearer {SECRET}"}}})
    try:
        raise RuntimeError(f"auth failed for {SECRET}")
    except RuntimeError:
        log.exception("search failed")
    text = (logs / "app.jsonl").read_text()
    assert SECRET not in text and logging_setup.REDACTED in text


def test_log_file_rotates(logs):
    for i in range(400):
        logging.getLogger("t").info("line %d %s", i, "x" * 100)
    assert (logs / "app.jsonl.1").exists() and not (logs / "app.jsonl.3").exists()


def test_backend_writes_an_access_log_line_per_request(logs):
    with app_client(make_services(logs)) as client:
        rid = client.get("/health?deep=false").headers["X-Request-ID"]
    access = [e for e in _lines(logs / "app.jsonl") if e["message"].startswith("GET /health")]
    assert access and access[-1]["data"]["request_id"] == rid and access[-1]["data"]["status"] == 200


# --- Per-session traces ---

def _session(tmp_path, path="/sessions", body=None):
    with app_client(make_services(tmp_path)) as client:
        return client.post(path, params=WAIT, json=body or {"study_id": SID, "driver": "fixed"}).json()["session_id"]


def test_each_session_gets_a_complete_valid_trace_and_saved_report(tmp_path):
    sid = _session(tmp_path)
    events, saved = traces.load(sid, str(tmp_path))
    assert traces.validate(sid, events, saved) == []
    assert events[0].message == "Session queued" and events[-1].kind == "status"
    assert DISCLAIMER in saved["state"]["final_report"]["disclaimer"]


def test_baseline_sessions_get_traces_too(tmp_path):
    sid = _session(tmp_path, "/baseline", {"study_id": SID})
    events, saved = traces.load(sid, str(tmp_path))
    assert traces.validate(sid, events, saved) == []
    assert {e.agent.value for e in events} >= {"baseline", "model_service"}


def test_validate_catches_a_damaged_trace(tmp_path):
    sid = _session(tmp_path)
    path = tmp_path / "traces" / f"{sid}.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + lines[3:-1]) + "\n")  # drop one event and the final status
    problems = traces.validate(sid, *traces.load(sid, str(tmp_path)))
    assert any("seq" in p for p in problems) and any("never finished" in p for p in problems)


def test_list_show_and_export(tmp_path, capsys):
    sid = _session(tmp_path)
    assert traces.list_sessions(str(tmp_path))[0]["session_id"] == sid
    assert traces.main(["--log-dir", str(tmp_path), "show", sid]) == 0
    assert "trace OK" in capsys.readouterr().out
    assert traces.main(["--log-dir", str(tmp_path), "check"]) == 0
    out = tmp_path / "export"
    written = traces.export([sid], out, str(tmp_path))
    assert sorted(p.name for p in written) == sorted([f"{sid}.jsonl", f"{sid}.json"])