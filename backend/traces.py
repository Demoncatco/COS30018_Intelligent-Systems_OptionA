"""Read, check and export per-session traces (logs/traces/*.jsonl) and reports (logs/reports/*.json).

    python -m backend.traces list                       # recent sessions
    python -m backend.traces show <session_id>          # timeline of one session
    python -m backend.traces check                      # integrity check of every saved session
    python -m backend.traces export <id> [<id> ...] --out evaluation/traces
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import pairwise
from pathlib import Path

from shared.config import get_settings
from shared.logging_setup import TraceWriter
from shared.schema import DISCLAIMER, TERMINAL_STATUSES, AgentEvent


def _dirs(log_dir: str | None) -> tuple[Path, Path]:
    root = Path(log_dir or get_settings().log_dir)
    return root / "traces", root / "reports"


def load(session_id: str, log_dir: str | None = None) -> tuple[list[AgentEvent], dict | None]:
    _, reports = _dirs(log_dir)
    events = TraceWriter(session_id, log_dir or get_settings().log_dir).read()
    path = reports / f"{session_id}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return events, report


def validate(session_id: str, events: list[AgentEvent], saved: dict | None) -> list[str]:
    """Problems that would make a trace unusable as submission evidence. Empty list = OK."""
    problems = []
    if not events:
        return ["trace file is missing or empty"]
    if [e.seq for e in events] != list(range(len(events))):
        problems.append("seq numbers are not 0..n-1 in order (events missing or duplicated)")
    if {e.session_id for e in events} != {session_id}:
        problems.append("trace contains events from another session")
    if any(b.timestamp < a.timestamp for a, b in pairwise(events)):
        problems.append("timestamps go backwards")
    last_status = next((e.data.get("status") for e in reversed(events) if e.kind == "status"), None)
    if last_status not in {s.value for s in TERMINAL_STATUSES}:
        problems.append(f"session never finished (last status: {last_status})")
    if saved is None:
        problems.append("no saved report in logs/reports/")
        return problems
    report = (saved.get("state") or {}).get("final_report")
    if report is None:
        problems.append("saved session has no final_report")
    else:
        if DISCLAIMER not in report.get("disclaimer", ""):
            problems.append("final report is missing the disclaimer")
        if last_status and report.get("status") != last_status:
            problems.append(f"report status {report.get('status')} differs from trace status {last_status}")
    return problems


def summarise(saved: dict) -> dict:
    state = saved.get("state") or {}
    report = state.get("final_report") or {}
    return {"session_id": state.get("session_id"), "produced_by": saved.get("kind"), "driver": saved.get("driver"),
            "status": state.get("status"), "study_id": state.get("study_id"),
            "tokens": report.get("total_tokens"), "latency_ms": report.get("total_latency_ms"),
            "created_at": state.get("created_at")}


def list_sessions(log_dir: str | None = None, limit: int = 20) -> list[dict]:
    _, reports = _dirs(log_dir)
    if not reports.is_dir():
        return []
    rows = []
    for path in reports.glob("*.json"):
        try:
            rows.append(summarise(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda r: -(r["created_at"] or 0))[:limit]


def timeline(events: list[AgentEvent]) -> list[str]:
    if not events:
        return []
    start = events[0].timestamp
    lines = []
    for e in events:
        extra = []
        if e.tokens_in or e.tokens_out:
            extra.append(f"{(e.tokens_in or 0) + (e.tokens_out or 0)} tok")
        if e.latency_ms is not None:
            extra.append(f"{e.latency_ms:.0f} ms")
        suffix = f"  ({', '.join(extra)})" if extra else ""
        lines.append(f"[{e.seq:>3}] +{e.timestamp - start:7.2f}s {e.agent.value:<14} {e.kind:<11} {e.message}{suffix}")
    return lines


def export(session_ids: list[str], out: Path, log_dir: str | None = None) -> list[Path]:
    traces, reports = _dirs(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for sid in session_ids:
        for source in (traces / f"{sid}.jsonl", reports / f"{sid}.json"):
            if not source.exists():
                raise FileNotFoundError(source)
            written.append(Path(shutil.copy2(source, out / source.name)))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", help="defaults to LOG_DIR")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").add_argument("--limit", type=int, default=20)
    sub.add_parser("show").add_argument("session_id")
    sub.add_parser("check").add_argument("session_ids", nargs="*", help="default: every saved session")
    exp = sub.add_parser("export")
    exp.add_argument("session_ids", nargs="+")
    exp.add_argument("--out", default="evaluation/traces")
    args = parser.parse_args(argv)

    if args.command == "list":
        for row in list_sessions(args.log_dir, args.limit):
            print(f"{row['session_id']}  {row['produced_by']:<11} {row['driver'] or '':<8} {row['status']:<12} "
                  f"tokens={row['tokens']} ms={row['latency_ms']}  {row['study_id']}")
        return 0
    if args.command == "show":
        events, saved = load(args.session_id, args.log_dir)
        print("\n".join(timeline(events)) or "(no trace)")
        problems = validate(args.session_id, events, saved)
        print("trace OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    if args.command == "check":
        ids = args.session_ids or [r["session_id"] for r in list_sessions(args.log_dir, limit=100_000)]
        failed = 0
        for sid in ids:
            problems = validate(sid, *load(sid, args.log_dir))
            failed += bool(problems)
            print(f"{sid}: {'OK' if not problems else '; '.join(problems)}")
        print(f"{len(ids) - failed}/{len(ids)} sessions OK")
        return 1 if failed else 0
    for path in export(args.session_ids, Path(args.out), args.log_dir):
        print(f"wrote {path}")
    return 0



if __name__ == "__main__":
    sys.exit(main())