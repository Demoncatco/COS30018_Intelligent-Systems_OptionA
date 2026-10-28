# Run selected tasks through the multi-agent system and single-agent baseline
# Serve controlled scores through the real Model Service in replay mode and apply requested faults
# Save each completed run immediately so an interrupted evaluation can resume from its records

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agents.coordinator.agent import CoordinatorAgent, Driver
from agents.coordinator.tools import RUN_DIAGNOSTIC, RUN_RECOMMENDATION
from agents.coordinator.wiring import load_diagnostic
from backend.baseline import BaselineAgent
from backend.model_service_client import ModelServiceClient
from evaluation import eval_metrics
from evaluation.eval_checks import Check, quality_score, run_checks
from evaluation.eval_faults import (
    UNREACHABLE_URL,
    FailFirst,
    failing_retrieve,
    failing_search,
    no_web_search,
)
from evaluation.eval_tasks import (
    TASK_FILES,
    Task,
    build_replay_file,
    load_tasks,
)
from shared.config import Settings, get_settings
from shared.llm import LLMClient
from shared.schema import AgentName, FinalReport, SessionState

logger = logging.getLogger("evaluation")

RESULTS_DIR = Path("evaluation/results/system")
SYSTEMS: tuple[str, ...] = ("multi_agent", "baseline")
DEFAULT_TOOL_TIMEOUT_S = 180.0  # Allow more time for multi-call model workflows than the application default
RAG_INDEX = Path("agents/recommendation_agent/rag/data/index/embeddings.npy")

# Lines the Recommendation Agent puts in its trace (see agents/recommendation_agent/graph.py and runner.py)
RAG_CALL = re.compile(r"\] (query_doc_store|web_search)\s+->")
RAG_FAILURE = re.compile(r"ERROR (query_doc_store|web_search) failed")


@dataclass
class EvalOptions:
    """Store task selection, system choices, repeats and service options for a reproducible run"""
    splits: list[str] = field(default_factory=lambda: ["development", "unseen"])
    task_ids: list[str] = field(default_factory=list)  # empty = all tasks in the splits
    systems: list[str] = field(default_factory=lambda: list(SYSTEMS))
    driver: Driver = "tools"  # Coordinator driver: tools (Qwen), json, fixed (no LLM)
    repeats: int = 1
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    no_web: bool = False  # True: web search finds nothing (reproducible runs; Tavily results change over time)


@dataclass
class EvalHooks:
    """Supply fake services for harness tests and record their use in the saved configuration
    Normal command-line evaluation does not supply these replacements
    """

    recommendation_llm_fn: Callable[..., Any] | None = None
    retrieve_fn: Callable[..., Any] | None = None
    search_fn: Callable[..., Any] | None = None
    baseline_llm: LLMClient | None = None


def make_settings(base: Settings, **updates) -> Settings:
    """Construct a validated settings copy with run-specific overrides"""
    return type(base)(**(base.model_dump() | updates))


# --- The Model Service, running inside this process ---

class ModelServiceServer:
    """Run the real replay-mode Model Service on a local background server"""

    def __init__(self, settings: Settings) -> None:
        """Choose a local port and prepare the service thread without starting it"""
        import uvicorn

        from model_service.api.main import create_app

        with socket.socket() as sock:  # ask the OS for a free port
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(create_app(settings), host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="eval-model-service")

    def start(self, timeout_s: float = 30.0) -> None:
        """Wait for server startup and confirm its health endpoint reports replay mode"""
        self.thread.start()
        deadline = time.time() + timeout_s
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("the Model Service did not start")
            time.sleep(0.05)
        health = httpx.get(f"{self.url}/health", timeout=10).json()
        if health.get("status") != "ok" or health.get("mode") != "replay":
            raise RuntimeError(f"Model Service not ready for the evaluation: {health}")

    def stop(self) -> None:
        """Request server shutdown and wait briefly for the thread to finish"""
        self.server.should_exit = True
        self.thread.join(timeout=10)


# --- Running one task ---

@dataclass
class RunContext:
    """Pass shared clients, settings, options and optional test hooks to each task run"""
    settings: Settings
    llm: LLMClient
    options: EvalOptions
    hooks: EvalHooks | None


def _down_llm(settings: Settings) -> LLMClient:
    """Point an LLM client at the designated unavailable endpoint with HTTP retries disabled"""
    return LLMClient(make_settings(settings, ollama_host=UNREACHABLE_URL, llm_http_retries=0))


def _merge_usage(clients: list[LLMClient]) -> dict[str, dict]:
    """Combine per-agent usage from all clients without adding their total rows twice"""
    merged: dict[str, dict] = {}
    for client in clients:
        for agent, totals in client.usage_summary().items():
            if agent == "total":
                continue
            target = merged.setdefault(agent, {})
            for key, value in totals.items():
                target[key] = target.get(key, 0) + value
    return merged


def run_multi_agent(task: Task, ctx: RunContext, session_id: str) -> tuple[SessionState, FinalReport, dict]:
    """Build the backend-selected agents and apply task faults before running the Coordinator
    Collect executed tools, retrieval traces and usage from the normal application workflow
    """
    from agents.recommendation_agent.runner import RecommendationAgent

    faults, settings, hooks = task.faults, ctx.settings, ctx.hooks or EvalHooks()
    extra_clients: list[LLMClient] = []

    # Use the same Diagnostic implementation selection as the backend
    diag_settings = make_settings(settings, model_service_url=UNREACHABLE_URL) if faults.model_service_down else settings
    diagnostic, _, _ = load_diagnostic(diag_settings, ctx.llm)
    if faults.diagnostic_fail_first:
        diagnostic = FailFirst(diagnostic, faults.diagnostic_fail_first, faults.diagnostic_error, "Diagnostic Agent")

    # Recommendation Agent, with the task's faults in its tools
    rec_llm, rec_llm_fn = ctx.llm, hooks.recommendation_llm_fn
    if faults.recommendation_llm_down:
        rec_llm, rec_llm_fn = _down_llm(settings), None
        extra_clients.append(rec_llm)
    search_fn = hooks.search_fn
    if ctx.options.no_web:
        search_fn = no_web_search
    # An explicit search fault takes priority over the no-web empty-result option
    if faults.web_down:
        search_fn = failing_search
    retrieve_fn = failing_retrieve if faults.rag_down else hooks.retrieve_fn
    recommendation = RecommendationAgent(llm=rec_llm, settings=settings, retrieve_fn=retrieve_fn,
                                         search_fn=search_fn, llm_fn=rec_llm_fn)
    if faults.recommendation_fail_first:
        recommendation = FailFirst(recommendation, faults.recommendation_fail_first, faults.recommendation_error,
                                   "Recommendation Agent")

    # Coordinator
    coordinator_llm = None
    if ctx.options.driver != "fixed":
        coordinator_llm = ctx.llm
        if faults.coordinator_llm_down:
            coordinator_llm = _down_llm(settings)
            extra_clients.append(coordinator_llm)
    coordinator = CoordinatorAgent(diagnostic, recommendation, llm=coordinator_llm, settings=settings,
                                   driver=ctx.options.driver)

    try:
        result = coordinator.run(task.study_id, task.patient, session_id=session_id)
        usage = _merge_usage([ctx.llm, *extra_clients])
    finally:
        for client in extra_clients:
            client.close()

    state, stats = result.state, result.stats
    executed = [d.tool for d in result.decisions if d.outcome == "executed"]
    rec_lines = [e.message for e in state.events if e.agent == AgentName.RECOMMENDATION and e.kind == "message"]
    rag_calls = sum(1 for line in rec_lines if RAG_CALL.search(line) or "general guidance" in line)
    rag_failures = sum(1 for line in rec_lines
                       if RAG_FAILURE.search(line) or "general guidance lookup failed" in line)
    details = {
        "executed_tools": executed,
        "decision_path": " > ".join(f"{d.tool}[{d.source}:{d.outcome}]" for d in result.decisions),
        "coordinator_steps": state.coordinator_steps,
        "llm_proposed_calls": stats.proposed_calls, "llm_valid_calls": stats.valid_calls,
        "refused_calls": stats.refused_calls, "fallback_steps": stats.fallback_steps,
        "forced_stops": stats.forced_stops,
        "agent_tool_calls": sum(1 for t in executed if t in (RUN_DIAGNOSTIC, RUN_RECOMMENDATION)),
        "agent_tool_ok": len(state.diagnostic_results) + len(state.recommendation_results),
        "rag_tool_calls": rag_calls, "rag_tool_failures": rag_failures,
        "usage": usage,
    }
    return state, result.report, details


def run_baseline(task: Task, ctx: RunContext, session_id: str) -> tuple[SessionState, FinalReport, dict]:
    """Run the baseline on the same study and patient context and collect comparable usage
    Agent-only faults are skipped when planned runs are constructed
    """
    settings = ctx.settings
    if task.faults.model_service_down:
        settings = make_settings(settings, model_service_url=UNREACHABLE_URL)
    llm = (ctx.hooks.baseline_llm if ctx.hooks and ctx.hooks.baseline_llm else ctx.llm)
    scorer = ModelServiceClient(settings)
    try:
        state = BaselineAgent(scorer, llm, settings).run(task.study_id, task.patient, session_id=session_id)
    finally:
        scorer.close()
    scored = not any(e.agent == AgentName.MODEL_SERVICE and e.kind == "error" for e in state.events)
    details = {
        "executed_tools": [], "decision_path": "model_service /predict > one baseline LLM call",
        "coordinator_steps": 0, "llm_proposed_calls": 0, "llm_valid_calls": 0, "refused_calls": 0,
        "fallback_steps": 0, "forced_stops": 0,
        "agent_tool_calls": 1, "agent_tool_ok": 1 if scored else 0,  # the one Model Service call
        "rag_tool_calls": 0, "rag_tool_failures": 0,
        "usage": _merge_usage([llm]),
    }
    if state.final_report is None:  # A completed baseline run must provide a final report
        raise RuntimeError("the baseline finished without a report")
    return state, state.final_report, details


def observations(task: Task, report: FinalReport | None, details: dict, checks: list[Check]) -> list[str]:
    """Explain injected faults, recovery decisions and failed checks for the human review sheet"""
    notes = []
    if task.faults.active():
        notes.append(f"Injected: {', '.join(task.faults.active())}")
    if report is not None:
        for error in report.errors[:2]:
            notes.append(f"Error: {error[:160]}")
        if report.recheck_rounds:
            notes.append(f"{report.recheck_rounds} recheck round(s); {len(report.disagreements)} open disagreement(s)")
    if details.get("refused_calls"):
        notes.append(f"Guardrails refused {details['refused_calls']} LLM tool call(s)")
    if details.get("fallback_steps"):
        notes.append(f"Fixed fallback policy took {details['fallback_steps']} step(s)")
    if details.get("forced_stops"):
        notes.append("Stopped at the Coordinator step limit")
    if details.get("rag_tool_failures"):
        notes.append(f"{details['rag_tool_failures']} retrieval/search call(s) failed")
    for check in checks:
        if not check.passed:
            notes.append(f"FAILED {check.name}: {check.detail}")
    return notes


def run_one(task: Task, system: str, repeat: int, ctx: RunContext) -> dict:
    """Execute one system, task and repeat and return a serializable record
    Exceptions from the system runner become crash records so later tasks can still run
    """
    short = "ma" if system == "multi_agent" else "bl"
    session_id = f"{short}-{task.id}-r{repeat}"
    record = {
        "run_id": f"{system}:{task.id}:r{repeat}", "system": system, "task_id": task.id, "split": task.split,
        "difficulty": task.difficulty, "category": task.category, "title": task.title, "repeat": repeat,
        "driver": ctx.options.driver if system == "multi_agent" else "-", "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected_status": list(task.expect.status), "injected_faults": task.faults.active(),
        "trace_file": f"logs/traces/{session_id}.jsonl", "crashed": False, "crash_error": "",
    }
    # Reset counters before each run so recorded usage belongs only to this task and repeat
    ctx.llm.reset_usage()
    if ctx.hooks and ctx.hooks.baseline_llm:
        ctx.hooks.baseline_llm.reset_usage()
    started = time.perf_counter()
    try:
        runner = run_multi_agent if system == "multi_agent" else run_baseline
        state, report, details = runner(task, ctx, session_id)
    except Exception as error:  # noqa: BLE001 - a crash is a result to record, not a reason to stop the evaluation
        logger.exception("run crashed: %s", record["run_id"])
        record.update({"crashed": True, "crash_error": f"{type(error).__name__}: {error}",
                       "wall_ms": round((time.perf_counter() - started) * 1000, 1), "status": "crashed",
                       "passed": False, "had_error": True, "checks": [], "quality": {"score": 0.0, "groups": {}},
                       "failed_checks": ["crashed"], "observations": [f"CRASHED: {type(error).__name__}: {error}"]})
        return record
    wall_ms = (time.perf_counter() - started) * 1000

    checks = run_checks(task, report, system, details["executed_tools"])
    usage = details.pop("usage")
    record.update(details)
    record.update({
        "wall_ms": round(wall_ms, 1),
        "status": report.status.value,
        "findings": [f.name for f in report.findings],
        "low_confidence": [f.name for f in report.findings if f.low_confidence],
        "urgency": report.urgency.value if report.urgency else "none",
        "evidence_status": report.evidence_status.value if report.evidence_status else "none",
        "citations": report.citations, "impression": report.impression,
        "recommendation_text": report.recommendation_text, "user_action": report.user_action,
        "disagreements": report.disagreements, "errors": report.errors, "recheck_rounds": report.recheck_rounds,
        "had_error": bool(report.errors),
        "llm_calls": sum(u.get("calls", 0) for u in usage.values()),
        "llm_errors": sum(u.get("errors", 0) for u in usage.values()),
        "tokens_in": sum(u.get("tokens_in", 0) for u in usage.values()),
        "tokens_out": sum(u.get("tokens_out", 0) for u in usage.values()),
        "tokens_by_agent": {agent: u.get("tokens_in", 0) + u.get("tokens_out", 0) for agent, u in usage.items()},
        "llm_calls_by_agent": {agent: u.get("calls", 0) for agent, u in usage.items()},
        "checks": [c.to_dict() for c in checks],
        "passed": all(c.passed for c in checks),
        "failed_checks": [c.name for c in checks if not c.passed],
        "quality": quality_score(checks),
        "observations": observations(task, report, details, checks),
    })
    record["tokens_total"] = record["tokens_in"] + record["tokens_out"]
    return record


# --- The whole evaluation ---

def skip_reason(task: Task, system: str, options: EvalOptions) -> str:
    """Explain why an agent-specific fault does not apply to this system or driver"""
    if system == "baseline" and task.faults.agent_internal:
        return "fault targets an agent; the baseline has no agents"
    if system == "multi_agent" and task.faults.coordinator_llm_down and options.driver == "fixed":
        return "the fixed driver uses no Coordinator LLM, so this fault does nothing"
    return ""


def select_tasks(options: EvalOptions) -> list[Task]:
    """Load selected splits and reject requested task identifiers that are not present"""
    tasks = load_tasks(tuple(options.splits))
    if options.task_ids:
        wanted = {t.upper() for t in options.task_ids}
        unknown = wanted - {t.id for t in tasks}
        if unknown:
            raise SystemExit(f"unknown task id(s) {sorted(unknown)} in splits {options.splits}; see --list")
        tasks = [t for t in tasks if t.id in wanted]
    return tasks


def planned_runs(tasks: list[Task], options: EvalOptions) -> list[tuple[str, Task, int]]:
    """Expand systems, tasks and repeats while omitting combinations with a skip reason"""
    return [(system, task, repeat) for system in options.systems for task in tasks
            for repeat in range(options.repeats) if not skip_reason(task, system, options)]


def _git_info() -> dict:
    """Record the code revision and working-tree change count when Git is available"""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout
        return {"commit": commit or "unknown", "uncommitted_changes": len(dirty.splitlines())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "uncommitted_changes": None}


def preflight(settings: Settings, llm: LLMClient, options: EvalOptions, hooks: EvalHooks | None,
              diagnostic_impl: str) -> dict:
    """Check service and model availability before a long run and capture setup details
    Test hooks bypass model health checks, while a missing local index produces a warning
    """
    info: dict = {"rag_index_found": RAG_INDEX.exists(), "tavily_key_set": get_settings().has_tavily_key,
                  "web_search": "disabled (--no-web)" if options.no_web else "enabled"}
    if hooks is not None:
        info["ollama"] = "not used (test hooks)"
        return info
    health = llm.health()
    info["ollama"] = health
    if health.get("ollama") != "ok":
        raise SystemExit(f"Ollama is not reachable at {settings.ollama_host}. Start it (`ollama serve`) and pull the "
                         f"models in .env, then run again. Details: {health.get('error')}")
    needed = {"recommendation"} | ({"baseline"} if "baseline" in options.systems else set())
    if options.driver != "fixed":
        needed.add("coordinator")
    if diagnostic_impl == "agent":  # the real Diagnostic Agent uses MedGemma; the interim adapter uses no LLM
        needed.add("diagnostic")
    missing = [f"{a} ({health['models'][a]['model']})" for a in sorted(needed) if not health["models"][a]["available"]]
    if missing:
        raise SystemExit(f"Models not pulled: {missing}. Run `ollama pull <name>` for each, then run again.")
    if not info["rag_index_found"]:
        print(f"WARNING: no RAG index at {RAG_INDEX}; local retrieval will fail. Build it with "
              "`python agents/recommendation_agent/rag/build_index.py` (from that folder).")
    return info


def write_config(run_dir: Path, options: EvalOptions, settings: Settings, tasks: list[Task], extra: dict) -> None:
    """Save options, sanitized settings and environment details needed to interpret the results"""
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "options": asdict(options),
        "task_files": TASK_FILES,
        "n_tasks": len(tasks),
        "git": _git_info(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "settings": settings.safe_summary(),
        "model_scores": "controlled fixture served by the real Model Service in replay mode (not trained-model output)",
        **extra,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def run_evaluation(options: EvalOptions, run_dir: Path, hooks: EvalHooks | None = None) -> Path:
    # Keep the console to one line per run. Everything still goes into the records and the session traces
    """Serve replay predictions, run unfinished combinations and append each result immediately
    Reuse saved records on resume, close service resources and regenerate the report outputs
    """
    for noisy in ("agents.coordinator.wiring", "recommendation_agent", "shared.llm"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(options)
    runs = planned_runs(tasks, options)
    # Stable identifiers make resume independent of how many earlier runs passed
    done = {r["run_id"] for r in eval_metrics.load_records(run_dir)}
    todo = [(s, t, r) for s, t, r in runs if f"{s}:{t.id}:r{r}" not in done]

    fixture = run_dir / "predictions_fixture.json"
    if not fixture.exists():  # on --resume the same fixture is reused
        fixture.write_text(build_replay_file(load_tasks()).model_dump_json(indent=2), encoding="utf-8")

    base = get_settings()
    settings = make_settings(base, log_dir=str(run_dir / "logs"), tool_timeout_s=options.tool_timeout_s,
                             model_mode="replay", replay_predictions=str(fixture))
    server = ModelServiceServer(settings)
    server.start()
    settings = make_settings(settings, model_service_url=server.url)
    llm = LLMClient(settings)
    try:
        _, diag_impl, diag_note = load_diagnostic(settings, llm)
        if not (run_dir / "run_config.json").exists():
            info = preflight(settings, llm, options, hooks, diag_impl)
            write_config(run_dir, options, settings, tasks, {
                "diagnostic_impl": diag_impl, "diagnostic_note": diag_note, "test_hooks_used": hooks is not None,
                "preflight": info,
                "skipped": [{"system": s, "task_id": t.id, "reason": skip_reason(t, s, options)}
                            for s in options.systems for t in tasks if skip_reason(t, s, options)],
            })
        elif hooks is None:
            preflight(settings, llm, options, hooks, diag_impl)

        print(f"{len(todo)} run(s) to do ({len(runs) - len(todo)} already done) -> {run_dir}")

        ctx = RunContext(settings, llm, options, hooks)
        for index, (system, task, repeat) in enumerate(todo, start=1):
            record = run_one(task, system, repeat, ctx)
            # Save before starting the next task so completed work survives interruption
            with (run_dir / "runs.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, default=str) + "\n")
            verdict = "PASS" if record["passed"] else "FAIL"
            print(f"[{index}/{len(todo)}] {system:<11} {task.id} ({task.split}, {task.difficulty:<9}) "
                  f"{record['status']:<12} {verdict} quality={record['quality']['score']:.2f} "
                  f"{record['wall_ms'] / 1000:.1f}s {record.get('tokens_total', 0)} tokens"
                  + (f"  [{', '.join(record['failed_checks'])}]" if record["failed_checks"] else ""))
    except KeyboardInterrupt:
        print(f"\nStopped. Finished runs are saved; continue with: python -m evaluation.run_eval --resume {run_dir}")
    finally:
        llm.close()
        server.stop()
    eval_metrics.write_outputs(run_dir)
    return run_dir


# --- Command line ---

def _options_from_args(args) -> EvalOptions:
    """Expand command-line choices into the options stored with the evaluation"""
    splits = ["development", "unseen"] if args.split == "all" else [args.split]
    systems = list(SYSTEMS) if args.system == "both" else [args.system]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else []
    return EvalOptions(splits=splits, task_ids=tasks, systems=systems, driver=args.driver, repeats=args.repeats,
                       tool_timeout_s=args.tool_timeout, no_web=args.no_web)


def main(argv: list[str] | None = None) -> int:
    """Dispatch task listing, report regeneration, resume or a new evaluation from command-line options"""
    parser = argparse.ArgumentParser(description="Run the system evaluation on the test set.")
    parser.add_argument("--split", choices=["development", "unseen", "all"], default="all")
    parser.add_argument("--tasks", help="comma-separated task ids, e.g. D01,U06")
    parser.add_argument("--system", choices=["multi_agent", "baseline", "both"], default="both")
    parser.add_argument("--driver", choices=["tools", "json", "fixed"], default="tools",
                        help="Coordinator driver (default tools = Qwen tool calls)")
    parser.add_argument("--repeats", type=int, default=1, help="run every task this many times")
    parser.add_argument("--tool-timeout", type=float, default=DEFAULT_TOOL_TIMEOUT_S, help="TOOL_TIMEOUT_S for this run")
    parser.add_argument("--no-web", action="store_true", help="disable Tavily for reproducible runs")
    parser.add_argument("--out-dir", type=Path, help="where to write results (default: a new timestamped folder)")
    parser.add_argument("--resume", type=Path, help="continue an earlier run folder with its original options")
    parser.add_argument("--report", type=Path, help="only rebuild the CSV/JSON/Markdown outputs of a run folder")
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        for task in load_tasks():
            faults = f"  faults: {', '.join(task.faults.active())}" if task.faults.active() else ""
            print(f"{task.id}  {task.split:<11} {task.difficulty:<9} {task.category:<21} {task.title}{faults}")
        return 0
    if args.report:
        eval_metrics.write_outputs(args.report)
        print(f"outputs rebuilt in {args.report}")
        return 0

    if args.resume:
        config = json.loads((args.resume / "run_config.json").read_text(encoding="utf-8"))
        options, run_dir = EvalOptions(**config["options"]), args.resume
    else:
        options = _options_from_args(args)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.out_dir or RESULTS_DIR / f"{stamp}_{options.driver}"
    run_evaluation(options, run_dir)
    print(f"\nDone. Report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
