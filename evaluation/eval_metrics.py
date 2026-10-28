# Rebuild evaluation summaries and review files from saved run records
# The module docstring below also serves as command-line help and is kept verbatim

"""Turns the raw run records (runs.jsonl) into the metrics, tables and review sheet for the final report.

    python -m evaluation.eval_metrics <run folder>                # rebuild every output from runs.jsonl
    python -m evaluation.eval_metrics human <run folder>/human_review.csv   # summarise the team's 1-5 scores

Every number in summary.json and report.md is computed here from runs.jsonl; nothing is typed in by hand.
Metric definitions are in METRIC_DEFINITIONS below and in evaluation/README.md.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

METRIC_DEFINITIONS = {
    "task_success_rate": "Runs where every applicable automatic check passed / runs.",
    "quality_score_mean": "Mean per-run quality score: share of the check groups (outcome, safety, grounding, "
                          "relevance, uncertainty) that fully pass, counting only groups that apply to the task.",
    "safety_pass_rate": "Runs where every safety check passed / runs.",
    "tool_use_success_rate": "Successful tool calls / tool calls. Tool calls = Coordinator calls to the Diagnostic "
                             "and Recommendation Agents (baseline: its Model Service call) + the Recommendation "
                             "Agent's document-store and web-search calls.",
    "agent_tool_success_rate": "Agent calls that returned a usable result / agent calls.",
    "rag_tool_success_rate": "Document-store and web-search calls that did not raise / those calls.",
    "llm_valid_call_rate": "Coordinator LLM tool calls with a known tool name and schema-valid arguments / proposed "
                           "(multi-agent with an LLM driver only).",
    "mean_coordinator_steps": "Mean Coordinator loop iterations per run (iterations).",
    "mean_recheck_rounds": "Mean Coordinator -> Diagnostic recheck rounds per run.",
    "mean_llm_calls": "Mean LLM calls per run, all agents together (includes JSON retries).",
    "latency_s": "Wall-clock seconds per run (mean, median, 90th percentile, max).",
    "tokens": "Prompt + completion tokens per run, all agents (local Ollama models: no API cost).",
    "error_rate": "Runs with at least one error in the report (injected or not) / runs.",
    "failure_rate": "Runs that crashed, or ended 'failed' when the task expected a usable result / runs.",
    "recovery_rate": "Of the runs with at least one error: those that still ended in an expected final status "
                     "without crashing (recovered, or failed gracefully when failing was the correct outcome).",
}

RUN_COLUMNS = [
    "run_id", "system", "task_id", "split", "difficulty", "category", "title", "repeat", "driver", "status",
    "expected_status", "passed", "quality_score", "failed_checks", "findings", "low_confidence", "urgency",
    "evidence_status", "n_citations", "recheck_rounds", "coordinator_steps", "llm_calls", "tokens_in",
    "tokens_out", "tokens_total", "wall_s", "agent_tool_calls", "agent_tool_ok", "rag_tool_calls",
    "rag_tool_failures", "llm_proposed_calls", "llm_valid_calls", "refused_calls", "fallback_steps",
    "had_error", "crashed", "injected_faults", "errors", "user_action", "decision_path", "trace_file",
]

HUMAN_CRITERIA = ("accuracy", "relevance", "reasoning", "safety_reliability", "failure_handling")


# --- Reading ---

def load_records(run_dir: Path) -> list[dict]:
    """Read saved JSON lines, retaining the latest record for each run identifier
    An absent record file represents an evaluation with no completed runs
    """
    path = Path(run_dir) / "runs.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["run_id"]] = record  # a re-run of the same run id replaces the earlier one
    return list(latest.values())


# --- Metrics ---

def _rate(numerator: float, denominator: float) -> float | None:
    """Round a ratio to three decimal places, or return None when its denominator is zero"""
    return round(numerator / denominator, 3) if denominator else None


def _mean(values: list[float]) -> float | None:
    """Return a rounded arithmetic mean, using None when there are no observations"""
    return round(statistics.fmean(values), 3) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    """Select a rounded index from sorted observations rather than interpolating between values"""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))], 3)


def _safety_ok(record: dict) -> bool:
    """Require a non-crashed run with no failed safety checks"""
    return not record.get("crashed") and all(c["passed"] for c in record.get("checks", []) if c["group"] == "safety")


def aggregate(records: list[dict]) -> dict:
    """Combine run results into success, quality, tool-use, latency, token and recovery metrics
    Recovery is measured only among errored runs and depends on reaching an expected status
    """
    n = len(records)
    if n == 0:
        return {"runs": 0}
    agent_calls = sum(r.get("agent_tool_calls", 0) for r in records)
    agent_ok = sum(r.get("agent_tool_ok", 0) for r in records)
    rag_calls = sum(r.get("rag_tool_calls", 0) for r in records)
    rag_ok = rag_calls - sum(r.get("rag_tool_failures", 0) for r in records)
    proposed = sum(r.get("llm_proposed_calls", 0) for r in records)
    walls = [r["wall_ms"] / 1000 for r in records if "wall_ms" in r]
    # Token statistics omit crash records, which may not contain complete usage
    tokens = [r.get("tokens_total", 0) for r in records if not r.get("crashed")]
    errored = [r for r in records if r.get("had_error")]
    # A graceful failure can count as recovery when failed was an expected task outcome
    recovered = [r for r in errored if not r.get("crashed") and r.get("status") in r.get("expected_status", [])]
    failures = [r for r in records if r.get("crashed")
                or (r.get("status") == "failed" and "failed" not in r.get("expected_status", []))]
    by_agent: Counter = Counter()
    for r in records:
        by_agent.update(r.get("tokens_by_agent", {}))
    return {
        "runs": n,
        "tasks": len({r["task_id"] for r in records}),
        "task_success_rate": _rate(sum(1 for r in records if r.get("passed")), n),
        "quality_score_mean": _mean([r.get("quality", {}).get("score", 0.0) for r in records]),
        "safety_pass_rate": _rate(sum(1 for r in records if _safety_ok(r)), n),
        "tool_use_success_rate": _rate(agent_ok + rag_ok, agent_calls + rag_calls),
        "agent_tool_success_rate": _rate(agent_ok, agent_calls),
        "rag_tool_success_rate": _rate(rag_ok, rag_calls),
        "llm_valid_call_rate": _rate(sum(r.get("llm_valid_calls", 0) for r in records), proposed),
        "mean_coordinator_steps": _mean([r.get("coordinator_steps", 0) for r in records]),
        "mean_recheck_rounds": _mean([r.get("recheck_rounds", 0) for r in records]),
        "mean_llm_calls": _mean([r.get("llm_calls", 0) for r in records]),
        "latency_s": {"mean": _mean(walls), "median": _percentile(walls, 0.5), "p90": _percentile(walls, 0.9),
                      "max": round(max(walls), 3) if walls else None},
        "tokens": {"mean_per_run": _mean(tokens), "total": sum(tokens),
                   "mean_by_agent": {a: round(v / n, 1) for a, v in sorted(by_agent.items())}},
        "error_rate": _rate(len(errored), n),
        "failure_rate": _rate(len(failures), n),
        "recovery_rate": _rate(len(recovered), len(errored)),
        "runs_with_errors": len(errored),
        "crashes": sum(1 for r in records if r.get("crashed")),
        "fallback_steps_total": sum(r.get("fallback_steps", 0) for r in records),
        "guardrail_refusals_total": sum(r.get("refused_calls", 0) for r in records),
    }


def _grouped(records: list[dict], key: str) -> dict:
    """Partition records by a task attribute and calculate the same metrics for each group"""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return {value: aggregate(rs) for value, rs in sorted(groups.items())}


def compare_systems(records: list[dict]) -> dict:
    """Compare both systems using only task identifiers present in both sets of records
    All recorded repeats of those tasks contribute to each system summary
    """
    by_system: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by_system.setdefault(r["system"], {}).setdefault(r["task_id"], []).append(r)
    if not {"multi_agent", "baseline"} <= set(by_system):
        return {}
    # Agent-specific fault tasks may be absent from baseline runs, so compare the intersection
    common = sorted(set(by_system["multi_agent"]) & set(by_system["baseline"]))
    out: dict = {"common_tasks": common}
    for system in ("multi_agent", "baseline"):
        runs = [r for t in common for r in by_system[system][t]]
        urgency_checks = [c for r in runs for c in r.get("checks", []) if c["name"] == "urgency"]
        grounded = [c for r in runs for c in r.get("checks", []) if c["name"] == "advice_is_grounded"]
        stats = aggregate(runs)
        out[system] = {
            "runs": len(runs), "task_success_rate": stats["task_success_rate"],
            "quality_score_mean": stats["quality_score_mean"], "safety_pass_rate": stats["safety_pass_rate"],
            "urgency_correct_rate": _rate(sum(c["passed"] for c in urgency_checks), len(urgency_checks)),
            "grounded_advice_rate": _rate(sum(c["passed"] for c in grounded), len(grounded)),
            "status_correct_rate": _rate(sum(1 for r in runs if r.get("status") in r.get("expected_status", [])),
                                         len(runs)),
            "latency_s_median": stats["latency_s"]["median"], "tokens_mean_per_run": stats["tokens"]["mean_per_run"],
        }
    return out


def build_summary(records: list[dict], config: dict) -> dict:
    """Assemble per-system totals, task-group breakdowns and the shared-task comparison"""
    present = {r["system"] for r in records}
    systems = [s for s in ("multi_agent", "baseline") if s in present]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "computed by evaluation/eval_metrics.py from runs.jsonl",
        "config": {k: config.get(k) for k in ("options", "diagnostic_impl", "git",
                                              "test_hooks_used", "model_scores")},
        "systems": {},
        "comparison": compare_systems(records),
        "failed_check_counts": {},
        "definitions": METRIC_DEFINITIONS,
    }
    for system in systems:
        rs = [r for r in records if r["system"] == system]
        summary["systems"][system] = {
            "overall": aggregate(rs), "by_split": _grouped(rs, "split"),
            "by_difficulty": _grouped(rs, "difficulty"), "by_category": _grouped(rs, "category"),
        }
        summary["failed_check_counts"][system] = dict(Counter(c for r in rs for c in r.get("failed_checks", []))
                                                      .most_common())
    return summary


# --- Writing ---

def _order(record: dict) -> tuple:
    """Sort multi-agent runs before baseline runs, then by task identifier and repeat"""
    return (record["system"] != "multi_agent", record["task_id"], record["repeat"])


def _cell(value) -> str:
    """Flatten saved values into CSV cells without changing the underlying run records"""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return "" if value is None else str(value)


def write_runs_csv(records: list[dict], path: Path) -> None:
    """Write one summary row per run, deriving display fields such as seconds and citation count"""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        for r in sorted(records, key=_order):
            row: dict = {col: _cell(r.get(col)) for col in RUN_COLUMNS}
            row["quality_score"] = r.get("quality", {}).get("score")
            row["n_citations"] = len(r.get("citations", []))
            row["wall_s"] = round(r.get("wall_ms", 0) / 1000, 2)
            writer.writerow(row)


def write_checks_csv(records: list[dict], path: Path) -> None:
    """Write one row per automatic check so failures can be inspected independently"""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "system", "task_id", "split", "difficulty", "category", "check", "group",
                         "passed", "detail"])
        for r in sorted(records, key=_order):
            for c in r.get("checks", []):
                writer.writerow([r["run_id"], r["system"], r["task_id"], r["split"], r["difficulty"], r["category"],
                                 c["name"], c["group"], c["passed"], c["detail"]])


def write_human_review(records: list[dict], path: Path, tasks_by_id: dict) -> bool:
    """Create a review sheet with report text and blank rating cells
    Keep an existing sheet unchanged so reviewer scores survive report regeneration
    """
    if path.exists():
        return False
    columns = ["run_id", "system", "task_id", "split", "difficulty", "category", "title", "purpose",
               "patient_symptoms", "status", "findings", "low_confidence", "urgency", "evidence_status",
               "recommendation_text", "disagreements", "errors", "user_action", "auto_failed_checks",
               "observations", "trace_file", *HUMAN_CRITERIA, "reviewer", "notes"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:  # utf-8-sig so Excel shows accents correctly
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for r in sorted(records, key=_order):
            task = tasks_by_id.get(r["task_id"])
            writer.writerow({
                **{c: _cell(r.get(c)) for c in ("run_id", "system", "task_id", "split", "difficulty", "category",
                                                "title", "status", "findings", "low_confidence", "urgency",
                                                "evidence_status", "recommendation_text", "disagreements",
                                                "errors", "user_action", "observations", "trace_file")},
                "purpose": task.purpose if task else "",
                "patient_symptoms": task.patient.symptoms if task else "",
                "auto_failed_checks": _cell(r.get("failed_checks")),
            })
    return True


def _pct(value) -> str:
    """Display a proportion as a percentage, or n/a for a missing value"""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value, digits: int = 1) -> str:
    """Format a number with the requested precision, or n/a for a missing value"""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _metrics_table(rows: dict[str, dict], label: str) -> list[str]:
    """Build Markdown table lines from non-empty metric groups"""
    lines = [f"| {label} | Runs | Success | Quality | Safety | Tool use | Iterations | Rechecks | "
             "Latency median (s) | Tokens/run | Error rate | Failure rate | Recovery |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        if not m.get("runs"):
            continue
        lines.append(
            f"| {name} | {m['runs']} | {_pct(m['task_success_rate'])} | {_num(m['quality_score_mean'], 2)} | "
            f"{_pct(m['safety_pass_rate'])} | {_pct(m['tool_use_success_rate'])} | "
            f"{_num(m['mean_coordinator_steps'])} | {_num(m['mean_recheck_rounds'], 2)} | "
            f"{_num(m['latency_s']['median'])} | {_num(m['tokens']['mean_per_run'], 0)} | "
            f"{_pct(m['error_rate'])} | {_pct(m['failure_rate'])} | {_pct(m['recovery_rate'])} |")
    return lines


def write_report(records: list[dict], summary: dict, config: dict, path: Path) -> None:
    """Render recorded setup, grouped metrics and per-run results as a Markdown report
    Mark runs that used test hooks so harness checks are not mistaken for real evaluation results
    """
    git = config.get("git") or {}
    options = config.get("options") or {}
    preflight = config.get("preflight") or {}
    ollama = preflight.get("ollama")
    models = []
    if isinstance(ollama, dict) and ollama.get("models"):
        models = [f"{agent}: {m['model']} ({m.get('digest', '?')})" for agent, m in ollama["models"].items()]
    lines = [
        "# System evaluation report",
        "",
        f"Generated {summary['generated_at']} by `evaluation/eval_metrics.py` from `runs.jsonl` "
        f"({len(records)} runs). Do not edit numbers by hand; rebuild with "
        "`python -m evaluation.run_eval --report <this folder>`.",
        "",
        "## Setup",
        "",
        f"- Code: commit `{git.get('commit')}`, {git.get('uncommitted_changes')} uncommitted change(s)",
        f"- Coordinator driver: `{options.get('driver')}`; repeats: {options.get('repeats')}; "
        f"TOOL_TIMEOUT_S: {options.get('tool_timeout_s')}; web search: {preflight.get('web_search')} "
        f"(Tavily key set: {preflight.get('tavily_key_set')})",
        f"- Diagnostic Agent implementation: `{config.get('diagnostic_impl')}` ({config.get('diagnostic_note')})",
        f"- Models: {'; '.join(models) if models else 'see run_config.json'}",
        f"- Image-model scores: {config.get('model_scores')}",
        f"- RAG index found: {preflight.get('rag_index_found')}",
    ]
    if config.get("test_hooks_used"):
        lines += ["", "> **Harness self-test only.** Fake models and retrieval were used (test hooks). These numbers "
                      "say nothing about the real system."]
    if config.get("diagnostic_impl") == "interim":
        lines += ["", "> The Diagnostic Agent is the interim Model Service adapter (no LLM). Diagnostic results are "
                      "therefore deterministic; rechecks return the same scores."]
    for system, data in summary["systems"].items():
        lines += ["", f"## {system}", ""]
        lines += _metrics_table({"all": data["overall"]}, "Group")
        lines += [""] + _metrics_table(data["by_split"], "Split")
        lines += [""] + _metrics_table(data["by_difficulty"], "Difficulty")
        lines += [""] + _metrics_table(data["by_category"], "Category")
        o = data["overall"]
        lines += ["", f"Tool use detail: agent calls {_pct(o['agent_tool_success_rate'])}, retrieval/search calls "
                      f"{_pct(o['rag_tool_success_rate'])}, Coordinator LLM schema-valid calls "
                      f"{_pct(o['llm_valid_call_rate'])}. Latency mean {_num(o['latency_s']['mean'])} s, p90 "
                      f"{_num(o['latency_s']['p90'])} s, max {_num(o['latency_s']['max'])} s. Mean LLM calls "
                      f"{_num(o['mean_llm_calls'])}. Tokens by agent (mean per run): {o['tokens']['mean_by_agent']}. "
                      f"Crashes: {o['crashes']}. Guardrail refusals: {o['guardrail_refusals_total']}. "
                      f"Fallback steps: {o['fallback_steps_total']}."]
        counts = summary["failed_check_counts"].get(system, {})
        if counts:
            lines += ["", "Most frequent failed checks: " + ", ".join(f"`{k}` x{v}" for k, v in counts.items())]

    comparison = summary.get("comparison")
    if comparison:
        lines += ["", "## Multi-agent vs baseline (same tasks)", "",
                  f"{len(comparison['common_tasks'])} tasks run by both systems.", "",
                  "| Metric | Multi-agent | Baseline |", "|---|---|---|"]
        for key, label in [("task_success_rate", "Task success"), ("quality_score_mean", "Quality score"),
                           ("safety_pass_rate", "Safety pass"), ("status_correct_rate", "Correct final status"),
                           ("urgency_correct_rate", "Correct urgency"), ("grounded_advice_rate", "Advice grounded"),
                           ("latency_s_median", "Median latency (s)"), ("tokens_mean_per_run", "Tokens per run")]:
            a, b = comparison["multi_agent"][key], comparison["baseline"][key]
            if key == "quality_score_mean":
                cells = (_num(a, 2), _num(b, 2))
            elif key in ("latency_s_median", "tokens_mean_per_run"):
                cells = (_num(a), _num(b))
            else:
                cells = (_pct(a), _pct(b))
            lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Per-run results", "",
              "| Run | Split | Difficulty | Category | Status | Pass | Quality | Time (s) | Tokens | Failed checks |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=_order):
        lines.append(f"| {r['run_id']} | {r['split']} | {r['difficulty']} | {r['category']} | {r['status']} | "
                     f"{'yes' if r.get('passed') else 'no'} | {r.get('quality', {}).get('score', 0):.2f} | "
                     f"{r.get('wall_ms', 0) / 1000:.1f} | {r.get('tokens_total', 0)} | "
                     f"{', '.join(r.get('failed_checks', [])) or '-'} |")
    skipped = config.get("skipped") or []
    if skipped:
        lines += ["", "Skipped by design: " + "; ".join(f"{s['system']} {s['task_id']} ({s['reason']})"
                                                        for s in skipped)]
    lines += ["", "## Metric definitions", ""] + [f"- **{k}**: {v}" for k, v in METRIC_DEFINITIONS.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(run_dir: Path) -> dict:
    """Regenerate machine-readable summaries and tables from runs.jsonl
    Create the human review sheet only if it does not already exist
    """
    from evaluation.eval_tasks import load_tasks

    run_dir = Path(run_dir)
    records = load_records(run_dir)
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    summary = build_summary(records, config)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_runs_csv(records, run_dir / "runs.csv")
    write_checks_csv(records, run_dir / "checks.csv")
    write_report(records, summary, config, run_dir / "report.md")
    write_human_review(records, run_dir / "human_review.csv", {t.id: t for t in load_tasks()})
    return summary


# --- Human review scores ---

def summarise_human_scores(sheet: Path) -> dict:
    """Validate numeric ratings from one to five, skipping blank and NA cells
    Summarise each criterion separately so missing ratings do not lower its mean
    """
    with Path(sheet).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    problems, scored = [], []
    for row in rows:
        values = {}
        for criterion in HUMAN_CRITERIA:
            raw = (row.get(criterion) or "").strip()
            if raw == "" or raw.upper() == "NA":
                continue
            try:
                value = float(raw)
            except ValueError:
                value = None
            if value is None or not 1 <= value <= 5:
                problems.append(f"{row['run_id']} {criterion}={raw!r} (must be 1-5, blank or NA)")
                continue
            values[criterion] = value
        if values:
            scored.append((row, values))
    if problems:
        raise ValueError("fix these cells in the review sheet: " + "; ".join(problems[:10]))

    def means(items) -> dict:
        """Calculate the mean and sample count using only ratings supplied for each criterion"""
        out = {}
        for criterion in HUMAN_CRITERIA:
            vals = [v[criterion] for _, v in items if criterion in v]
            out[criterion] = {"mean": round(statistics.fmean(vals), 2) if vals else None, "n": len(vals)}
        return out

    result = {"runs_scored": len(scored), "runs_in_sheet": len(rows), "overall": {}, "by_split": {},
              "reviewers": sorted({r.get("reviewer", "") for r, _ in scored if r.get("reviewer")})}
    for system in sorted({r["system"] for r, _ in scored}):
        items = [(r, v) for r, v in scored if r["system"] == system]
        result["overall"][system] = means(items)
        result["by_split"][system] = {split: means([(r, v) for r, v in items if r["split"] == split])
                                      for split in sorted({r["split"] for r, _ in items})}
    return result


def main(argv: list[str] | None = None) -> int:
    """Rebuild outputs or summarise human ratings, otherwise show the existing usage text"""
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == "human":
        sheet = Path(args[1])
        result = summarise_human_scores(sheet)
        out = sheet.with_name("human_summary.json")
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{result['runs_scored']} of {result['runs_in_sheet']} runs scored -> {out}")
        for system, crit in result["overall"].items():
            print(f"  {system}: " + ", ".join(f"{c} {v['mean']} (n={v['n']})" for c, v in crit.items()))
        return 0
    if len(args) == 1:
        write_outputs(Path(args[0]))
        print(f"outputs rebuilt in {args[0]}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
