"""Runs the 20 scenarios and reports valid-call rate, correct-decision rate, override rate and latency.

    python -m evaluation.coordinator_validation [--driver fixed | --model medgemma:4b | --compare]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agents.coordinator.agent import CoordinatorAgent, CoordinatorResult
from agents.coordinator.fakes import ScriptedDiagnostic, ScriptedRecommendation
from evaluation.coordinator_scenarios import SCENARIOS, Scenario
from shared.config import Settings, get_settings
from shared.llm import LLMClient
from shared.schema import DISCLAIMER

DEFAULT_OUT = Path("evaluation/results/coordinator")
VALIDATION_TOOL_TIMEOUT_S = 1.0  # below S09's 1.5 s delay


class _NoThinkSettings(Settings):
    # MedGemma has no thinking mode.
    def think_for(self, agent: str) -> bool | None:
        self.model_for(agent)
        return None


@dataclass
class RunConfig:
    driver: str  # tools / json / fixed
    model: str  # "-" for fixed

    @property
    def label(self) -> str:
        return "fixed" if self.driver == "fixed" else f"{self.driver}:{self.model}"


@dataclass
class ScenarioOutcome:
    scenario: Scenario
    result: CoordinatorResult
    failed_checks: list[str] = field(default_factory=list)
    diagnostic: ScriptedDiagnostic | None = None
    recommendation: ScriptedRecommendation | None = None

    @property
    def passed(self) -> bool:
        return not self.failed_checks


def settings_for(config: RunConfig, base: Settings, out_dir: Path) -> Settings:
    update = {"tool_timeout_s": VALIDATION_TOOL_TIMEOUT_S, "log_dir": str(out_dir)}
    if config.driver != "fixed":
        update["coordinator_model"] = config.model
    data = base.model_dump() | update
    return _NoThinkSettings(**data) if config.driver == "json" else Settings(**data)


def check_outcome(scenario: Scenario, outcome: CoordinatorResult, diag: ScriptedDiagnostic) -> list[str]:
    report, executed = outcome.report, outcome.executed_tools
    failed: list[str] = []
    if report is None:
        return ["no final report"]
    if report.status not in scenario.expected_status:
        failed.append(f"status {report.status.value} not in {[s.value for s in scenario.expected_status]}")
    low, high = scenario.expected_rechecks
    if not low <= report.recheck_rounds <= high:
        failed.append(f"recheck_rounds {report.recheck_rounds} not in [{low}, {high}]")
    for tool in scenario.must_call:
        if tool not in executed:
            failed.append(f"{tool} never executed")
    for tool in scenario.must_not_call:
        if tool in executed:
            failed.append(f"{tool} executed but must not be")
    if scenario.first_recheck_focus:
        rechecks = [r for r in diag.requests if r.task_type.startswith("recheck")]
        if not rechecks:
            failed.append("no recheck was sent")
        elif not set(scenario.first_recheck_focus) <= set(rechecks[0].focus_findings):
            failed.append(f"first recheck focus {rechecks[0].focus_findings} misses {list(scenario.first_recheck_focus)}")
    if scenario.expect_disagreements is not None and bool(report.disagreements) != scenario.expect_disagreements:
        failed.append(f"disagreements present={bool(report.disagreements)}, expected {scenario.expect_disagreements}")
    if scenario.expect_errors is not None and bool(report.errors) != scenario.expect_errors:
        failed.append(f"errors present={bool(report.errors)}, expected {scenario.expect_errors}")
    if scenario.user_action_contains and scenario.user_action_contains.lower() not in report.user_action.lower():
        failed.append(f"user_action {report.user_action!r} does not mention {scenario.user_action_contains!r}")
    if DISCLAIMER not in report.disclaimer:
        failed.append("disclaimer missing")
    return failed


def run_scenario(scenario: Scenario, config: RunConfig, settings: Settings, llm: LLMClient | None,
                 repeat: int = 0) -> ScenarioOutcome:
    diag = ScriptedDiagnostic(scenario.diagnostics)
    rec = ScriptedRecommendation(scenario.recommendations)
    coordinator = CoordinatorAgent(diagnostic=diag, recommendation=rec, llm=llm, settings=settings,
                                   driver=config.driver)  # type: ignore[arg-type]
    session_id = f"val-{scenario.id}-{config.label.replace(':', '-').replace('/', '-')}-r{repeat}"
    result = coordinator.run(scenario.study_id, scenario.patient, session_id=session_id)
    return ScenarioOutcome(scenario, result, check_outcome(scenario, result, diag), diag, rec)


def row_for(outcome: ScenarioOutcome, config: RunConfig, repeat: int) -> dict:
    r, s = outcome.result, outcome.result.stats
    return {
        "scenario": outcome.scenario.id, "difficulty": outcome.scenario.difficulty, "config": config.label,
        "repeat": repeat, "passed": outcome.passed, "failed_checks": " | ".join(outcome.failed_checks),
        "status": r.report.status.value if r.report else "", "recheck_rounds": r.state.recheck_rounds,
        "steps": r.state.coordinator_steps, "llm_calls": s.llm_calls, "proposed_calls": s.proposed_calls,
        "valid_calls": s.valid_calls, "refused_calls": s.refused_calls, "fallback_steps": s.fallback_steps,
        "forced_stops": s.forced_stops, "no_tool_replies": s.no_tool_replies, "llm_errors": s.llm_errors,
        "override_rate": round(r.override_rate, 3) if r.override_rate is not None else "",
        "tokens_in": s.tokens_in, "tokens_out": s.tokens_out, "llm_latency_ms": round(s.llm_latency_ms, 1),
        "wall_ms": round(s.wall_ms, 1), "path": " > ".join(f"{d.tool}[{d.source}:{d.outcome}]" for d in r.decisions),
    }


def summarise(rows: list[dict], config: RunConfig) -> dict:
    def rate(numerator: float, denominator: float) -> float | None:
        return round(numerator / denominator, 3) if denominator else None

    proposed = sum(r["proposed_calls"] for r in rows)
    walls = sorted(r["wall_ms"] for r in rows)
    by_difficulty = {}
    for difficulty in ("simple", "moderate", "difficult", "edge"):
        subset = [r for r in rows if r["difficulty"] == difficulty]
        if subset:
            by_difficulty[difficulty] = rate(sum(r["passed"] for r in subset), len(subset))
    counted_decisions = sum(len([p for p in r["path"].split(" > ") if p and ":invalid]" not in p]) for r in rows)
    overridden = sum(len([p for p in r["path"].split(" > ")
                          if "[fallback:" in p or "[guardrail:" in p or ":refused]" in p]) for r in rows)
    return {
        "config": config.label, "driver": config.driver, "model": config.model, "runs": len(rows),
        "correct_decision_rate": rate(sum(r["passed"] for r in rows), len(rows)),
        "correct_by_difficulty": by_difficulty,
        "valid_call_rate": rate(sum(r["valid_calls"] for r in rows), proposed),
        # Meaningless for the fixed driver.
        "override_rate": None if config.driver == "fixed" else rate(overridden, counted_decisions),
        "latency_ms": {"mean": round(statistics.fmean(walls), 1) if walls else None,
                       "median": round(statistics.median(walls), 1) if walls else None,
                       "p90": walls[min(len(walls) - 1, int(0.9 * len(walls)))] if walls else None},
        "mean_tokens": round(statistics.fmean(r["tokens_in"] + r["tokens_out"] for r in rows), 1) if rows else None,
        "mean_llm_calls": round(statistics.fmean(r["llm_calls"] for r in rows), 2) if rows else None,
        "failed": [f"{r['scenario']}: {r['failed_checks']}" for r in rows if not r["passed"]],
    }


def run_config(config: RunConfig, scenarios: list[Scenario], repeats: int, out_dir: Path,
               base: Settings) -> tuple[list[dict], dict]:
    settings = settings_for(config, base, out_dir)
    llm = None
    if config.driver != "fixed":
        llm = LLMClient(settings)
        try:
            llm.preload(("coordinator",))
        except Exception as error:  # noqa: BLE001 - report and skip this configuration
            print(f"[{config.label}] cannot load the model: {error}", file=sys.stderr)
            llm.close()
            return [], {"config": config.label, "error": str(error)}
    rows = []
    try:
        for repeat in range(repeats):
            for scenario in scenarios:
                outcome = run_scenario(scenario, config, settings, llm, repeat)
                row = row_for(outcome, config, repeat)
                rows.append(row)
                mark = "PASS" if outcome.passed else "FAIL"
                print(f"[{config.label}] {scenario.id} {mark} {row['status']:<12} rechecks={row['recheck_rounds']} "
                      f"{row['wall_ms']:>8.0f} ms  {row['failed_checks']}")
    finally:
        if llm is not None:
            llm.close()
    return rows, summarise(rows, config)


def write_outputs(out_dir: Path, rows: list[dict], summaries: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017 - datetime.UTC needs 3.11
    if rows:
        with (out_dir / f"runs_{stamp}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    table = ["| Config | Correct decisions | Valid calls | Override rate | Median latency (ms) | Mean tokens |",
             "|---|---|---|---|---|---|"]
    for s in summaries:
        if "error" in s:
            table.append(f"| {s['config']} | error: {s['error'][:60]} | | | | |")
            continue
        table.append(f"| {s['config']} | {_pct(s['correct_decision_rate'])} | {_pct(s['valid_call_rate'])} | "
                     f"{_pct(s['override_rate'])} | {s['latency_ms']['median']} | {s['mean_tokens']} |")
    markdown = out_dir / f"comparison_{stamp}.md"
    markdown.write_text("\n".join(table) + "\n", encoding="utf-8")
    return markdown


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", choices=["auto", "tools", "json", "fixed"], default="auto",
                        help="auto = tools for Qwen, json for MedGemma")
    parser.add_argument("--model", help="Ollama tag for the Coordinator (default: COORDINATOR_MODEL)")
    parser.add_argument("--compare", action="store_true", help="run Qwen (tools), MedGemma (json) and fixed")
    parser.add_argument("--medgemma-model", help="MedGemma tag for --compare (default: DIAGNOSTIC_MODEL)")
    parser.add_argument("--only", help="comma-separated scenario ids, e.g. S06,S12")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    base = get_settings()
    scenarios = SCENARIOS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        scenarios = [s for s in SCENARIOS if s.id in wanted]
        if not scenarios:
            parser.error(f"no scenarios match {args.only}")

    if args.compare:
        configs = [RunConfig("tools", args.model or base.coordinator_model),
                   RunConfig("json", args.medgemma_model or base.diagnostic_model),
                   RunConfig("fixed", "-")]
    else:
        model = args.model or base.coordinator_model
        driver = args.driver
        if driver == "auto":
            driver = "json" if "medgemma" in model.lower() or "gemma" in model.lower() else "tools"
        configs = [RunConfig(driver, model if driver != "fixed" else "-")]

    started = time.perf_counter()
    all_rows, summaries = [], []
    for config in configs:
        rows, summary = run_config(config, scenarios, args.repeats, args.out, base)
        all_rows.extend(rows)
        summaries.append(summary)
    table = write_outputs(args.out, all_rows, summaries)
    print("\n" + table.read_text(encoding="utf-8"))
    for s in summaries:
        for failure in s.get("failed", []):
            print(f"  [{s['config']}] {failure}")
    print(f"Results in {args.out}/ ({time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())