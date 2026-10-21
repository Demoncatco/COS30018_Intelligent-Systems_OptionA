# Checks the 20-scenario validation set itself, and runs it through the fixed policy and a simulated Qwen.

from collections import Counter
from dataclasses import replace

import pytest
from coordinator_fakes import PolicyOllama, make_settings

from evaluation import coordinator_validation as v
from evaluation.coordinator_scenarios import SCENARIOS


def test_the_set_has_20_unique_scenarios_across_all_difficulties():
    assert len(SCENARIOS) == 20
    assert len({s.id for s in SCENARIOS}) == 20 and len({s.study_id for s in SCENARIOS}) == 20
    assert Counter(s.difficulty for s in SCENARIOS) == {"simple": 5, "moderate": 6, "difficult": 5, "edge": 4}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_fixed_policy_passes_every_scenario(scenario, tmp_path):
    settings = make_settings(tmp_path, tool_timeout_s=v.VALIDATION_TOOL_TIMEOUT_S)
    outcome = v.run_scenario(scenario, v.RunConfig("fixed", "-"), settings, llm=None)
    assert outcome.passed, outcome.failed_checks


def test_simulated_qwen_runs_the_whole_set_through_the_tool_calling_path(tmp_path):
    settings = make_settings(tmp_path, tool_timeout_s=v.VALIDATION_TOOL_TIMEOUT_S)
    llm = PolicyOllama().client(settings)
    config = v.RunConfig("tools", settings.coordinator_model)
    rows = [v.row_for(v.run_scenario(s, config, settings, llm), config, 0) for s in SCENARIOS]
    summary = v.summarise(rows, config)
    assert summary["correct_decision_rate"] == 1.0, summary["failed"]
    assert summary["valid_call_rate"] == 1.0
    assert 0 <= summary["override_rate"] < 0.2
    assert set(summary["correct_by_difficulty"]) == {"simple", "moderate", "difficult", "edge"}


def test_checks_catch_a_wrong_outcome(tmp_path):
    scenario = next(s for s in SCENARIOS if s.id == "S04")  # must be inconclusive
    broken = replace(scenario, expected_status=SCENARIOS[0].expected_status)
    settings = make_settings(tmp_path, tool_timeout_s=v.VALIDATION_TOOL_TIMEOUT_S)
    outcome = v.run_scenario(broken, v.RunConfig("fixed", "-"), settings, llm=None)
    assert not outcome.passed and "status inconclusive" in outcome.failed_checks[0]


def test_outputs_are_written(tmp_path):
    settings = make_settings(tmp_path, tool_timeout_s=v.VALIDATION_TOOL_TIMEOUT_S)
    config = v.RunConfig("fixed", "-")
    rows = [v.row_for(v.run_scenario(s, config, settings, None), config, 0) for s in SCENARIOS[:3]]
    table = v.write_outputs(tmp_path / "out", rows, [v.summarise(rows, config)])
    assert "| fixed | 100.0% |" in table.read_text()
    assert len(list((tmp_path / "out").glob("runs_*.csv"))) == 1