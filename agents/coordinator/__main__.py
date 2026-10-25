"""Smoke run against the Model Service: python -m agents.coordinator --study-id <id> [--driver fixed].

Diagnostic: interim Model Service adapter until the Diagnostic Agent is merged (see build_runners).
Recommendation: the real Recommendation Agent (needs Ollama and the local index).
"""

from __future__ import annotations

import argparse
import json

from agents.coordinator.agent import CoordinatorAgent
from agents.coordinator.runners import ModelServiceDiagnostic
from agents.recommendation_agent.runner import RecommendationAgent
from shared.config import get_settings
from shared.llm import LLMClient
from shared.logging_setup import setup_logging
from shared.schema import PatientMetadata


def build_runners(settings, llm: LLMClient):
    # TODO: swap in the real Diagnostic Agent.
    return ModelServiceDiagnostic(settings), RecommendationAgent(llm=llm, settings=settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Coordinator session and print the FinalReport.")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--driver", choices=["tools", "json", "fixed"], default="tools")
    parser.add_argument("--age", type=int)
    parser.add_argument("--sex", choices=["female", "male", "other", "unknown"], default="unknown")
    parser.add_argument("--symptoms", default="")
    parser.add_argument("--injury-mechanism", default="")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    # One client for the whole run. The Recommendation Agent always needs it; the Coordinator only when an LLM drives it.
    llm = LLMClient(settings)
    diagnostic, recommendation = build_runners(settings, llm)
    coordinator = CoordinatorAgent(diagnostic, recommendation, llm=llm if args.driver != "fixed" else None,
                                   settings=settings, driver=args.driver,
                                   on_event=lambda e: print(f"[{e.seq:>3}] {e.agent.value:<14} {e.kind:<11} {e.message}"))
    patient = PatientMetadata(age=args.age, sex=args.sex, symptoms=args.symptoms,
                              injury_mechanism=args.injury_mechanism)
    result = coordinator.run(args.study_id, patient)
    print(json.dumps(result.report.model_dump(mode="json"), indent=2))
    print(json.dumps(result.stats.to_dict(), indent=2))
    print(f"trace: {settings.log_dir}/traces/{result.state.session_id}.jsonl")
    print(json.dumps(llm.usage_summary(), indent=2))  # tokens and latency per agent
    llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())