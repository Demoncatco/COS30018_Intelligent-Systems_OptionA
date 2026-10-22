# Run the graph on sample inputs. Use --demo for scripted responses.

import argparse
import logging
import os

# Set these before importing the graph; the model libraries read them on import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from .demo_fakes import DEFAULT_DEMO_SCRIPT, DEMO_SCRIPTS, ScriptedLLM, fake_retrieve, fake_search
from .graph import build_graph, initial_state
from .stub_inputs import STUB_INPUTS


def run_one(stub: dict, demo: bool, max_steps: int) -> dict:
    graph_options = {}
    if demo:
        graph_options = {
            "llm": ScriptedLLM(DEMO_SCRIPTS.get(stub["name"], DEFAULT_DEMO_SCRIPT)),
            "retrieve_fn": fake_retrieve,
            "search_fn": fake_search,
        }
    app = build_graph(**graph_options)
    return app.invoke(
        initial_state(stub["finding"], stub.get("user_query"), max_steps),
        config={"recursion_limit": 2 * (max_steps + 2) + 10},
    )


def print_result(final: dict) -> None:
    print("\n--- RESULT ---")
    print(final["recommendation"])
    sources = {
        chunk["source_url"]
        for chunk in final.get("retrieved_chunks", []) + final.get("web_results", [])
    }
    print("\nSources available to the model:", sorted(sources) or "none")
    print(f"Steps: {final['step_count']} decisions | actions: {final['actions_taken']}")
    if final.get("errors"):
        print("Errors caught during the run:")
        for error in final["errors"]:
            print("  -", error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Recommendation Agent graph on stub inputs")
    parser.add_argument("--demo", action="store_true", help="use scripted fakes instead of Qwen/index/Tavily")
    parser.add_argument("--only", help="run only the stub with this name")
    parser.add_argument("--list", action="store_true", help="list the available stub names and exit")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    if args.list:
        for stub in STUB_INPUTS:
            extra = f"  | {stub['user_query']}" if stub.get("user_query") else ""
            print(f"{stub['name']:<28} {stub['finding']}{extra}")
        return
    if args.only and args.only not in {stub["name"] for stub in STUB_INPUTS}:
        raise SystemExit(f"No stub named {args.only!r}. Run with --list to see the names.")

    # Show agent progress and keep library logs quiet.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("recommendation_agent").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # "unauthenticated requests" warning
    for logger_name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "filelock"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if not args.demo and not os.environ.get("TAVILY_API_KEY"):
        print("NOTE: TAVILY_API_KEY not set - web_search will fail (logged) if the model chooses it.\n")

    stubs = [stub for stub in STUB_INPUTS if not args.only or stub["name"] == args.only]
    for stub in stubs:
        print("=" * 78)
        print(f"STUB: {stub['name']}   finding={stub['finding']!r}")
        print("=" * 78)
        print_result(run_one(stub, args.demo, args.max_steps))
        print()


if __name__ == "__main__":
    main()
