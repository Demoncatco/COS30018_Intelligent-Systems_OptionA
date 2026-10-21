# Coordinator Agent

Owner: Ren. Qwen tool-calling loop that delegates to the Diagnostic Agent (junior doctor) and the
Recommendation Agent (senior doctor), resolves their disagreements and assembles the `FinalReport`.

## Files

| File | What it does |
|---|---|
| `agent.py` | `CoordinatorAgent`: the loop, LLM drivers, tool execution, events, stats |
| `tools.py` | The 4 tools (`run_diagnostic`, `run_recommendation`, `request_reupload`, `finalize_report`) as Pydantic argument models and Ollama tool schemas |
| `guardrails.py` | Deterministic rules: what may run next, recheck limit, junior/senior rule, fixed fallback policy |
| `report.py` | Builds the `FinalReport`: status, findings, disagreements, fixed disclaimer |
| `prompts.py` | System prompt and messages for the LLM |
| `runners.py` | The call signatures the other agents must match, tool timeouts, interim Model Service adapter |
| `fakes.py` | Scripted Diagnostic/Recommendation stand-ins for tests and validation |
| `__main__.py` | Smoke run: `python -m agents.coordinator --study-id <id>` |

## How a session runs

1. The LLM (Qwen, `COORDINATOR_MODEL`) gets the system prompt, the patient details and the tools.
2. Each step it proposes a tool call. `guardrails.check_call()` accepts it, narrows it (e.g. drops recheck
   findings nobody disputes) or refuses it with a reason and the list of `allowed_tools`.
3. Accepted calls run the agent (with `TOOL_TIMEOUT_S`), the result is summarised back to the LLM, and every
   step is added to `SessionState.events` (live UI trace) and `logs/traces/<session_id>.jsonl`.
4. The session ends on `request_reupload` (status `inconclusive`) or `finalize_report`
   (`completed` / `partial` / `failed`, decided by code from the results, never by the LLM).

### Guardrails

| Rule | Type |
|---|---|
| `study_id` is never a tool argument (the LLM cannot mistype it) | design |
| No recommendation or report on unusable images (`needs_reupload`) | hard |
| No re-upload request when the images were usable | hard |
| Max `MAX_RECHECK_ROUNDS` (default 2) rechecks; a failed recheck still uses a round | hard |
| After senior review, a recheck may only focus on the senior's `suggested_recheck` findings | hard |
| The senior must review a recheck result before another recheck | hard |
| Each agent call: one retry after a failure, then move on | hard |
| Finalising while the senior still wants a recheck (or has not reviewed) needs a second attempt; whatever is open goes into `disagreements` | soft |
| Result for a different study than requested is rejected as a failure | hard |
| `NO_CREDIBLE_SOURCE_FOUND` is shown as a gap; the agent's text and citations are dropped | report |
| Fixed `DISCLAIMER` on every report | report |
| LLM down, 2 replies without a tool call, or 3 refusals in a row: the fixed policy takes one step | fallback |
| `MAX_COORDINATOR_STEPS` reached: forced finalise / re-upload | fallback |

### Junior / senior rule

When the senior's review lists `suggested_recheck`, the Coordinator sends the junior a targeted
`recheck: <concern>` with those `focus_findings`, then asks the senior to review the new result. If the senior
still disputes a finding after the last round, the finding is flagged `low_confidence` and the dispute is
listed in `FinalReport.disagreements`.

## Plugging in the real agents

The Coordinator only needs two callables (see `runners.py`):

```python
def diagnostic(request: DiagnosticAgentRequest, *, session_id: str) -> DiagnosticAgentResponse: ...
def recommendation(request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse: ...
```

Raise any exception on failure. Then:

```python
from agents.coordinator import CoordinatorAgent
from shared.llm import LLMClient

coordinator = CoordinatorAgent(diagnostic=run_diagnostic_agent, recommendation=run_recommendation_agent,
                               llm=LLMClient(), on_event=push_to_ui)
result = coordinator.run(study_id, patient_metadata)   # or run_session(state) to poll a SessionState
result.report, result.stats.to_dict(), result.override_rate
```

Until then `python -m agents.coordinator --study-id <id>` uses the interim `ModelServiceDiagnostic`
(Model Service `/predict`, no LLM) and a Recommendation placeholder that fails, so reports are `partial`.

## Validation (20 scenarios)

`evaluation/coordinator_scenarios.py` scripts what the two agents return in 20 situations
(5 simple, 6 moderate, 5 difficult, 4 edge), so the Coordinator's decisions are tested in isolation.

```bash
python -m evaluation.coordinator_validation                       # Qwen with native tool calls
python -m evaluation.coordinator_validation --driver fixed        # no LLM: fixed-workflow reference
python -m evaluation.coordinator_validation --model medgemma:4b   # optional MedGemma comparison (JSON driver)
python -m evaluation.coordinator_validation --compare --repeats 3 # all three, one table
```

| Metric | Definition |
|---|---|
| Valid-call rate | Tool calls with a known name and schema-valid arguments / tool calls the LLM proposed |
| Correct-decision rate | Scenarios passing every check (status, recheck count, required/forbidden tools, recheck focus, disagreements, errors, user action, disclaimer) / scenarios |
| Override rate | Decisions made by code instead of the LLM (refusal, fallback step, forced stop) / all decisions |
| Latency | Wall-clock per scenario; agents are scripted, so this is the Coordinator's own cost |

Results go to `evaluation/results/coordinator/` (CSV per run, JSON summary, Markdown table, traces).

## Known limitations

- `ReportedFinding.per_plane` stays empty: `DiagnosticAgentResponse` has no per-plane scores. Adding them
  to the schema would let the report show them.
- Timed-out agent calls keep running in a background thread until they return (Python cannot kill threads);
  their result is discarded.
- Token counts cover the Coordinator's own LLM calls. Sub-agent tokens appear in `LLMClient.usage_summary()`
  when the agents share the client.