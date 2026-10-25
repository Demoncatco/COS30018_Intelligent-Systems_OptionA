# Recommendation Agent

Owner: Christina. The "senior doctor": grounds the Diagnostic Agent's findings in trusted sources, writes a
plain-language recommendation with citations, sets an urgency level and raises concerns the Coordinator can
turn into a recheck.

## How it runs

```
Coordinator --CoordinatorRequest--> runner.RecommendationAgent --RecommendationResponse--> Coordinator
                                          |
                                          |  once per reported finding (strongest first, at most 3)
                                          v
                         graph.py: decide_action -> query_doc_store | web_search | synthesize -> ... -> stop
```

| File | What it does |
|---|---|
| `runner.py` | **Adapter for the Coordinator.** Picks the findings, runs the graph per finding, builds the `RecommendationResponse` (text, citations, evidence status, urgency, concerns, recheck requests) |
| `graph.py` | The LangGraph loop and its trace lines |
| `nodes/decide_action.py` | Qwen picks the next action; code guards stop early stops, repeats and "no sources" answers before both lookups were tried |
| `nodes/query_doc_store.py` | Local index search, one chunk per source |
| `nodes/web_search.py` | Tavily search restricted to `trusted_domains.py` (key from `.env`) |
| `nodes/synthesize.py` | Writes the recommendation from the excerpts only, shaped by the Diagnostic impression and patient details (labelled as data, flattened to one line), and enforces the "may indicate / consult a clinician" wording |
| `llm_client.py` | `call_qwen` / `make_qwen_caller`: every call goes through `shared.llm.LLMClient` as the `recommendation` agent |
| `rag/` | Fetch, chunk, embed and search the ~64 trusted documents |

## Rules set by code (not the LLM)

| Output | Rule |
|---|---|
| Findings looked up | Probability `>= REPORT_THRESHOLD`, strongest first, at most `DEFAULT_MAX_FINDINGS` (3) |
| `evidence_status` | `grounded_web` if any web result was used, else `grounded_local` if any local chunk was used, else `no_credible_source_found` |
| `citations` | Every source URL the graph used, once each |
| `urgency` | `urgent` if Fracture; `see_doctor_soon` if ACL, MCL, either meniscus, Contusion, Effusion or Synovitis; otherwise `routine` |
| `concerns` / `suggested_recheck` | Fracture without Contusion or Effusion; or overall confidence below `LOW_CONFIDENCE_THRESHOLD` with findings less than 0.1 above the threshold |
| No reported findings | General "when to see a doctor" guidance from the local index, `routine` |

The urgency and concern rules are triage defaults for the project, not clinical guidance; agree changes with the team.

Within one session a finding's result is reused while the Diagnostic impression stays the same, so the senior's
second review after a recheck only re-applies the rules (fast). If the recheck changed the impression, the finding
is looked up and written again so the text matches the new result.

## Running

```bash
python -m agents.recommendation_agent.run_graph --demo          # scripted fakes, no Ollama
python -m agents.recommendation_agent.run_graph --only acl_plain # real Qwen + index
python -m agents.coordinator --study-id <id> --driver fixed      # through the Coordinator
pytest agents/recommendation_agent                               # no Ollama, index or key needed
```

Each finding costs a few Qwen calls. With `qwen3:8b` and three findings this can take longer than the default
`TOOL_TIMEOUT_S=60`, after which the Coordinator stops waiting; raise it in `.env` for real runs.
