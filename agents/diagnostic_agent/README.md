# Diagnostic Agent

`DiagnosticAgent(llm, settings)` in `agent.py` is what `agents/coordinator/wiring.py` imports. It matches
`runners.DiagnosticRunner`: `agent(request, session_id=...)` returns a `DiagnosticAgentResponse` and raises on failure.

## Where the image scores come from
Chosen once when the agent is created, and written into every trace ("Image scores from: ...").

| source | when | what runs |
|---|---|---|
| local model | `weights/best_model.pt` exists (or `DIAGNOSTIC_LOCAL_CHECKPOINT` points at a file) | the notebook-trained network, in this process, on the study's DICOM files |
| Model Service | otherwise | `POST {MODEL_SERVICE_URL}/predict`, retried inside `TOOL_TIMEOUT_S` |

The local model takes two kinds of study id:
- UI uploads (`upl-...`): read from `UPLOAD_DIR`, using the series the upload step already picked per plane (`upload.json`).
- Kaggle studies: found the way the Model Service's live mode does: `LIVE_SERIES_CSV` (train_series.csv),
  `LIVE_DATASET_ROOT` and `LIVE_SPLIT_FOLDER`.

With the local model, the backend also gives it to the baseline (`/baseline`), so both systems score with the same
network, and `/health` reports `scoring.source = "local"` (the Model Service is then not needed for scoring). It reuses the repo's `select_series`, `load_series`, `build_prediction` and
`read_laterality`; only the network and its preprocessing (`local_model/`) are specific to the notebook checkpoint.

## Files
| file | role |
|---|---|
| `agent.py` | the agent: picks the score source, retries, applies the project's decision rule, builds the response |
| `local_model/model.py` | the notebook network (ResNet18 per slice, max over slices, 3 planes, Linear 1536 to 12) and `plane_logits` |
| `local_model/preprocess.py` | percentile clip, z-score, 32 even slices, resize to 224 (identical to the notebook) |
| `local_model/predictor.py` | `LocalKneePredictor`: study_id to `PredictionResponse` |
| `model_client.py` | Model Service call; HTTP 404 becomes `StudyNotAvailableError` |
| `impression.py` | MedGemma wording of the impression (`llm.decide`), with guardrails and a rule-based fallback |
| `report_crosscheck.py` | compares `findings.csv` (from `extract_knee_findings.py`) with the image result; trace only |
| `weights/` | `best_model.pt` (committed, about 45 MB) |
| `tests/` | `test_local_model.py`: upload scoring, missing planes, unknown uploads, scorer sharing |

## What is decided where
- Numbers, cross-plane variance, overall confidence and FINALIZE/ESCALATE come from `prediction_to_diagnostic()` in
  `agents/coordinator/runners.py`, so they match the interim adapter exactly. The agent never emits RETRY.
- MedGemma only phrases the impression. Findings, uncertain findings and missing planes are decided by code; text that
  is over 300 characters or names a finding outside those sets is replaced by the rule-based text.
- `findings.csv` never changes a number. Agreement or disagreement with the report is added to the trace.
- Quality flags are the schema's own: `low_slice_count` and `missing_plane:<plane>`.

## After dropping it in
- If a trace says "INTERIM adapter", the agent was not imported; look for an import error in the logs.
- Per-plane scores from the local model are an exact split of one jointly trained head (the mean of the three plane
  logits equals the study logit), not three separately trained models. A missing plane rescales the others.
- A recheck re-runs the same model on the same files, so it gives the same numbers.
- The first local prediction is slow on CPU; compare it with `TOOL_TIMEOUT_S`.
- The checkpoint expects a plain `state_dict`. Retrain with the updated notebook (no left-right flip augmentation).