"""
Diagnostic Agent: DiagnosticAgentRequest -> DiagnosticAgentResponse  (contracts: shared/schema.py,
agents/coordinator/runners.py).

    agent = DiagnosticAgent(llm, settings)
    response = agent(request, session_id=...)          # matches runners.DiagnosticRunner; raises on failure

Where the image scores come from (chosen once, when the agent is created, and written to every trace):
  1. local model   agents/diagnostic_agent/weights/best_model.pt exists (or settings.diagnostic_local_checkpoint /
                   env DIAGNOSTIC_LOCAL_CHECKPOINT points at a file): the notebook-trained network runs in this process
                   on the study's DICOM files (local_model/), found through the same LIVE_* settings as the Model Service.
  2. Model Service otherwise: POST {MODEL_SERVICE_URL}/predict (model_client.py), retried inside TOOL_TIMEOUT_S.

What the agent adds around the scores:
  - numbers, cross-plane variance, overall confidence and the FINALIZE/ESCALATE decision come from
    prediction_to_diagnostic() in agents/coordinator/runners.py, so they match the interim adapter exactly
  - a MedGemma-worded impression (impression.py), with a rule-based fallback
  - a real trace, and a cross-check against the report findings in findings.csv (trace only, numbers untouched)

Errors follow the Coordinator's rule "agents raise on failure": a missing study raises runners.StudyNotAvailableError
(never retried); other Model Service failures are retried while time remains, then raised.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agents.coordinator.runners import StudyNotAvailableError, prediction_to_diagnostic
from shared.config import Settings, get_settings
from shared.schema import FINDINGS, DiagnosticAgentRequest, DiagnosticAgentResponse, PredictionResponse

from .impression import build_impression
from .model_client import ModelServiceClient
from .report_crosscheck import ReportCrossCheck

if TYPE_CHECKING:
    from shared.llm import LLMClient

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
DEFAULT_CHECKPOINT = HERE / "weights" / "best_model.pt"
DEFAULT_FINDINGS_CSV = HERE / "findings.csv"

DEFAULT_MAX_RETRIES = 2         # failed Model Service calls retried while time is left in the budget below
TIME_BUDGET_SHARE = 0.9         # the Coordinator abandons the call after TOOL_TIMEOUT_S, so retries stop before that
MIN_ATTEMPT_S = 1.0


class DiagnosticAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        *,
        predictor: Callable[..., PredictionResponse] | None = None,
        findings_csv: str | os.PathLike | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.llm = llm
        self.settings = settings or (llm.settings if llm is not None else get_settings())
        self.max_retries = max_retries
        if predictor is not None:
            self.predictor, self.source = predictor, "injected predictor"
        else:
            self.predictor, self.source = self._choose_predictor()
        # Local predictor is also handed to the baseline, so both systems score with the same model
        self.scorer = self.predictor if hasattr(self.predictor, "predict") else None
        csv = findings_csv or os.environ.get("KNEE_FINDINGS_CSV") or DEFAULT_FINDINGS_CSV
        self.report_check = ReportCrossCheck(csv, FINDINGS)

    def _choose_predictor(self) -> tuple[Callable[..., PredictionResponse], str]:
        configured = (getattr(self.settings, "diagnostic_local_checkpoint", None)
                      or os.environ.get("DIAGNOSTIC_LOCAL_CHECKPOINT"))
        checkpoint = Path(configured) if configured else DEFAULT_CHECKPOINT
        if checkpoint.is_file():
            from .local_model.predictor import LocalKneePredictor   # imports torch only on this path

            log.info("Diagnostic Agent: local model %s", checkpoint)
            return LocalKneePredictor(checkpoint, self.settings), f"local model ({checkpoint.name})"
        log.info("Diagnostic Agent: no local checkpoint at %s; using the Model Service", checkpoint)
        return ModelServiceClient(self.settings), f"Model Service ({self.settings.model_service_url})"

    # ------------------------------------------------------------------
    def __call__(self, request: DiagnosticAgentRequest, *, session_id: str) -> DiagnosticAgentResponse:
        trace = [f"session {session_id}: task_type={request.task_type}; study={request.study_id}"
                 + (f"; focus={request.focus_findings}" if request.focus_findings else ""),
                 f"Image scores from: {self.source}."]
        prediction, retries = self._score(request.study_id, trace)

        # Numbers, variance, overall confidence and the FINALIZE/ESCALATE decision come from the project's own rule.
        base = prediction_to_diagnostic(prediction, request, self.settings.report_threshold)
        flags = list(base.quality_flags)        # only the schema's own flags: low_slice_count, missing_plane:<plane>
        per_plane = {p.plane: dict(p.scores) for p in prediction.per_plane}
        stub = prediction.model_version.startswith("stub-")   # the schema says 'stub-*' means random output

        trace.append(f"Model {prediction.model_version}: planes {list(per_plane)}, {prediction.latency_ms:.0f} ms; "
                     f"laterality {prediction.laterality} ({prediction.laterality_source}).")
        if stub:
            trace.append("The model is a stub: scores are random placeholders.")
        trace.append(f"QC flags: {flags}" if flags else f"QC passed: all {len(per_plane)} planes present.")
        if base.cross_plane_variance:
            trace.append("Planes disagree: " + ", ".join(f"{n} {v:.2f}" for n, v in base.cross_plane_variance.items()))
        trace.extend(self.report_check.trace_lines(request.study_id, prediction.findings, self.settings.report_threshold))

        focus = request.focus_findings if request.task_type.startswith("recheck") else []
        uncertain = sorted(base.cross_plane_variance, key=lambda n: -base.cross_plane_variance[n])
        impression, how = build_impression(
            self.llm, session_id, prediction.findings, self.settings.report_threshold, uncertain,
            flags + (["stub_model"] if stub else []), focus, per_plane)
        trace.append(f"Impression written by {how}.")
        trace.append(f"Overall confidence {base.overall_confidence:.2f}. Decision: {base.decision.value}.")

        # Rebuilt through the constructor so the schema validators run on the final values.
        return DiagnosticAgentResponse(**{**base.model_dump(), "retry_count": retries, "quality_flags": flags,
                                          "impression": impression, "trace": trace})

    # ------------------------------------------------------------------
    def _score(self, study_id: str, trace: list[str]) -> tuple[PredictionResponse, int]:
        """Runs the predictor with retries. Returns (prediction, failed attempts). StudyNotAvailableError is never retried."""
        deadline = time.monotonic() + float(self.settings.tool_timeout_s) * TIME_BUDGET_SHARE
        failures = 0
        while True:
            try:
                return self.predictor(study_id, timeout=max(deadline - time.monotonic(), MIN_ATTEMPT_S)), failures
            except StudyNotAvailableError:
                raise
            except Exception as error:  # noqa: BLE001 - network, timeout, 5xx, bad response or a model error
                failures += 1
                trace.append(f"Scoring failed ({type(error).__name__}: {error}); attempt {failures}.")
                if failures > self.max_retries or deadline - time.monotonic() < MIN_ATTEMPT_S:
                    raise