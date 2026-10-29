"""
DiagnosticAgent for the knee project.

    agent = DiagnosticAgent(llm, settings)
    response = agent(request, session_id=...)

What it does per call:
  1. image model:  uploaded DICOM files -> probability for each of the 12 findings   (knee_diagnosis.KneePredictor)
  2. report side:  report text -> MedGemma extraction, or a lookup in findings.csv    (knee_diagnosis.ReportFindings)
  3. fuse both into one row per finding, and draft a short summary / recommendation

  
Need to implement:
- a way to pull dicom files/ path and associated study or series IDs from input 
- output to to diagnostic or coorindator agent 
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .knee_diagnosis import KneePredictor, ReportFindings, fuse
from .knee_diagnosis.config import ABNORMALITIES

log = logging.getLogger(__name__)

DISCLAIMER = ("Automated research output, not a diagnosis. Findings must be reviewed by a qualified "
              "radiologist or orthopaedic clinician before any clinical decision.")


def _setting(settings: Any, name: str, env: str, default: Any = None) -> Any:
    """settings.<name> if present, else the environment variable, else the default."""
    value = settings.get(name) if isinstance(settings, dict) else getattr(settings, name, None)
    if value in (None, ""):
        value = os.environ.get(env, default)
    return value


def _get(obj: Any, *names: str) -> Any:
    """First non-empty attribute / dict key among `names`, else None."""
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v not in (None, "", []):
            return v
    return None


def _check_finding_names() -> None:
    """The Coordinator indexes request.finding[name] for every name in shared.schema.FINDINGS, so ours must match."""
    try:
        from shared.schema import FINDINGS
    except ImportError:          # running outside the main project (e.g. standalone tests)
        return
    if set(FINDINGS) != set(ABNORMALITIES):
        log.warning("finding names differ from shared.schema.FINDINGS: only in model=%s, only in schema=%s",
                    sorted(set(ABNORMALITIES) - set(FINDINGS)), sorted(set(FINDINGS) - set(ABNORMALITIES)))


@dataclass
class KneeInputs:
    dicom_paths: list[str]
    report_text: str | None = None
    study_uid: str | None = None


class DiagnosticAgent:
    def __init__(self, llm: Any, settings: Any):
        self.llm = llm
        self.settings = settings
        here = Path(__file__).parent
        _check_finding_names()
        # Same threshold the Coordinator/Recommendation agent use to decide which findings count as "reported".
        threshold = _setting(settings, "knee_threshold", "KNEE_THRESHOLD", None)
        if threshold is None:
            threshold = _get(settings, "report_threshold") or 0.5
        self.predictor = KneePredictor(
            model_path=_setting(settings, "knee_model_path", "KNEE_MODEL_PATH", here / "weights" / "dicom_model.pt"),
            threshold=float(threshold),
        )
        # The Coordinator stops waiting after TOOL_TIMEOUT_S, so live report extraction is kept short: one
        # attempt, 60 s. If it fails, the agent falls back to findings.csv. Keep this below TOOL_TIMEOUT_S.
        self.reports = ReportFindings(
            findings_csv=_setting(settings, "knee_findings_csv", "KNEE_FINDINGS_CSV", here / "data" / "findings.csv"),
            ollama_host=_setting(settings, "ollama_host", "OLLAMA_HOST", "http://localhost:11434"),
            model=_setting(settings, "knee_report_model", "KNEE_REPORT_MODEL", "medgemma:4b"),
            timeout=int(_setting(settings, "knee_report_timeout", "KNEE_REPORT_TIMEOUT", 60)),
            retries=0,
        )

    def warm_up(self) -> None:
        """Load the model weights now (call at app start) so the first request is not slow."""
        self.predictor._get_model()

    # ------------------------------------------------------------------ ADAPTER 1
    def _read_request(self, request: Any) -> KneeInputs:
        """
        ADAPTER: map the project's request object onto KneeInputs.
        The attribute names tried below are GUESSES. Replace them with the real field names.
        """
        paths = _get(request, "dicom_paths", "dicom_files", "files", "file_paths", "study_dir", "upload_dir")
        if paths is None:
            raise ValueError("request contains no DICOM files/folder (adapt DiagnosticAgent._read_request)")
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        return KneeInputs(
            dicom_paths=[str(p) for p in paths],
            report_text=_get(request, "report", "report_text", "radiology_report"),
            study_uid=_get(request, "study_instance_uid", "study_uid", "study_id"),
        )

    # ------------------------------------------------------------------ ADAPTER 2
    def _build_response(self, session_id: str, result: dict) -> Any:
        """
        ADAPTER: turn `result` into the project's DiagnosticAgentResponse, following prediction_to_diagnostic()
        in agents/coordinator/runners.py. Until that is wired in, the plain dict is returned.

        result keys: finding ({label: probability}), impression (text), findings (fused rows), summary,
                     recommendation, disclaimer, study_instance_uid, series_used, missing_planes,
                     report_source, warnings, session_id
        `finding` and `impression` are what downstream agents read (request.finding[name], request.impression);
        `confidence` still has to be filled in here with the project's own rule.
        """
        return result

    # ------------------------------------------------------------------
    def __call__(self, request: Any, *, session_id: str) -> Any:
        inputs = self._read_request(request)
        image = self.predictor.predict(inputs.dicom_paths)          # raises ValueError / FileNotFoundError with a clear message

        study_uid = image["study_instance_uid"] or inputs.study_uid
        report, source, report_warnings = self.reports.get(study_uid, inputs.report_text)

        findings = fuse(image["probabilities"], report, image["threshold"])
        result = {
            "session_id": session_id,
            "study_instance_uid": study_uid,
            "findings": findings,
            "series_used": image["series_used"],
            "missing_planes": image["missing_planes"],
            "report_source": source,
            "warnings": image["warnings"] + report_warnings,
            "disclaimer": DISCLAIMER,
        }
        result["summary"] = self._summarise(findings, source, image["missing_planes"])
        result["recommendation"] = self._recommend(findings)
        # The fields the Coordinator / Recommendation agent read downstream (request.finding[name], request.impression).
        result["finding"] = dict(image["probabilities"])
        result["impression"] = result["summary"]
        # `confidence` is also read downstream (request.confidence). It is NOT set here: use the project's own rule
        # from prediction_to_diagnostic() in _build_response, so the value means the same as it does for the interim adapter.
        return self._build_response(session_id, result)

    # ------------------------------------------------------------------
    @staticmethod
    def _summarise(findings: list[dict], report_source: str | None, missing_planes: list[str]) -> str:
        positives = [f for f in findings if f["image_positive"]]
        parts = []
        if positives:
            items = ", ".join(f"{f['abnormality']} ({f['image_probability']:.0%})" for f in
                              sorted(positives, key=lambda f: -f["image_probability"]))
            parts.append(f"Imaging model flags: {items}.")
        else:
            parts.append("Imaging model flags no abnormality above the threshold.")
        reported = [f"{f['abnormality']}: {f['report_severity']}" for f in findings if f["report_state"] == "abnormal"]
        if reported:
            parts.append("Report findings: " + "; ".join(reported) + ".")
        elif report_source:
            parts.append("Report lists no abnormal findings among the tracked structures.")
        else:
            parts.append("No radiology report was available to compare against.")
        conflicts = [f["abnormality"] for f in findings if f["agreement"] in ("conflict", "report_only")]
        if conflicts:
            parts.append("Image model and report disagree on: " + ", ".join(conflicts) + ".")
        if missing_planes:
            parts.append(f"Missing imaging plane(s): {', '.join(missing_planes)}; lower confidence.")
        return " ".join(parts)

    def _recommend(self, findings: list[dict]) -> str:
        """
        Placeholder rules-based text. To have the LLM write this, call self.llm here with the fused findings,
        using whatever interface the project's `llm` object exposes (not visible to me).
        """
        flagged = [f["abnormality"] for f in findings if f["agreement"] in ("confirmed", "image_only", "conflict", "report_only")]
        if not flagged:
            return "No abnormality flagged. Clinical correlation and routine radiologist review are still advised."
        return ("Flagged for clinician review: " + ", ".join(flagged) +
                ". Recommend radiologist confirmation against the full study and correlation with clinical examination "
                "before any treatment decision.")
