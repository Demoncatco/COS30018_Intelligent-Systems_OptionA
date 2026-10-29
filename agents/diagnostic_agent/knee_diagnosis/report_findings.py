"""
Report-side findings: either run extract_knee_findings (MedGemma via Ollama) on report text,
or look a study up in the findings.csv that script already produced.

Each finding is {"abnormality": <one of the 12 labels>, "severity": <short phrase>}.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .config import ABNORMALITIES
from .report_extraction import query_model   # extract_knee_findings.py, unchanged

log = logging.getLogger(__name__)


class ReportFindings:
    def __init__(self, findings_csv=None, ollama_host: str = "http://localhost:11434",
                 model: str = "medgemma:4b", num_ctx: int = 8192, timeout: int = 180,
                 retries: int = 2, max_chars: int = 6000):
        self.findings_csv = Path(findings_csv) if findings_csv else None
        self.host, self.model, self.num_ctx = ollama_host, model, num_ctx
        self.timeout, self.retries, self.max_chars = timeout, retries, max_chars
        self._index: dict[str, list[dict]] | None = None

    # ---- findings.csv lookup -------------------------------------------------
    def _load_index(self) -> dict[str, list[dict]]:
        if self._index is None:
            self._index = {}
            if self.findings_csv and self.findings_csv.is_file():
                df = pd.read_csv(self.findings_csv)
                for uid, g in df.groupby("StudyInstanceUID"):
                    rows = g[g.Abnormality.isin(ABNORMALITIES)]       # drops the 'None detected' marker rows
                    self._index[str(uid)] = [{"abnormality": r.Abnormality, "severity": str(r.Severity)}
                                             for r in rows.itertuples()]
            else:
                log.warning("findings.csv not found at %s; only live report extraction is available", self.findings_csv)
        return self._index

    def lookup(self, study_uid: str | None) -> list[dict] | None:
        """Findings for a study already processed, [] if its report had none, None if it is not in the file."""
        return self._load_index().get(str(study_uid)) if study_uid else None

    # ---- live extraction -----------------------------------------------------
    def extract(self, report_text: str) -> list[dict]:
        """Runs the local MedGemma model on one report. Raises the last error if every attempt fails."""
        text = (report_text or "").strip()[: self.max_chars]
        if not text:
            return []
        last = None
        for attempt in range(self.retries + 1):
            try:
                return query_model(text, self.host, self.model, self.num_ctx, self.timeout)
            except Exception as e:      # Ollama down, timeout, bad JSON
                last = e
                time.sleep(1 + attempt)
        raise last

    # ---- what the agent calls ------------------------------------------------
    def get(self, study_uid: str | None, report_text: str | None = None) -> tuple[list[dict] | None, str | None, list[str]]:
        """
        Returns (findings, source, warnings). source is 'report', 'findings_csv' or None.
        A supplied report is extracted fresh; if that fails, falls back to findings.csv.
        """
        warnings = []
        if report_text and report_text.strip():
            try:
                return self.extract(report_text), "report", warnings
            except Exception as e:
                warnings.append(f"report extraction failed ({type(e).__name__}); trying findings.csv")
        cached = self.lookup(study_uid)
        if cached is not None:
            return cached, "findings_csv", warnings
        return None, None, warnings
