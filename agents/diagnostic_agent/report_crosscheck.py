"""
Cross-check against report findings.

findings.csv (StudyInstanceUID, Abnormality, Severity) is the output of extract_knee_findings.py, which reads each
radiology report with MedGemma. If the study being scored is in that file, its report findings are compared with the
image model's probabilities and the result goes into the agent's trace. It never changes the probabilities or the
confidence, so the numbers stay purely image-based.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Same words as extract_knee_findings.py uses to recognise a "normal" severity.
_NORMAL_WORDS = ("normal", "intact", "healthy", "unremarkable", "no abnormal", "absent",
                 "none", "negative", "no evidence", "preserved", "not seen", "no ")


def _is_normal(severity: str) -> bool:
    s = str(severity).lower()
    return any(w in s for w in _NORMAL_WORDS)


class ReportCrossCheck:
    def __init__(self, findings_csv, finding_names: list[str]):
        self.path = Path(findings_csv) if findings_csv else None
        self.names = set(finding_names)
        self._index: dict[str, list[tuple[str, str]]] | None = None

    def _load(self) -> dict[str, list[tuple[str, str]]]:
        if self._index is None:
            self._index = {}
            if self.path and self.path.is_file():
                df = pd.read_csv(self.path)
                for uid, g in df.groupby("StudyInstanceUID"):
                    self._index[str(uid)] = [(r.Abnormality, str(r.Severity)) for r in g.itertuples()
                                             if r.Abnormality in self.names]
            else:
                log.info("report cross-check disabled: findings.csv not found at %s", self.path)
        return self._index

    def trace_lines(self, study_id: str, findings: dict[str, float], threshold: float) -> list[str]:
        """Trace lines describing where the report and the image model agree or differ. [] if the study is not in the file."""
        rows = self._load().get(study_id)
        if rows is None:
            return []
        if not rows:
            return ["Report cross-check: report lists no abnormal findings among the tracked structures."]
        lines, reported = [], set()
        for name, severity in rows:
            if _is_normal(severity):
                continue
            reported.add(name)
            p = findings[name]
            lines.append(f"Report cross-check: {name} '{severity}'; image model {p:.2f} "
                         f"({'agrees' if p >= threshold else 'does not reach the threshold'}).")
        extra = [n for n, p in findings.items() if p >= threshold and n not in reported]
        if extra:
            lines.append("Report cross-check: image model flags " + ", ".join(extra) + " which the report does not list.")
        return lines or ["Report cross-check: report describes the tracked structures as normal."]
