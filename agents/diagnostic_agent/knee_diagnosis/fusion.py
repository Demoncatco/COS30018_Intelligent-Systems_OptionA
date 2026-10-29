"""Puts the image model's probabilities next to the report's findings, one row per label."""

from __future__ import annotations

from .config import ABNORMALITIES, is_normal


def fuse(probabilities: dict[str, float], report_findings: list[dict] | None, threshold: float = 0.5) -> list[dict]:
    """
    agreement values:
      confirmed   image positive and report abnormal
      conflict    image positive but the report calls the structure normal
      image_only  image positive, report silent (or no report supplied)
      report_only report abnormal but image below threshold
      normal      report calls it normal and image is below threshold
      no_finding  neither mentions it
    """
    report = {}
    for f in report_findings or []:
        report[f["abnormality"]] = f["severity"]

    rows = []
    for a in ABNORMALITIES:
        p = probabilities.get(a)
        img_pos = p is not None and p >= threshold
        sev = report.get(a)
        rep = None if sev is None else ("normal" if is_normal(sev) else "abnormal")
        if img_pos:
            agreement = {"abnormal": "confirmed", "normal": "conflict", None: "image_only"}[rep]
        else:
            agreement = {"abnormal": "report_only", "normal": "normal", None: "no_finding"}[rep]
        rows.append({"abnormality": a, "image_probability": p, "image_positive": img_pos,
                     "report_severity": sev, "report_state": rep, "agreement": agreement})
    return rows
