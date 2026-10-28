"""Evaluation helpers: per-label AUC that copes with labels that have only one class present."""

import numpy as np
from sklearn.metrics import roc_auc_score

from config import ABNORMALITIES


def per_label_auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Returns {label: (auc or None, n_positive)}. AUC is None if the split lacks positives or negatives."""
    out = {}
    for j, name in enumerate(ABNORMALITIES):
        pos = int(y_true[:, j].sum())
        ok = 0 < pos < len(y_true)
        out[name] = (float(roc_auc_score(y_true[:, j], y_prob[:, j])) if ok else None, pos)
    return out


def mean_auc(per_label: dict) -> float:
    vals = [a for a, _ in per_label.values() if a is not None]
    return float(np.mean(vals)) if vals else float("nan")


def format_report(per_label: dict) -> str:
    lines = [f"  {name:<17} AUC {('%.3f' % a) if a is not None else '  n/a'}   (positives: {n})"
             for name, (a, n) in per_label.items()]
    lines.append(f"  {'MEAN':<17} AUC {mean_auc(per_label):.3f}")
    return "\n".join(lines)
