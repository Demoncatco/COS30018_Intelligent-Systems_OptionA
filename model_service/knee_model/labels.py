"""Merges gold labels (train.csv) and Person 2's weak labels into one table for training.

Finding cells are 1 (present), 0 (absent) or NaN (unknown, masked out of the loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.schema import FINDINGS

GOLD_WEIGHT = 1.0
WEAK_WEIGHT = 0.5

# Any grade above this counts as present.
POSITIVE_ABOVE = 0.0


def _binarise(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[FINDINGS].apply(pd.to_numeric, errors="coerce")
    return values.where(values.isna(), (values > POSITIVE_ABOVE).astype(float))


def gold_labels(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FINDINGS if c not in train.columns]
    if missing:
        raise ValueError(f"train.csv is missing finding columns: {missing}")

    gold_ids = manifest.loc[manifest["label_source"] == "gold", "study_id"]
    rows = train[train["StudyInstanceUID"].isin(gold_ids)].copy()
    if len(rows) != len(gold_ids):
        raise ValueError(f"manifest lists {len(gold_ids)} gold studies but train.csv has {len(rows)} of them")

    table = _binarise(rows)
    if table.isna().any().any():
        raise ValueError("a gold study has a blank finding; gold studies must have all 12 filled in")
    table.insert(0, "study_id", rows["StudyInstanceUID"].to_numpy())
    table.insert(1, "label_source", "gold")
    table.insert(2, "sample_weight", GOLD_WEIGHT)
    return table.reset_index(drop=True)


def weak_labels(weak: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if "study_id" not in weak.columns:
        raise ValueError("the weak label file needs a 'study_id' column")
    missing = [c for c in FINDINGS if c not in weak.columns]
    if missing:
        raise ValueError(f"the weak label file is missing finding columns: {missing}")
    if not weak["study_id"].is_unique:
        raise ValueError("the weak label file lists a study more than once")

    weak_ids = set(manifest.loc[manifest["label_source"] == "weak", "study_id"])
    rows = weak[weak["study_id"].isin(weak_ids)].copy()
    table = _binarise(rows)
    table.insert(0, "study_id", rows["study_id"].to_numpy())
    table.insert(1, "label_source", "weak")
    table.insert(2, "sample_weight", WEAK_WEIGHT)
    # Drop weak studies with no known labels.
    return table[table[FINDINGS].notna().any(axis=1)].reset_index(drop=True)


def build_label_table(
    train: pd.DataFrame, manifest: pd.DataFrame, weak: pd.DataFrame | None = None
) -> pd.DataFrame:
    # Gold-only when no weak file is given (the v0 run).
    parts = [gold_labels(train, manifest)]
    if weak is not None:
        parts.append(weak_labels(weak, manifest))
    table = pd.concat(parts, ignore_index=True)
    assert table["study_id"].is_unique, "a study appears in both the gold and weak labels"
    return table


def positive_counts(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, part in table.groupby("label_source"):
        for name in FINDINGS:
            known = part[name].dropna()
            rows.append({"label_source": source, "finding": name, "positives": int((known == 1).sum()),
                         "negatives": int((known == 0).sum()), "unknown": int(part[name].isna().sum())})
    return pd.DataFrame(rows)


def label_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = table[FINDINGS].to_numpy(dtype=np.float32)
    mask = (~np.isnan(values)).astype(np.float32)
    return np.nan_to_num(values, nan=0.0), mask