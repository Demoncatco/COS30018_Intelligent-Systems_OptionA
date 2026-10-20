"""Merges gold labels (train.csv) and Person 2's weak labels into one table for training.

Finding cells are 1 (present), 0 (absent) or NaN (unknown, masked out of the loss).
Weak file: study_id + the 12 FINDINGS columns; cells 1, 0, -1 (hedged) or blank. -1 and blank become unknown.
Check a weak file: python -m model_service.knee_model.labels check --weak <file>
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from shared.schema import FINDINGS

GOLD_WEIGHT = 1.0
WEAK_WEIGHT = 0.5

# Any grade above this counts as present.
POSITIVE_ABOVE = 0.0

# Allowed cell values in the weak label file. -1 (hedged) becomes unknown.
WEAK_PRESENT, WEAK_ABSENT, WEAK_UNCERTAIN = 1, 0, -1
WEAK_ALLOWED_VALUES = {WEAK_PRESENT, WEAK_ABSENT, WEAK_UNCERTAIN}

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.csv"


def _binarise(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[FINDINGS].apply(pd.to_numeric, errors="coerce")
    return values.where(values.isna(), (values > POSITIVE_ABOVE).astype(float))


# --- Gold ---

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


# --- Weak ---

def validate_weak_file(weak: pd.DataFrame) -> pd.DataFrame:
    """Checks a weak label file against the agreed format and returns study_id + 12 findings as 1.0 / 0.0 / NaN."""
    weak = weak.copy()
    if "study_id" not in weak.columns and "StudyInstanceUID" in weak.columns:
        weak = weak.rename(columns={"StudyInstanceUID": "study_id"})
    if "study_id" not in weak.columns:
        raise ValueError("the weak label file needs a 'study_id' (or 'StudyInstanceUID') column")

    missing = [c for c in FINDINGS if c not in weak.columns]
    if missing:
        close = [c for c in weak.columns if c.lower() in {f.lower() for f in missing}]
        hint = f" (check spelling/case: found {close})" if close else ""
        raise ValueError(f"the weak label file is missing finding columns: {missing}{hint}")

    weak["study_id"] = weak["study_id"].astype(str).str.strip()
    if (weak["study_id"] == "").any() or weak["study_id"].isin(["nan", "None"]).any():
        raise ValueError("the weak label file has a row with an empty study_id")
    if not weak["study_id"].is_unique:
        dupes = sorted(weak.loc[weak["study_id"].duplicated(), "study_id"].unique())[:5]
        raise ValueError(f"the weak label file lists a study more than once, e.g. {dupes}")

    values = weak[FINDINGS].apply(pd.to_numeric, errors="coerce")
    # Anything that is not blank and not one of 1 / 0 / -1 is a mistake, e.g. "yes", 2 or a typo.
    raw_filled = weak[FINDINGS].notna() & weak[FINDINGS].astype(str).apply(lambda col: col.str.strip() != "")
    invalid = raw_filled & ~values.isin(WEAK_ALLOWED_VALUES)
    if invalid.any().any():
        examples = []
        for row, col in zip(*np.nonzero(invalid.to_numpy())):
            examples.append(f"{weak['study_id'].iloc[row]} / {FINDINGS[col]} = {weak[FINDINGS[col]].iloc[row]!r}")
            if len(examples) == 5:
                break
        raise ValueError(f"weak label cells must be 1, 0, -1 or blank; {int(invalid.sum().sum())} are not, "
                         f"e.g. {examples}")

    cleaned = values.where(values != WEAK_UNCERTAIN)  # hedged -> unknown
    cleaned.insert(0, "study_id", weak["study_id"].to_numpy())
    return cleaned.reset_index(drop=True)


def weak_labels(weak: pd.DataFrame, manifest: pd.DataFrame, weight: float = WEAK_WEIGHT) -> pd.DataFrame:
    if not 0.0 < weight <= GOLD_WEIGHT:
        raise ValueError(f"weak weight must be in (0, {GOLD_WEIGHT}], got {weight}")
    cleaned = validate_weak_file(weak)

    weak_ids = set(manifest.loc[manifest["label_source"] == "weak", "study_id"])
    gold_ids = set(manifest.loc[manifest["label_source"] == "gold", "study_id"])
    in_gold = cleaned["study_id"].isin(gold_ids)
    unknown = ~cleaned["study_id"].isin(weak_ids) & ~in_gold
    if in_gold.any():
        warnings.warn(f"{int(in_gold.sum())} weak rows are gold studies; the gold labels are kept", stacklevel=2)
    if unknown.any():
        warnings.warn(f"{int(unknown.sum())} weak rows are not in the manifest and are dropped", stacklevel=2)

    table = cleaned[cleaned["study_id"].isin(weak_ids)].copy()
    table.insert(1, "label_source", "weak")
    table.insert(2, "sample_weight", float(weight))
    # Drop weak studies with no known labels.
    return table[table[FINDINGS].notna().any(axis=1)].reset_index(drop=True)


def build_label_table(
    train: pd.DataFrame, manifest: pd.DataFrame, weak: pd.DataFrame | None = None, weak_weight: float = WEAK_WEIGHT
) -> pd.DataFrame:
    # Gold-only when no weak file is given (the v0 run).
    parts = [gold_labels(train, manifest)]
    if weak is not None:
        parts.append(weak_labels(weak, manifest, weight=weak_weight))
    table = pd.concat(parts, ignore_index=True)
    if not table["study_id"].is_unique:
        raise ValueError("a study appears in both the gold and weak labels")
    return table


def weak_label_template(manifest: pd.DataFrame) -> pd.DataFrame:
    """Empty file in the agreed format, one row per weak study in the manifest."""
    ids = manifest.loc[manifest["label_source"] == "weak", "study_id"].tolist()
    template = pd.DataFrame({"study_id": ids})
    for name in FINDINGS:
        template[name] = pd.Series([pd.NA] * len(ids), dtype="Int64")
    return template


# --- Summaries ---

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


# --- CLI ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Weak label template and checker.")
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("template", help="write an empty weak label file for the manifest's weak studies")
    template.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    template.add_argument("--out", type=Path, required=True)

    check = sub.add_parser("check", help="validate a weak label file and print per-finding counts")
    check.add_argument("--weak", type=Path, required=True)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.command == "template":
        weak_label_template(manifest).to_csv(args.out, index=False)
        print(f"wrote {args.out} ({int((manifest['label_source'] == 'weak').sum())} weak studies)")
        return

    table = weak_labels(pd.read_csv(args.weak), manifest)
    n_weak = int((manifest["label_source"] == "weak").sum())
    print(f"OK: {len(table)} of {n_weak} manifest weak studies have at least one known label")
    if len(table):  # an empty (e.g. freshly made template) file has no counts to show
        print(positive_counts(table).drop(columns="label_source").to_string(index=False))


if __name__ == "__main__":
    main()