"""
Builds the per-study label table from manifest.csv, train.csv and (for weak studies) findings.csv.

Output columns: study_id, label_source ('gold'|'weak'), then the 12 label columns as 0/1 ints.
"""

import pandas as pd

from config import ABNORMALITIES, is_normal


def load_gold_labels(manifest: pd.DataFrame, train_csv) -> pd.DataFrame:
    """Gold studies: the 12 label columns of train.csv, used as-is."""
    train = pd.read_csv(train_csv).rename(columns={"StudyInstanceUID": "study_id"})
    gold = manifest[manifest.label_source == "gold"].merge(
        train[["study_id"] + ABNORMALITIES], on="study_id", how="left"
    )
    if gold[ABNORMALITIES].isna().any().any():
        bad = gold[gold[ABNORMALITIES].isna().any(axis=1)].study_id.tolist()
        raise ValueError(f"{len(bad)} gold studies have missing labels in train.csv, e.g. {bad[:3]}")
    return gold


def derive_weak_labels(manifest: pd.DataFrame, findings_csv) -> pd.DataFrame:
    """
    Weak studies: a label is 1 if the report mentions it with a non-normal severity.
    Normal-and-not-mentioned both become 0. 'Not mentioned = 0' is an assumption (radiologists
    mostly comment on what is wrong) and is the main source of noise in weak labels.
    """
    weak = manifest[manifest.label_source == "weak"].copy()
    if weak.empty:
        return weak.assign(**{a: pd.Series(dtype=int) for a in ABNORMALITIES})
    if findings_csv is None:
        raise ValueError("manifest has weak studies but no findings_csv was given")

    raw = pd.read_csv(findings_csv).rename(columns={"StudyInstanceUID": "study_id"})
    missing = set(weak.study_id) - set(raw.study_id)
    if missing:
        print(f"WARNING: {len(missing)} weak studies are not in {findings_csv}; all their labels are 0.")

    f = raw[raw.Abnormality.isin(ABNORMALITIES)]
    f = f[~f.Severity.fillna("").map(is_normal)]
    positives = set(zip(f.study_id, f.Abnormality))
    for a in ABNORMALITIES:
        weak[a] = [int((s, a) in positives) for s in weak.study_id]
    return weak


def build_label_table(manifest_csv, train_csv, findings_csv=None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    table = pd.concat(
        [load_gold_labels(manifest, train_csv), derive_weak_labels(manifest, findings_csv)],
        ignore_index=True,
    )
    table[ABNORMALITIES] = table[ABNORMALITIES].astype(int)
    return table[["study_id", "label_source"] + ABNORMALITIES]
