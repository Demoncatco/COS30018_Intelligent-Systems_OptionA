"""Build the Role 1 working-subset manifest: which studies are in scope, and why.

Run from the repository root:

    python -m model_service.knee_model.make_manifest

Reads  data/cache/kaggle_meta/train.csv          (gitignored, cached from Kaggle)
Writes model_service/knee_model/manifest.csv     (tracked by git — teammates read this)

The manifest has two columns:

    study_id       the StudyInstanceUID, which is also the folder name on disk
    label_source   'gold'  = all 12 finding columns are filled in train.csv
                   'weak'  = no findings filled; labels must be extracted from Report text

Why it exists: Person 2 cannot begin labelling and Person 3 cannot scope the
training set until the list of studies is fixed. This script makes that list
reproducible — same CSV, same seed, same manifest, every time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Decisions. Change these deliberately and record the change in the worklog.
# ---------------------------------------------------------------------------

SEED = 30018            # The unit code. Chosen so nobody has to look it up.
N_WEAK = 250            # Per the step-by-step plan: ~250 weak studies alongside the gold set.

# Paths are anchored to this file so the script works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train.csv"
DEFAULT_SERIES_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train_series.csv"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.csv"

# The 12 finding columns, in the exact order they appear in train.csv.
# This deliberately matches shared.schema.FINDINGS so the two can be asserted equal.
FINDING_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def build_manifest(train: pd.DataFrame, seed: int = SEED, n_weak: int = N_WEAK) -> pd.DataFrame:
    """Split train.csv into gold and weak studies and return the manifest table.

    Gold  = rows where any finding column is non-empty. In practice every gold row
            has all 12 filled, and that is asserted below rather than assumed.
    Weak  = a seeded random sample, without replacement, from every other row.

    The sample is drawn from the candidate IDs *sorted alphabetically*, not in CSV
    row order. This means the result depends only on the set of study IDs and the
    seed, not on how Kaggle happened to order the file.
    """
    missing = [c for c in FINDING_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"train.csv is missing finding columns: {missing}")

    has_any_label = train[FINDING_COLUMNS].notna().any(axis=1)
    has_all_labels = train[FINDING_COLUMNS].notna().all(axis=1)

    # A partially labelled row would be neither gold nor weak. None exist today;
    # fail loudly if that ever changes rather than silently misfiling it.
    partial = has_any_label & ~has_all_labels
    if partial.any():
        raise ValueError(f"{int(partial.sum())} rows are partially labelled; decide how to treat them")

    gold_ids = sorted(train.loc[has_any_label, "StudyInstanceUID"])
    candidate_ids = sorted(train.loc[~has_any_label, "StudyInstanceUID"])

    if n_weak > len(candidate_ids):
        raise ValueError(f"asked for {n_weak} weak studies but only {len(candidate_ids)} are unlabelled")

    # default_rng is numpy's modern generator. Its output for a given seed is stable
    # across numpy versions, which is the whole point of writing the seed down.
    rng = np.random.default_rng(seed)
    weak_ids = sorted(rng.choice(candidate_ids, size=n_weak, replace=False).tolist())

    manifest = pd.DataFrame(
        {
            "study_id": gold_ids + weak_ids,
            "label_source": ["gold"] * len(gold_ids) + ["weak"] * len(weak_ids),
        }
    )
    return manifest


def check_manifest(manifest: pd.DataFrame, train: pd.DataFrame, series: pd.DataFrame | None) -> None:
    """Assertions that must hold before the manifest is written. Cheap, and they
    catch the failure modes that would otherwise surface weeks later in training."""
    assert list(manifest.columns) == ["study_id", "label_source"], "column contract changed"
    assert manifest["study_id"].is_unique, "a study appears twice"
    assert set(manifest["label_source"]) <= {"gold", "weak"}, "unknown label_source value"
    assert set(manifest["study_id"]) <= set(train["StudyInstanceUID"]), "study id not in train.csv"
    if series is not None:
        # Every study must have at least one series row, or Step 4 has nothing to select from.
        assert set(manifest["study_id"]) <= set(series["StudyInstanceUID"]), "study id has no series rows"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--series-csv", type=Path, default=DEFAULT_SERIES_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-weak", type=int, default=N_WEAK)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    series = pd.read_csv(args.series_csv) if args.series_csv.exists() else None

    manifest = build_manifest(train, seed=args.seed, n_weak=args.n_weak)
    check_manifest(manifest, train, series)

    # lineterminator is pinned so the file is byte-identical on macOS, Linux and Windows.
    # A manifest that differs only in line endings would show as a git diff for no reason.
    manifest.to_csv(args.out, index=False, lineterminator="\n")

    counts = manifest["label_source"].value_counts()
    print(f"train.csv rows      : {len(train)}")
    print(f"gold studies        : {counts.get('gold', 0)}")
    print(f"weak studies        : {counts.get('weak', 0)}  (seed={args.seed}, n_weak={args.n_weak})")
    print(f"manifest rows       : {len(manifest)}")
    print(f"written to          : {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
