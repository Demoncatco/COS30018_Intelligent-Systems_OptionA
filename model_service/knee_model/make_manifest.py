# This file builds the maifest that holds the working subset of data. This makes the decision on which studies are in scope, and which ones are not, and why.
# To Run from the Repository Root, Run: python -m model_service.knee_model.make_manifest
# This file reads data/cache/kaggle_meta/train.csv. It then writes model_service/knee_model/maifest.csv

# The manifest.csv file has two columns:
# - study_id: The StudyInstanceUID
# - label_source: Either gold or weak. Gold means that all 12 finding columns are filled in train.csv. Weak means that no findings are filled and the labels must be extracted from the reports.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 30018            # The Seed that is used to randomly obtain the same StudyInstanceUID's each time for random consistency. The unit code was chosen to be used as the Seed, though can be anything.
N_WEAK = 250            # The number of weak studies that will be included on top of the gold studies. This amount was discussed in our team meeting on the 15th September

# Paths are established to this file so the script works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train.csv" # Builds path to train.csv
DEFAULT_SERIES_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train_series.csv" # Builds path to train_series.csv
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.csv" # Builds path to manifest.csv

# The 12 finding columns, in the exact order they appear in train.csv.
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

# This function splits train.csv into gold and weak studies, returning the manifest table
def build_manifest(train: pd.DataFrame, seed: int = SEED, n_weak: int = N_WEAK) -> pd.DataFrame:
    # Takes the train table, plus a seed and sample size and returns the manifest table. 
    
    # Gold = rows where any finding columns are NOT empty.
    # Weak = a seeded random sample study that does not have any findings details
    
    # Checks that all 12 findings columns actually exist in the csv.
    missing = [c for c in FINDING_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"train.csv is missing finding columns: {missing}")

    has_any_label = train[FINDING_COLUMNS].notna().any(axis=1) # Turns a true or false for whether at least one of the finding column cells are filled in per row
    has_all_labels = train[FINDING_COLUMNS].notna().all(axis=1) # Returns a true or false for whether all finding column cells are filled in per row

    # A row that has some but not all findings would be neither gold or weak. This ensures that none exist by throwing an error.
    partial = has_any_label & ~has_all_labels
    if partial.any():
        raise ValueError(f"{int(partial.sum())} rows are partially labelled; decide how to treat them")

    # .loc[mask, column] selects the study IDs where the mask is true.
    gold_ids = sorted(train.loc[has_any_label, "StudyInstanceUID"]) # gold_ids hold every study that has all findings columns labelled
    candidate_ids = sorted(train.loc[~has_any_label, "StudyInstanceUID"]) # candidate_ids hold every study that has no findings columns labelled. They are then candidates for the weak studies. ~ means not, which is why it is used here.

    # Ensures that we cannot draw a sample size that is greater than the actual number of candidates. This error should never occur, however it is accounted for in case.
    if n_weak > len(candidate_ids):
        raise ValueError(f"asked for {n_weak} weak studies but only {len(candidate_ids)} are unlabelled")

    # Creates the random number generator and then picks 250 candidates without getting any duplicates.
    rng = np.random.default_rng(seed)
    weak_ids = sorted(rng.choice(candidate_ids, size=n_weak, replace=False).tolist()) # .tolist converts from numpy array to plain python list, sorting it again for a neat sorted file.

    # Builds the two column manifest file with the study IDs of the gold list followed by the weak list, and then the label_source (whether the study id is gold or weak)
    manifest = pd.DataFrame(
        {
            "study_id": gold_ids + weak_ids,
            "label_source": ["gold"] * len(gold_ids) + ["weak"] * len(weak_ids),
        }
    )
    return manifest

# This function checks the manifest file.
def check_manifest(manifest: pd.DataFrame, train: pd.DataFrame, series: pd.DataFrame | None) -> None:
    # Set of assert statements that checks:
    # - Exactly two columns are expected
    # - No Study appears twice
    # - Only label values are weak or gold
    # - Every ID exists in train.csv
    # If the series CSV is available, every ID has at least one MRI series to select from.

    assert list(manifest.columns) == ["study_id", "label_source"], "Columns expected have changed"
    assert manifest["study_id"].is_unique, "A study has appeared twice"
    assert set(manifest["label_source"]) <= {"gold", "weak"}, "A studies label source is not as expected"
    assert set(manifest["study_id"]) <= set(train["StudyInstanceUID"]), "Study ID cannot be found in train.csv"
    if series is not None:
        # Every study must have at least one series row.
        assert set(manifest["study_id"]) <= set(series["StudyInstanceUID"]), "Study ID does not have any Series rows."

# This main function defines the command-line options.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--series-csv", type=Path, default=DEFAULT_SERIES_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-weak", type=int, default=N_WEAK)
    args = parser.parse_args()

    # Loads the two csvs into tables. 
    train = pd.read_csv(args.train_csv)
    series = pd.read_csv(args.series_csv) if args.series_csv.exists() else None # Series table load is optional

    # Builds the manifest table and tehn verifies it
    manifest = build_manifest(train, seed=args.seed, n_weak=args.n_weak)
    check_manifest(manifest, train, series)

    # lineterminator is pinned so the file is byte-identical on macOS, Linux and Windows.
    # A manifest that differs only in line endings would show as a git diff for no reason.
    # Writes the file to csv. index=false stops pandas from adding unwanted row-number columns.
    manifest.to_csv(args.out, index=False)

    counts = manifest["label_source"].value_counts()
    print(f"train.csv rows      : {len(train)}")
    print(f"gold studies        : {counts.get('gold', 0)}")
    print(f"weak studies        : {counts.get('weak', 0)}  (seed={args.seed}, n_weak={args.n_weak})")
    print(f"manifest rows       : {len(manifest)}")
    print(f"written to          : {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
