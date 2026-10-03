# This file builds the manifest that is the working subset that will be used for training.
# The manifest.csv file has three columns:
# - study_id which is the StudyInstanceUID
# - label_source is whether the label is gold or weak. Gold meaning that all 12 finding columns are filled in train.csv, and weak meaning that no findings are filled and the labels must be extracted from the reports.
# - fallback_level is the highest fallback level that any of the three planes needed for that study:
    # Level 1 is that every plane has a fluid-sensitive series (THIS IS IDEAL)
    # Level 2 is that at least one plane had none and fell back to another series.
    # Level 3 means that at least one plane is missing and this is an issue.

# How the file is laid out, from the smallest job to the biggest:
# 1. _check_finding_columns - stops if train.csv does not have all 12 finding columns.
# 2. _split_by_labels       - sorts the studies into gold (all 12 findings filled) and candidates for weak (none filled).
# 3. _pick_weak_studies     - picks n_weak of the candidates at random, the same ones every run.
# 4. build_manifest         - runs the three above and puts gold and weak together in one two-column table.
# 5. add_fallback_level     - adds the third column, worked out from train_series.csv.
# 6. check_manifest         - checks the finished table before it is saved.
# 7. _print_summary         - prints the counts so a run can be checked by eye.
# 8. main                   - reads the two CSV files, runs the steps above, saves manifest.csv.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model_service.knee_model.series_selection import build_selection_table

SEED = 30018 # Seed that is used to randomly obtain the same StudyInstanceUID's
N_WEAK = 250 # The number of weak studies that was decided on in group meeting.

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train.csv"
DEFAULT_SERIES_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train_series.csv"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.csv"

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


def _check_finding_columns(train: pd.DataFrame) -> None:
    # Checks that all 12 finding columns actually exist in the csv. Returns nothing: it either raises an error or lets the table through.
    missing = []
    for column in FINDING_COLUMNS:
        if column not in train.columns:
            missing.append(column)
    if missing:
        raise ValueError(f"train.csv is missing finding columns: {missing}")


def _split_by_labels(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Sorts the studies in train.csv into two lists: the gold studies, and the candidates that a weak study can be picked from.
    # Gold = Row has ALL 12 Finding labels
    # Candidate = Row has NO finding labels at all
    has_any_label = train[FINDING_COLUMNS].notna().any(axis=1) # returns a true or false for whether at least one of the finding column cells are filled in per row
    has_all_labels = train[FINDING_COLUMNS].notna().all(axis=1) # Returns a true or false for whether all finding column cells are filled in per row

    # A row that has some but not all findings would be neither gold nor weak. This ensures that none exist by throwing an error.
    partial = has_any_label & ~has_all_labels
    if partial.any():
        raise ValueError(f"{int(partial.sum())} rows are partially labelled; decide how to treat them")

    # .loc[mask, column] selects the study IDs where the mask is true. sorted() puts them in ID order, so the result does not depend on the row order of the CSV.
    gold_ids = sorted(train.loc[has_all_labels, "StudyInstanceUID"]) # gold_ids hold every study that has all findings columns labelled
    candidate_ids = sorted(train.loc[~has_any_label, "StudyInstanceUID"]) # candidate_ids hold every study that has no findings columns labelled. They are then candidates for the weak studies. ~ means not, which is why it is used here.
    return gold_ids, candidate_ids


def _pick_weak_studies(candidate_ids: list[str], n_weak: int, seed: int) -> list[str]:
    # Picks n_weak studies out of the candidates at random. The seed makes the random generator give the same picks every run.

    # Ensures that we cannot draw a sample size that is greater than the actual number of candidates. This error should never occur, however it is accounted for in case.
    if n_weak > len(candidate_ids):
        raise ValueError(f"asked for {n_weak} weak studies but only {len(candidate_ids)} are unlabelled")

    # Creates the random number generator and then picks n_weak (currently set to 250) candidates without getting any duplicates.
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidate_ids, size=n_weak, replace=False)
    weak_ids = sorted(picked.tolist()) # .tolist converts from numpy array to plain python list, sorting it again for a neat sorted file.
    return weak_ids


# This function splits train.csv into gold and weak studies, returning the manifest table
def build_manifest(train: pd.DataFrame, seed: int = SEED, n_weak: int = N_WEAK) -> pd.DataFrame:
    # Takes the train table, plus a seed and sample size and returns the manifest table.
    _check_finding_columns(train)
    gold_ids, candidate_ids = _split_by_labels(train)
    weak_ids = _pick_weak_studies(candidate_ids, n_weak, seed)

    # Builds the two columns that will be used for the manifest file with the study IDs of the gold list followed by the weak list, and then the label_source (whether the study id is gold or weak).
    # ["gold"] * 58 is a list of the word gold repeated 58 times, one for each gold study, so the two columns are the same length.
    manifest = pd.DataFrame(
        {
            "study_id": gold_ids + weak_ids,
            "label_source": ["gold"] * len(gold_ids) + ["weak"] * len(weak_ids),
        }
    )
    return manifest


# This function adds the fallback_level column, which is what makes the manifest version 2.
def add_fallback_level(manifest: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    # Takes the two column manifest and the series table (train_series.csv), and returns the manifest with a third column.

    # The series selection from series_selection.py is run for every study in the manifest.
    # build_selection_table returns its rows in the same order the study IDs are given, so the column lines up row for row.
    selection = build_selection_table(manifest["study_id"].tolist(), series)
    manifest = manifest.copy() # A copy is changed so the table that was passed in is left as it was.
    manifest["fallback_level"] = selection["worst_level"].to_numpy()
    return manifest


# This function checks the manifest file.
def check_manifest(manifest: pd.DataFrame, train: pd.DataFrame, series: pd.DataFrame) -> None:
    # Set of assert statements that checks:
    # - Exactly three columns are expected
    # - No Study appears twice
    # - Only label values are weak or gold
    # - Only fallback levels are 1, 2 or 3
    # - Every ID exists in train.csv
    # - Every ID has at least one series row

    assert list(manifest.columns) == ["study_id", "label_source", "fallback_level"], "Columns expected have changed"
    assert manifest["study_id"].is_unique, "A study has appeared twice"
    assert set(manifest["label_source"]).issubset({"gold", "weak"}), "A studies label source is not as expected"
    assert set(manifest["fallback_level"]).issubset({1, 2, 3}), "A studies fallback level is not as expected"
    assert set(manifest["study_id"]).issubset(set(train["StudyInstanceUID"])), "Study ID cannot be found in train.csv"
    # Every study must have at least one series row.
    assert set(manifest["study_id"]).issubset(set(series["StudyInstanceUID"])), "Study ID does not have any Series rows."


def _print_summary(manifest: pd.DataFrame, train: pd.DataFrame, out_path: Path, seed: int, n_weak: int) -> None:
    # Prints what was read, what was picked and where it was saved, so a run can be checked by eye.

    # The path is shown from the repository root when the file is inside the repository, and in full when it is not.
    out_path = out_path.resolve()
    if out_path.is_relative_to(REPO_ROOT):
        out_path = out_path.relative_to(REPO_ROOT)

    counts = manifest["label_source"].value_counts()
    print(f"train.csv rows: {len(train)}")
    print(f"gold studies: {counts.get('gold', 0)}")
    print(f"weak studies: {counts.get('weak', 0)}  (seed={seed}, n_weak={n_weak})")
    print(f"manifest rows: {len(manifest)}")
    print(f"fallback level: {manifest['fallback_level'].value_counts().sort_index().to_dict()}")
    print(f"written to: {out_path}")


# This main function defines, loads, builds, checks, saves and prints the manifest.csv
def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest.csv, the working subset of gold and weak studies.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--series-csv", type=Path, default=DEFAULT_SERIES_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-weak", type=int, default=N_WEAK)
    args = parser.parse_args()

    # Reads the two input files, stopping with a clear message if either is not on this machine.
    if not args.train_csv.exists():
        raise FileNotFoundError(f"{args.train_csv} is needed to build the manifest")
    train = pd.read_csv(args.train_csv)
    # The series table used to be optional. It is needed now, because the fallback_level column is worked out from it.
    if not args.series_csv.exists():
        raise FileNotFoundError(f"{args.series_csv} is needed to work out the fallback level")
    series = pd.read_csv(args.series_csv)

    # Builds the manifest table, adds the fallback level, and then verifies it
    manifest = build_manifest(train, seed=args.seed, n_weak=args.n_weak)
    manifest = add_fallback_level(manifest, series)
    check_manifest(manifest, train, series)

    # lineterminator is pinned so the file is byte-identical on macOS, Linux and Windows.
    # A manifest that differs only in line endings would show as a git diff for no reason.
    # Writes the file to csv. index=false stops pandas from adding unwanted row-number columns.
    manifest.to_csv(args.out, index=False, lineterminator="\n")

    _print_summary(manifest, train, args.out, args.seed, args.n_weak)


if __name__ == "__main__":
    main()
