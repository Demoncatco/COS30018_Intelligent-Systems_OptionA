# This file makes the selection on which MRI series for each of the three planes should be used for a Study.
# The model only requires one series per plane, however most studies have up to 2-3 series for each plane, therefore a decision needs to be made on which series for each plane should be chosen.
# The choice is made from train_series.csv and does not open / view any DICOM files to make the decisions. The decision is based on a fallback ladder, ideally each series wants to meet level 1, however there are three possible levels:
# Level 1: A fluid-sensitive series in that plane. Fluid shows up bright on these, and most injuries are fluid based, therefore this is the ideal series that gets selected
# Level 2: If there are no fluid-sensitive series for that plane, any series will have to do for that plane
# Level 3: The plane is missing entirely from that study, therefore there are no series for that plane. This should be made aware that the review is based off of only two planes.

# If there are multiple level 1 series, there needs to be a tie-break as only one series can be selected. If this occurs, then take the lowest SeriesInstanceUID.


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.schema import PLANES

# The three fallback levels, named so the code below reads as words rather than bare numbers.
LEVEL1_FLUID_SENSITIVE = 1
LEVEL2_ANY_SERIES = 2
LEVEL3_PLANE_MISSING = 3

# Paths are established to this file so the script works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_CSV = REPO_ROOT / "data" / "cache" / "kaggle_meta" / "train_series.csv"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "series_selection.csv"

@dataclass
class SelectedSeries:
    # Dataclass for the series that was chosen for one plane of one study, providing information on how it was chosen.
    plane: str  # either sagittal, coronal or axial
    series_id: str | None  # The SeriesInstanceUID that was picked. If the plane is missing from the Study then None.
    level: int  # Which fallback level was used?
    n_candidates: int  # How many series were on that fallback level. If there are more than 1 then it means that the tie-break function below made the choice

# Picks one series when a level has more than one for a plane.
def _tie_break(candidates: pd.DataFrame) -> str:
    # min() on text gives the lowest UID. If there is only one candidate then it just returns that one.
    return min(candidates["SeriesInstanceUID"])

# Picks the series for one plane.
# study_rows is the rows of one study only, whilst plane is either sagittal, coronal or axial.
def _select_series_per_plane(study_rows: pd.DataFrame, plane: str) -> SelectedSeries:

    # Keeps only the rows that are in this plane. Firstly selects the plane and then keeps only the study rows in that plane.
    plane_names = study_rows["Anatomical_Plane"].str.lower() # CSV writes the planes with capital letter, whilst the shared schema uses lowercase. This is then converted to lowercase to match shared schema.
    in_plane = study_rows[plane_names == plane]

    # Keeps only the plane filtered study rows that are fluid-sensitive. If fluid sensitive then the column holds 1 for yes and 0 for no.
    fluid_sensitive = in_plane[in_plane["Fluid_Sensitive"] == 1]

    # Works down the fallback levels and stops at the first one that has at least one series.
    # Ideally this is always Level 1 (fluid_sensitive)
    if len(fluid_sensitive) > 0:
        level = LEVEL1_FLUID_SENSITIVE
        candidates = fluid_sensitive
    elif len(in_plane) > 0:
        level = LEVEL2_ANY_SERIES
        candidates = in_plane
    else:
        # If the series gets to Level 3, then there is nothing to pick and therefore the series is None. This is provided back to however called this to then decide what to do.
        return SelectedSeries(plane=plane, series_id=None, level=LEVEL3_PLANE_MISSING, n_candidates=0)

    # Implements the tie-break so that there is only open selected candidate for the series. If there was only one candidate from the start, this always picks the one candidate.
    series_id = _tie_break(candidates)
    return SelectedSeries(plane=plane, series_id=series_id, level=level, n_candidates=len(candidates))

# Filters to the study. If it is empty then it raises an error. This calls the above function _select_series_per_plane for each plane.
def select_series(study_id: str, series_table: pd.DataFrame) -> dict[str, SelectedSeries]:
    # Picks one series per plane for one study. It returns this in a dictionary that is keyed by the plane. The order of planes are always sagittal, coronal and then axial.

    # Keeps only the rows that belong to this study.
    study_rows = series_table[series_table["StudyInstanceUID"] == study_id]

    # A study that is not in the table does not have any data and must be stopped. An error is thrown here.
    if len(study_rows) == 0:
        raise KeyError(f"study {study_id} has no rows in the series table")

    # PLANES is in order of sagittal, coronal and then axial.
    selection = {}
    for plane in PLANES:
        selection[plane] = _select_series_per_plane(study_rows, plane)

    return selection


def series_path(dataset_root: str | Path, study_id: str, series_id: str, split_folder: str = "train_series") -> Path:
    # Builds the folder path of a chosen series. This is used by load_study to find the folder.
    return Path(dataset_root) / split_folder / study_id / series_id

# Turns the selection of one study into one row of a table.
# There are three columns for each plane, showing the series that was picked, the level used, and how many series were tied on that level.
def _selection_row(study_id: str, selection: dict[str, SelectedSeries]) -> dict:
    row = {"study_id": study_id}

    # worst_level is the highest level that any plane needed. This starts at 1 which is ideal, and is raised whenever a plane needed more.
    # This is the single per-study number that is used for the fallback_level column in the manifest.csv file.
    worst_level = LEVEL1_FLUID_SENSITIVE

    for plane in PLANES:
        chosen = selection[plane]
        row[f"{plane}_series"] = chosen.series_id
        row[f"{plane}_level"] = chosen.level
        row[f"{plane}_candidates"] = chosen.n_candidates
        # max() keeps whichever is higher. the worst level so far, or this plane's level.
        worst_level = max(worst_level, chosen.level)

    row["worst_level"] = worst_level
    return row

# Filters, loops over the studies and then makes the table.
def build_selection_table(study_ids: list[str], series_table: pd.DataFrame) -> pd.DataFrame:
    # This runs select_series over many studies and returns one row per study, in the same order that the study IDs were given.

    # This cuts down the table to the wanted studies, making each per-study lookup much faster than searching all rows.
    wanted = series_table[series_table["StudyInstanceUID"].isin(study_ids)]

    rows = []
    for study_id in study_ids:
        selection = select_series(study_id, wanted)
        rows.append(_selection_row(study_id, selection))

    return pd.DataFrame(rows)

# This loops over rows with one plain assert each.
def check_selection_table(table: pd.DataFrame, study_ids: list[str]) -> None:
    # The set of asset statements check that:
    # Every study asked for has exactly one row, in the correct order.
    # Every level is either 1, 2 or 3
    # A series is named whenever the level is 1 or 2. And is empty when the level is 3.

    assert table["study_id"].tolist() == list(study_ids), "Studies in the table do not match the studies that were asked for"

    allowed_levels = [LEVEL1_FLUID_SENSITIVE, LEVEL2_ANY_SERIES, LEVEL3_PLANE_MISSING]

    for plane in PLANES:
        levels = table[f"{plane}_level"].tolist()
        series_ids = table[f"{plane}_series"].tolist()

        # Goes through the table one row at a time. zip() hands over the level and the series of the same row together.
        for level, series_id in zip(levels, series_ids):
            assert level in allowed_levels, f"Unexpected level value in {plane}"

            has_series = not pd.isna(series_id)
            plane_missing = level == LEVEL3_PLANE_MISSING
            # Exactly one of the two must be true. A row has a series, or its plane is missing, never both.
            assert has_series != plane_missing, f"A {plane} series and its level do not agree"


def _count_rows(table: pd.DataFrame, column: str, value: int) -> int:
    # Counts how many rows of the table have this value in this column.
    # The comparison gives True or False per row, and sum() counts the True's because True counts as 1.
    return int((table[column] == value).sum())


def _print_summary(table: pd.DataFrame, series_table: pd.DataFrame, out_path: Path) -> None:
    # Prints what was read, what was picked and where it was saved, so a run can be checked by eye.

    # There are three levels only because these two columns are the same.
    flags_identical = bool((series_table["Fluid_Sensitive"] == series_table["Fat_Suppression"]).all())

    print(f"series rows read: {len(series_table)}")
    print(f"fluid == fat column: {flags_identical}")
    print(f"studies selected: {len(table)}")

    # One line per plane: how many studies landed on each level, and how many were decided by the tie-break.
    for plane in PLANES:
        level_1 = _count_rows(table, f"{plane}_level", LEVEL1_FLUID_SENSITIVE)
        level_2 = _count_rows(table, f"{plane}_level", LEVEL2_ANY_SERIES)
        level_3 = _count_rows(table, f"{plane}_level", LEVEL3_PLANE_MISSING)
        # More than one candidate means the tie-break made the choice.
        tied = int((table[f"{plane}_candidates"] > 1).sum())
        print(f"{plane}: level 1 = {level_1}, level 2 = {level_2}, level 3 = {level_3}, decided by tie-break = {tied}")

    print(f"ideal in all planes: {_count_rows(table, 'worst_level', LEVEL1_FLUID_SENSITIVE)}")
    print(f"needed a fallback: {_count_rows(table, 'worst_level', LEVEL2_ANY_SERIES)}")
    print(f"missing a plane: {_count_rows(table, 'worst_level', LEVEL3_PLANE_MISSING)}")

    # The path is shown from the repository root when the file is inside the repository, and in full when it is not.
    out_path = out_path.resolve()
    if out_path.is_relative_to(REPO_ROOT):
        out_path = out_path.relative_to(REPO_ROOT)
    print(f"written to          : {out_path}")


# This main function defines the command-line options, builds the table for every study in the manifest and saves it.
def main() -> None:
    parser = argparse.ArgumentParser(description="Pick one series per plane for every study in the manifest.")
    parser.add_argument("--series-csv", type=Path, default=DEFAULT_SERIES_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Reads the two input files. The manifest gives the list of studies, the series table gives the series of every study.
    series_table = pd.read_csv(args.series_csv)
    manifest = pd.read_csv(args.manifest)
    study_ids = manifest["study_id"].tolist()

    # Builds the table and checks it before anything is written.
    table = build_selection_table(study_ids, series_table)
    check_selection_table(table, study_ids)

    table.to_csv(args.out, index=False, lineterminator="\n")

    _print_summary(table, series_table, args.out)


if __name__ == "__main__":
    main()
