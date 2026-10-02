# Tests for the series selection, checking that the fallback levels and the tie-break pick the series they are meant to.
# The real train_series.csv is competition data and is not in the repository, so these tests build tiny fake series tables by hand.
# The last three tests read series_selection.csv, which is tracked by git, so they also run on a machine with no Kaggle data.

from pathlib import Path

import pandas as pd
import pytest

from model_service.knee_model.series_selection import (
    DEFAULT_SERIES_CSV,
    build_selection_table,
    select_series,
    series_path,
)
from shared.schema import PLANES

KNEE_MODEL = Path(__file__).resolve().parents[1] / "knee_model"
MANIFEST = KNEE_MODEL / "manifest.csv"
SELECTION = KNEE_MODEL / "series_selection.csv"


# Builds a fake series table. Each row is given as (study, series, fluid_sensitive, plane).
def _table(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"])
    # The two flag columns are identical in the real data, so the fake table copies one into the other.
    table["Fat_Suppression"] = table["Fluid_Sensitive"]
    return table


# A normal study: one fluid-sensitive series in every plane, plus one extra sagittal series that is not fluid-sensitive.
def _ideal_study() -> pd.DataFrame:
    return _table(
        [
            ("study1", "s.100", 0, "Sagittal"),
            ("study1", "s.200", 1, "Sagittal"),
            ("study1", "s.300", 1, "Coronal"),
            ("study1", "s.400", 1, "Axial"),
        ]
    )


# The non fluid-sensitive sagittal series has the lower UID. The fluid-sensitive one must still win, because the fallback level is decided before the tie-break.
def test_level_1_prefers_the_fluid_sensitive_series() -> None:
    selection = select_series("study1", _ideal_study())
    assert selection["sagittal"].series_id == "s.200"
    assert selection["sagittal"].level == 1
    assert selection["sagittal"].n_candidates == 1
    assert [selection[p].level for p in PLANES] == [1, 1, 1]


# Sagittal has no fluid-sensitive series, so it must fall back to the series that is there and say level 2. The other two planes must be unaffected.
def test_level_2_falls_back_to_any_series_in_the_plane() -> None:
    table = _table(
        [
            ("study1", "s.100", 0, "Sagittal"),
            ("study1", "s.300", 1, "Coronal"),
            ("study1", "s.400", 1, "Axial"),
        ]
    )
    selection = select_series("study1", table)
    assert selection["sagittal"].series_id == "s.100"
    assert selection["sagittal"].level == 2
    assert selection["coronal"].level == 1
    assert selection["axial"].level == 1


# The study has no axial series at all. Axial must come back as level 3 with no series, and it must not crash or stop the other planes being picked.
def test_level_3_reports_a_missing_plane_without_crashing() -> None:
    table = _table([("study1", "s.200", 1, "Sagittal"), ("study1", "s.300", 1, "Coronal")])
    selection = select_series("study1", table)
    assert selection["axial"].series_id is None
    assert selection["axial"].level == 3
    assert selection["axial"].n_candidates == 0
    assert selection["sagittal"].series_id == "s.200"
    assert selection["coronal"].series_id == "s.300"


# Three fluid-sensitive sagittal series. The lowest UID must be picked, and n_candidates must say that three were tied.
def test_tie_break_picks_the_lowest_series_uid() -> None:
    table = _table(
        [
            ("study1", "s.500", 1, "Sagittal"),
            ("study1", "s.200", 1, "Sagittal"),
            ("study1", "s.900", 1, "Sagittal"),
            ("study1", "s.300", 1, "Coronal"),
            ("study1", "s.400", 1, "Axial"),
        ]
    )
    selection = select_series("study1", table)
    assert selection["sagittal"].series_id == "s.200"
    assert selection["sagittal"].n_candidates == 3


# The same table with its rows in the opposite order must give exactly the same picks. This is the whole point of the tie-break:
# the answer depends on the UIDs, not on the order Kaggle happened to write the CSV in.
def test_tie_break_does_not_depend_on_row_order() -> None:
    table = _table(
        [
            ("study1", "s.500", 1, "Sagittal"),
            ("study1", "s.200", 1, "Sagittal"),
            ("study1", "s.700", 1, "Coronal"),
            ("study1", "s.300", 1, "Coronal"),
            ("study1", "s.400", 1, "Axial"),
        ]
    )
    reversed_table = table.iloc[::-1].reset_index(drop=True)
    assert select_series("study1", table) == select_series("study1", reversed_table)


# Two sagittal series and neither is fluid-sensitive. The tie-break must apply on level 2 as well, not only on level 1.
def test_tie_break_also_applies_on_level_2() -> None:
    table = _table(
        [
            ("study1", "s.800", 0, "Sagittal"),
            ("study1", "s.600", 0, "Sagittal"),
            ("study1", "s.300", 1, "Coronal"),
            ("study1", "s.400", 1, "Axial"),
        ]
    )
    selection = select_series("study1", table)
    assert selection["sagittal"].series_id == "s.600"
    assert selection["sagittal"].level == 2
    assert selection["sagittal"].n_candidates == 2


# The plane is written in different cases. All of them must be recognised.
def test_plane_names_are_matched_in_any_letter_case() -> None:
    table = _table([("study1", "s.200", 1, "SAGITTAL"), ("study1", "s.300", 1, "coronal"), ("study1", "s.400", 1, "Axial")])
    selection = select_series("study1", table)
    assert [selection[p].level for p in PLANES] == [1, 1, 1]


# A second study has lower UIDs in every plane. None of them may leak into the first study's selection.
def test_series_from_other_studies_are_ignored() -> None:
    other = _table([("study2", "s.001", 1, "Sagittal"), ("study2", "s.002", 1, "Coronal"), ("study2", "s.003", 1, "Axial")])
    table = pd.concat([other, _ideal_study()], ignore_index=True)
    selection = select_series("study1", table)
    assert [selection[p].series_id for p in PLANES] == ["s.200", "s.300", "s.400"]


# A study ID that is not in the table is a caller mistake. It must raise an error, not be reported as three missing planes.
def test_unknown_study_raises() -> None:
    with pytest.raises(KeyError):
        select_series("not-a-study", _ideal_study())


# The output contract fixes the plane order as sagittal, coronal, axial. The selection must come back in that order.
def test_selection_is_in_contract_plane_order() -> None:
    selection = select_series("study1", _ideal_study())
    assert tuple(selection) == PLANES == ("sagittal", "coronal", "axial")


# The folder path must be dataset root / split folder / study / series, which is the layout verified on Kaggle in Step 1.
def test_series_path_joins_the_uids_into_the_kaggle_layout() -> None:
    assert series_path("/root", "study1", "s.200") == Path("/root/train_series/study1/s.200")
    assert series_path("/root", "study1", "s.200", split_folder="test_series") == Path("/root/test_series/study1/s.200")


# Two studies, one ideal and one that needs a sagittal fallback. The table must have one row per study in the order given,
# three columns per plane, and worst_level equal to the highest level any plane used.
def test_selection_table_has_one_row_per_study_and_the_worst_level() -> None:
    fallback = _table([("study2", "s.110", 0, "Sagittal"), ("study2", "s.310", 1, "Coronal"), ("study2", "s.410", 1, "Axial")])
    table = build_selection_table(["study2", "study1"], pd.concat([_ideal_study(), fallback], ignore_index=True))
    assert table["study_id"].tolist() == ["study2", "study1"]
    assert table["worst_level"].tolist() == [2, 1]
    for plane in PLANES:
        assert {f"{plane}_series", f"{plane}_level", f"{plane}_candidates"} <= set(table.columns)


# The saved file must cover exactly the studies in the manifest, in the same order, so the two files line up row for row.
def test_saved_selection_matches_the_manifest() -> None:
    assert SELECTION.exists(), "run: python -m model_service.knee_model.series_selection"
    selection = pd.read_csv(SELECTION)
    manifest = pd.read_csv(MANIFEST)
    assert selection["study_id"].tolist() == manifest["study_id"].tolist()


# This is the "done when" of Step 4: all 308 studies return three series, each tagged with the level used.
# No plane is missing in the working subset, so every level must be 1 or 2 and every series must be named.
def test_saved_selection_has_three_series_for_every_study() -> None:
    selection = pd.read_csv(SELECTION)
    assert len(selection) == 308
    for plane in PLANES:
        assert selection[f"{plane}_series"].notna().all()
        assert set(selection[f"{plane}_level"]) <= {1, 2}
        assert (selection[f"{plane}_candidates"] >= 1).all()
    assert (selection["worst_level"] == selection[[f"{p}_level" for p in PLANES]].max(axis=1)).all()


# Guards against the saved file going stale. If the Kaggle metadata is on this machine, building the table again must
# give exactly what is saved. On a machine without the data (a teammate's laptop) the test is skipped, not failed.
def test_saved_selection_can_be_rebuilt_from_the_series_csv() -> None:
    if not DEFAULT_SERIES_CSV.exists():
        pytest.skip("train_series.csv is not on this machine")
    saved = pd.read_csv(SELECTION)
    rebuilt = build_selection_table(saved["study_id"].tolist(), pd.read_csv(DEFAULT_SERIES_CSV))
    pd.testing.assert_frame_equal(rebuilt, saved)
