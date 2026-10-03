# Test Program that checks that the manifest file exists, has exactly three columns, no duplicate ID's, only gold or weak labels, 58 gold, 250 weak, 308 total, and that the 12 finding columns are identical to expected.

from pathlib import Path

import pandas as pd

from model_service.knee_model.make_manifest import FINDING_COLUMNS
from shared.schema import FINDINGS

MANIFEST = Path(__file__).resolve().parents[1] / "knee_model" / "manifest.csv"

# Checks Manifest File exists
def test_manifest_exists() -> None:
    assert MANIFEST.exists(), "run: python -m model_service.knee_model.make_manifest"

# Checks that the file has exactly three columns named correctly, each study_id is unique and there are no duplicates, and that the only labels that exist are either gold or weak.
# The third column, fallback_level, was added in version 2 of the manifest. The first two are unchanged from version 1.
def test_manifest_contract() -> None:
    m = pd.read_csv(MANIFEST)
    assert list(m.columns) == ["study_id", "label_source", "fallback_level"]
    assert m["study_id"].is_unique
    assert set(m["label_source"]) == {"gold", "weak"}

# Checks that the manifest count is correct. This is based of the weak sample size decided in team meeting on 15th September to be 250 weak studies.
def test_manifest_counts() -> None:
    m = pd.read_csv(MANIFEST)
    counts = m["label_source"].value_counts()
    assert counts["gold"] == 58
    assert counts["weak"] == 250
    assert len(m) == 308

# Checks the fallback level column. No study in the working subset is missing a plane, so the only values are 1 and 2.
# 274 studies had a fluid-sensitive series in all three planes, and 34 needed the fallback in one plane.
def test_manifest_fallback_level_counts() -> None:
    m = pd.read_csv(MANIFEST)
    counts = m["fallback_level"].value_counts()
    assert set(m["fallback_level"]) == {1, 2}
    assert counts[1] == 274
    assert counts[2] == 34

# The fallback level is the same number as worst_level in series_selection.csv, which holds the full per plane detail.
# The two files are written by different scripts, so this checks that they agree study by study and neither has gone stale.
def test_manifest_fallback_level_matches_series_selection() -> None:
    m = pd.read_csv(MANIFEST)
    selection = pd.read_csv(MANIFEST.parent / "series_selection.csv")
    assert m["study_id"].tolist() == selection["study_id"].tolist()
    assert m["fallback_level"].tolist() == selection["worst_level"].tolist()

# Tests that the 12 fidning columns match the schema that we set up.
def test_finding_columns_match_schema() -> None:
    assert FINDING_COLUMNS == FINDINGS
