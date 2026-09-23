# Test Program that checks that the manifest file exists, has exactly two columns, no duplicate ID's, only gold or weak labels, 58 gold, 250 weak, 308 total, and that the 12 finding columns are identical to expected.

from pathlib import Path

import pandas as pd

from model_service.knee_model.make_manifest import FINDING_COLUMNS
from shared.schema import FINDINGS

MANIFEST = Path(__file__).resolve().parents[1] / "knee_model" / "manifest.csv"

# Checks Manifest File exists
def test_manifest_exists() -> None:
    assert MANIFEST.exists(), "run: python -m model_service.knee_model.make_manifest"

# Checks that the file has exactly two columns named correctly, each study_id is unique and there are no duplicates, and that the only labels that exist are either gold or weak.
def test_manifest_contract() -> None:
    m = pd.read_csv(MANIFEST)
    assert list(m.columns) == ["study_id", "label_source"]
    assert m["study_id"].is_unique
    assert set(m["label_source"]) == {"gold", "weak"}

# Checks that the manifest count is correct. This is based of the weak sample size decided in team meeting on 15th September to be 250 weak studies.
def test_manifest_counts() -> None:
    m = pd.read_csv(MANIFEST)
    counts = m["label_source"].value_counts()
    assert counts["gold"] == 58
    assert counts["weak"] == 250
    assert len(m) == 308

# Tests that the 12 fidning columns match the schema that we set up.
def test_finding_columns_match_schema() -> None:
    assert FINDING_COLUMNS == FINDINGS
