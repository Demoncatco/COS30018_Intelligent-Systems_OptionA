import numpy as np
import pandas as pd
import pytest

from model_service.knee_model.labels import build_label_table, label_arrays
from shared.schema import FINDINGS


def _inputs():
    train = pd.DataFrame({"StudyInstanceUID": ["g1", "g2", "w1", "w2"]})
    for name in FINDINGS:
        train[name] = [2, 0, np.nan, np.nan]  # a grade of 2 must become 1
    manifest = pd.DataFrame({"study_id": ["g1", "g2", "w1", "w2"], "label_source": ["gold", "gold", "weak", "weak"]})
    weak = pd.DataFrame({"study_id": ["w1", "w2"], **{name: [1, np.nan] for name in FINDINGS}})
    return train, manifest, weak


def test_gold_only_when_no_weak_file():
    train, manifest, _ = _inputs()
    table = build_label_table(train, manifest)
    assert list(table["study_id"]) == ["g1", "g2"]
    assert table.loc[0, FINDINGS].tolist() == [1.0] * 12 and table.loc[1, FINDINGS].tolist() == [0.0] * 12


def test_weak_rows_get_lower_weight_and_empty_rows_are_dropped():
    train, manifest, weak = _inputs()
    table = build_label_table(train, manifest, weak)
    assert list(table["study_id"]) == ["g1", "g2", "w1"]  # w2 has no known labels at all
    assert table.set_index("study_id").loc["w1", "sample_weight"] < table.set_index("study_id").loc["g1", "sample_weight"]


def test_unknown_labels_are_masked():
    labels, mask = label_arrays(pd.DataFrame({name: [1.0, np.nan] for name in FINDINGS}))
    assert mask[1].sum() == 0 and labels[1].sum() == 0 and mask[0].sum() == 12


def test_blank_gold_cell_is_rejected():
    train, manifest, _ = _inputs()
    train.loc[0, "ACL"] = np.nan
    with pytest.raises(ValueError):
        build_label_table(train, manifest)