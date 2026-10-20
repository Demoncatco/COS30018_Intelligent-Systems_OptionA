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

# --- Weak label file format ---

def test_hedged_weak_label_becomes_unknown():
    train, manifest, weak = _inputs()
    weak.loc[0, "ACL"] = -1
    table = build_label_table(train, manifest, weak).set_index("study_id")
    assert np.isnan(table.loc["w1", "ACL"]) and table.loc["w1", "MCL"] == 1.0


def test_invalid_weak_value_is_rejected_with_its_location():
    train, manifest, weak = _inputs()
    weak["ACL"] = weak["ACL"].astype(object)
    weak.loc[0, "ACL"] = "yes"
    with pytest.raises(ValueError, match="w1 / ACL"):
        build_label_table(train, manifest, weak)


def test_weak_grade_above_one_is_rejected():
    train, manifest, weak = _inputs()
    weak.loc[0, "Fracture"] = 2
    with pytest.raises(ValueError, match="1, 0, -1 or blank"):
        build_label_table(train, manifest, weak)


def test_misspelt_finding_column_gives_a_hint():
    train, manifest, weak = _inputs()
    weak = weak.rename(columns={"Baker's": "Bakers"})
    with pytest.raises(ValueError, match="Baker's"):
        build_label_table(train, manifest, weak)


def test_studyinstanceuid_column_is_accepted():
    train, manifest, weak = _inputs()
    table = build_label_table(train, manifest, weak.rename(columns={"study_id": "StudyInstanceUID"}))
    assert "w1" in set(table["study_id"])


def test_weak_rows_outside_the_manifest_are_dropped_with_a_warning():
    train, manifest, weak = _inputs()
    extra = weak.iloc[[0]].assign(study_id="not-in-manifest")
    gold_copy = weak.iloc[[0]].assign(study_id="g1")
    with pytest.warns(UserWarning):
        table = build_label_table(train, manifest, pd.concat([weak, extra, gold_copy], ignore_index=True))
    assert list(table["study_id"]) == ["g1", "g2", "w1"]
    assert table.set_index("study_id").loc["g1", "label_source"] == "gold"


def test_weak_weight_is_configurable():
    train, manifest, weak = _inputs()
    table = build_label_table(train, manifest, weak, weak_weight=0.3).set_index("study_id")
    assert table.loc["w1", "sample_weight"] == 0.3 and table.loc["g1", "sample_weight"] == 1.0
    with pytest.raises(ValueError):
        build_label_table(train, manifest, weak, weak_weight=0.0)


def test_template_round_trips_through_the_checker(tmp_path):
    from model_service.knee_model.labels import weak_label_template, weak_labels

    _, manifest, _ = _inputs()
    path = tmp_path / "template.csv"
    weak_label_template(manifest).to_csv(path, index=False)
    template = pd.read_csv(path)
    assert list(template.columns) == ["study_id", *FINDINGS] and list(template["study_id"]) == ["w1", "w2"]
    assert len(weak_labels(template, manifest)) == 0  # valid but empty
    template.loc[0, "ACL"] = 1
    assert len(weak_labels(template, manifest)) == 1


def test_check_command_works_on_an_empty_and_a_filled_file(tmp_path, monkeypatch, capsys):
    import sys

    from model_service.knee_model.labels import main, weak_label_template

    _, manifest, _ = _inputs()
    manifest_path, weak_path = tmp_path / "manifest.csv", tmp_path / "weak.csv"
    manifest.to_csv(manifest_path, index=False)
    template = weak_label_template(manifest)
    template.to_csv(weak_path, index=False)
    args = ["labels", "check", "--weak", str(weak_path), "--manifest", str(manifest_path)]

    monkeypatch.setattr(sys, "argv", args)
    main()
    assert "OK: 0 of 2" in capsys.readouterr().out

    template.loc[0, "ACL"] = 1
    template.to_csv(weak_path, index=False)
    main()
    out = capsys.readouterr().out
    assert "OK: 1 of 2" in out and "ACL" in out