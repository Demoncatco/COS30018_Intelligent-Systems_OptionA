import json

import pandas as pd

from model_service.knee_model.model import load_checkpoint
from model_service.knee_model.train import TrainConfig, make_split, train
from model_service.tests._fakes import label_table, write_cache
from shared.schema import FINDINGS


def _gold_val(split: pd.DataFrame) -> set[str]:
    return set(split.query("label_source == 'gold' and split == 'val'")["study_id"])


def test_split_is_repeatable_and_disjoint():
    table = label_table(n_gold=20, n_weak=10)
    a = make_split(table, n_val_gold=5, weak_val_fraction=0.2, seed=1)
    b = make_split(table, n_val_gold=5, weak_val_fraction=0.2, seed=1)
    assert a.equals(b)
    val = a[a["split"] == "val"]
    assert (val["label_source"] == "gold").sum() == 5 and (val["label_source"] == "weak").sum() == 2
    assert a["study_id"].is_unique


def test_smoke_training_run_writes_every_output(tmp_path):
    table = label_table(n_gold=12, n_weak=6)
    write_cache(tmp_path / "cache", table["study_id"].tolist(), missing_axial={"gold3"})
    config = TrainConfig(version="smoke", epochs=2, batch_size=4, freeze_epochs=1, n_val_gold=4,
                         pretrained=False, num_workers=0, patience=5)
    summary = train(config, table, tmp_path / "cache", tmp_path / "run")

    out = tmp_path / "run"
    for name in ["knee_smoke.pt", "splits.csv", "val_predictions.csv", "val_auc.csv", "label_counts.csv", "history.json"]:
        assert (out / name).exists(), name
    assert len(json.loads((out / "history.json").read_text())["history"]) == 2
    model, meta = load_checkpoint(out / "knee_smoke.pt")
    assert meta["n_slices"] == 4 and meta["image_size"] == 32 and meta["split_seed"] == config.seed
    val_ids = set(pd.read_csv(out / "splits.csv").query("split == 'val'")["study_id"])
    assert set(pd.read_csv(out / "val_predictions.csv")["study_id"]) == val_ids
    assert summary["best_epoch"] >= 1

def test_reused_split_keeps_gold_validation_and_assigns_new_weak():
    from model_service.knee_model.train import reuse_split

    gold_only = label_table(n_gold=20, n_weak=0)
    v0 = make_split(gold_only, n_val_gold=5, weak_val_fraction=0.2, seed=1)
    with_weak = label_table(n_gold=20, n_weak=10)
    v1 = reuse_split(v0, with_weak, weak_val_fraction=0.2, seed=99)

    assert _gold_val(v1) == _gold_val(v0)
    assert (v1.query("label_source == 'weak'")["split"] == "val").sum() == 2
    assert v1["split"].notna().all()


def test_reused_split_rejects_a_different_gold_set():
    import pytest

    from model_service.knee_model.train import reuse_split

    v0 = make_split(label_table(n_gold=20, n_weak=0), n_val_gold=5, weak_val_fraction=0.2, seed=1)
    with pytest.raises(ValueError, match="gold studies differ"):
        reuse_split(v0, label_table(n_gold=19, n_weak=0), weak_val_fraction=0.2, seed=1)


def test_v1_run_reuses_v0_split_and_records_run_config(tmp_path):
    gold_only = label_table(n_gold=12, n_weak=0)
    full = label_table(n_gold=12, n_weak=6)
    write_cache(tmp_path / "cache", full["study_id"].tolist())
    base = {"epochs": 1, "batch_size": 4, "freeze_epochs": 0, "n_val_gold": 4, "pretrained": False, "num_workers": 0}

    train(TrainConfig(version="v0", **base), gold_only, tmp_path / "cache", tmp_path / "v0")
    v0_split = pd.read_csv(tmp_path / "v0" / "splits.csv")
    summary = train(TrainConfig(version="v1", **base), full, tmp_path / "cache", tmp_path / "v1", previous_split=v0_split)

    v1_split = pd.read_csv(tmp_path / "v1" / "splits.csv")
    assert _gold_val(v1_split) == _gold_val(v0_split)
    info = json.loads((tmp_path / "v1" / "run_config.json").read_text())
    assert info["split_reused"] and set(info["pos_weight"]) == set(FINDINGS)
    assert all(1.0 <= w <= info["config"]["pos_weight_cap"] for w in info["pos_weight"].values())
    assert "val_mean_auc_weak" in summary["history"][0]