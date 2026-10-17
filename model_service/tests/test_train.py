import json

import pandas as pd

from model_service.knee_model.model import load_checkpoint
from model_service.knee_model.train import TrainConfig, make_split, train
from model_service.tests._fakes import label_table, write_cache


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