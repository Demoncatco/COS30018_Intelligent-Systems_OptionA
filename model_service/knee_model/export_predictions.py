"""Runs a trained checkpoint over a set of studies on Kaggle and writes predictions.json for MODEL_MODE=replay.

Usage (Kaggle notebook, from the repo root):
    python -m model_service.knee_model.export_predictions \
        --checkpoint /kaggle/working/run_v1/knee_v1.pt \
        --splits-csv /kaggle/working/run_v1/splits.csv --split val \
        --cache-dir /kaggle/working/cache \
        --dataset-root /kaggle/input/rsna-knee-abnormality-detection \
        --series-csv /kaggle/input/rsna-knee-abnormality-detection/train_series.csv \
        --out /kaggle/working/predictions.json

Then download predictions.json and put it at model_service/checkpoints/predictions.json (see README section 8.1).

Default is the validation split only: those studies were never trained on, so replaying them in the agent
evaluation gives honest scores. Use --split all only for demos, never for reported metrics.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from model_service.knee_model.dataset import cache_file
from model_service.knee_model.inference import KneePredictor
from shared.schema import PredictionResponse, ReplayPredictionsFile


def choose_study_ids(splits: pd.DataFrame, split: str) -> list[str]:
    if "study_id" not in splits.columns:
        raise ValueError("the study list needs a 'study_id' column")
    if split != "all":
        if "split" not in splits.columns:
            raise ValueError(f"--split {split} needs a 'split' column (use --split all for a plain study list)")
        splits = splits[splits["split"] == split]
    return list(dict.fromkeys(splits["study_id"].astype(str)))


def export(
    predictor: KneePredictor,
    study_ids: list[str],
    cache_dir: Path | None,
    series_table: pd.DataFrame | None,
    dataset_root: Path | None,
    split_folder: str,
    source: str,
) -> tuple[ReplayPredictionsFile, list[dict]]:
    # Scores every study; a study that fails is reported and skipped instead of stopping the run.
    predictions: list[PredictionResponse] = []
    failures: list[dict] = []
    try:
        from tqdm import tqdm
        progress = tqdm(study_ids, desc="scoring studies")
    except ImportError:
        progress = study_ids

    for study_id in progress:
        try:
            if cache_dir is not None and cache_file(cache_dir, study_id).exists():
                prediction = predictor.predict_cached(study_id, cache_dir, series_table, dataset_root, split_folder)
            elif series_table is not None and dataset_root is not None:
                prediction = predictor.predict_study(study_id, series_table, dataset_root, split_folder)
            else:
                raise FileNotFoundError("not in the cache, and no --dataset-root/--series-csv to load it from")
            predictions.append(prediction)
        except Exception as error:  # noqa: BLE001
            failures.append({"study_id": study_id, "error": repr(error)})

    replay = ReplayPredictionsFile(
        model_version=predictor.model_version,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=f"{source} ({len(predictions)} studies)",
        predictions=predictions,
    )
    return replay, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export predictions.json for the Model Service's replay mode.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-csv", type=Path, default=None,
                        help="splits.csv from the training run (default: next to the checkpoint)")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--cache-dir", type=Path, default=None, help="dataset.py's .npz cache (fast path)")
    parser.add_argument("--dataset-root", type=Path, default=None, help="needed for laterality and uncached studies")
    parser.add_argument("--series-csv", type=Path, default=None)
    parser.add_argument("--split-folder", default="train_series")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("predictions.json"))
    args = parser.parse_args(argv)

    splits_csv = args.splits_csv or args.checkpoint.parent / "splits.csv"
    study_ids = choose_study_ids(pd.read_csv(splits_csv), args.split)
    if not study_ids:
        print(f"no studies with split={args.split} in {splits_csv}", file=sys.stderr)
        return 1
    series_table = pd.read_csv(args.series_csv) if args.series_csv else None

    predictor = KneePredictor(args.checkpoint, device=args.device)
    print(f"{predictor.model_version} on {predictor.device}: scoring {len(study_ids)} studies (split={args.split})")
    replay, failures = export(predictor, study_ids, args.cache_dir, series_table, args.dataset_root,
                              args.split_folder, source=f"kaggle: {args.checkpoint.name} on {splits_csv.name} {args.split}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {len(replay.predictions)} predictions to {args.out}")
    if failures:
        print(f"{len(failures)} studies failed:", file=sys.stderr)
        for failure in failures[:10]:
            print(f"  {failure['study_id']}: {failure['error']}", file=sys.stderr)
    return 0 if replay.predictions else 1


if __name__ == "__main__":
    raise SystemExit(main())