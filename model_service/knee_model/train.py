"""Trains KneeNet and saves the checkpoint, split and validation results that Person 4 evaluates.

Usage: python -m model_service.knee_model.train --train-csv ... --cache-dir ... --out-dir ... --version v0
Add --weak-labels <file> once Person 2's labels exist.
Add --split-csv <v0 splits.csv> to reuse v0's gold validation split for v1.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model_service.knee_model.dataset import KneeDataset, cache_file
from model_service.knee_model.labels import (
    WEAK_WEIGHT,
    build_label_table,
    label_arrays,
    positive_counts,
)
from model_service.knee_model.model import KneeNet, save_checkpoint
from shared.schema import FINDINGS, PLANES

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.csv"


@dataclass
class TrainConfig:
    version: str = "v0"
    seed: int = 30018
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    freeze_epochs: int = 3
    plane_loss_weight: float = 0.3
    pos_weight_cap: float = 10.0
    weak_weight: float = WEAK_WEIGHT  # recorded here; applied when the label table is built
    grad_clip: float = 1.0  # max gradient norm; 0 turns clipping off
    amp: bool = True  # mixed precision, GPU only
    n_val_gold: int = 14
    weak_val_fraction: float = 0.15
    patience: int = 5
    pooling: str = "max"
    pretrained: bool = True
    num_workers: int = 2
    extra: dict = field(default_factory=dict)


# --- Split ---

def make_split(table: pd.DataFrame, n_val_gold: int, weak_val_fraction: float, seed: int, n_tries: int = 500) -> pd.DataFrame:
    # Tries many seeded splits and keeps the one giving rare findings the most validation positives.
    rng = np.random.default_rng(seed)
    gold = table[table["label_source"] == "gold"].reset_index(drop=True)
    weak = table[table["label_source"] == "weak"].reset_index(drop=True)
    if n_val_gold >= len(gold):
        raise ValueError(f"n_val_gold={n_val_gold} leaves no gold studies for training (only {len(gold)})")

    gold_labels, _ = label_arrays(gold)
    total_pos = gold_labels.sum(axis=0)
    best_choice, best_score = None, None
    for _ in range(n_tries):
        choice = rng.choice(len(gold), size=n_val_gold, replace=False)
        val_pos = gold_labels[choice].sum(axis=0)
        train_pos = total_pos - val_pos
        score = (min(val_pos[total_pos >= 2], default=0), min(train_pos[total_pos >= 2], default=0), -np.abs(val_pos / np.maximum(total_pos, 1) - n_val_gold / len(gold)).sum())
        if best_score is None or score > best_score:
            best_choice, best_score = choice, score

    split = pd.DataFrame({"study_id": table["study_id"], "label_source": table["label_source"], "split": "train"})
    val_ids = set(gold.loc[best_choice, "study_id"])
    if len(weak):
        n_weak_val = int(round(len(weak) * weak_val_fraction))
        val_ids |= set(weak.loc[rng.choice(len(weak), size=n_weak_val, replace=False), "study_id"])
    split.loc[split["study_id"].isin(val_ids), "split"] = "val"
    return split


def reuse_split(previous: pd.DataFrame, table: pd.DataFrame, weak_val_fraction: float, seed: int) -> pd.DataFrame:
    # Keeps an earlier run's gold validation studies (e.g. v0's splits.csv) so runs are compared on the same data.
    # Weak studies keep their earlier assignment; weak studies new to this run are split at weak_val_fraction.
    for column in ("study_id", "label_source", "split"):
        if column not in previous.columns:
            raise ValueError(f"the split file needs a '{column}' column")
    previous_gold = set(previous.loc[previous["label_source"] == "gold", "study_id"])
    current_gold = set(table.loc[table["label_source"] == "gold", "study_id"])
    if previous_gold != current_gold:
        raise ValueError(f"the gold studies differ from the split file ({len(previous_gold - current_gold)} only in the "
                         f"file, {len(current_gold - previous_gold)} only in this run); check the cache and manifest")

    split = pd.DataFrame({"study_id": table["study_id"], "label_source": table["label_source"]})
    split["split"] = split["study_id"].map(previous.set_index("study_id")["split"])
    new_weak = split.index[split["split"].isna()]
    split.loc[new_weak, "split"] = "train"
    if len(new_weak):
        rng = np.random.default_rng(seed)
        n_val = round(len(new_weak) * weak_val_fraction)
        split.loc[rng.choice(new_weak, size=n_val, replace=False), "split"] = "val"
    return split


# --- Loss ---

def compute_pos_weight(labels: np.ndarray, mask: np.ndarray, cap: float) -> torch.Tensor:
    pos = (labels * mask).sum(axis=0)
    neg = ((1 - labels) * mask).sum(axis=0)
    weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(weight, 1.0, cap), dtype=torch.float32)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor,
               sample_weight: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight, reduction="none")
    weights = label_mask * sample_weight.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def total_loss(study_logits, plane_logits, batch, pos_weight, plane_loss_weight) -> torch.Tensor:
    # Study loss plus a weighted per-plane loss.
    loss = masked_bce(study_logits, batch["labels"], batch["label_mask"], batch["sample_weight"], pos_weight)
    if plane_loss_weight > 0:
        plane_losses = []
        for p in range(plane_logits.shape[1]):
            present = batch["plane_mask"][:, p].float()
            if present.sum() == 0:
                continue
            plane_losses.append(masked_bce(plane_logits[:, p], batch["labels"], batch["label_mask"],
                                           batch["sample_weight"] * present, pos_weight))
        if plane_losses:
            loss = loss + plane_loss_weight * torch.stack(plane_losses).mean()
    return loss


# --- Training and evaluation ---

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(model: KneeNet, loader: DataLoader, device: torch.device, amp: bool = True) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            study_logits, plane_logits = model(batch["image"], batch["plane_mask"])
        study = torch.sigmoid(study_logits.float()).cpu().numpy()
        planes = torch.sigmoid(plane_logits.float()).cpu().numpy()
        for i, study_id in enumerate(batch["study_id"]):
            row = {"study_id": study_id, **{name: float(study[i, j]) for j, name in enumerate(FINDINGS)}}
            for p, plane in enumerate(PLANES):
                row.update({f"{name}|{plane}": float(planes[i, p, j]) for j, name in enumerate(FINDINGS)})
            rows.append(row)
    return pd.DataFrame(rows)


def per_finding_auc(predictions: pd.DataFrame, labels_table: pd.DataFrame) -> pd.DataFrame:
    merged = labels_table[["study_id", *FINDINGS]].merge(predictions[["study_id", *FINDINGS]], on="study_id", suffixes=("_true", "_pred"))
    rows = []
    for name in FINDINGS:
        known = merged[merged[f"{name}_true"].notna()]
        y, p = known[f"{name}_true"].to_numpy(), known[f"{name}_pred"].to_numpy()
        n_pos = int((y == 1).sum())
        auc = roc_auc_score(y, p) if 0 < n_pos < len(y) else float("nan")
        rows.append({"finding": name, "auc": auc, "positives": n_pos, "negatives": int((y == 0).sum())})
    return pd.DataFrame(rows)


def _mean_auc(auc: pd.DataFrame) -> float:
    return float(auc["auc"].mean(skipna=True)) if auc["auc"].notna().any() else float("nan")


def train(config: TrainConfig, label_table: pd.DataFrame, cache_dir: str | Path, out_dir: str | Path,
          previous_split: pd.DataFrame | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cached = label_table["study_id"].map(lambda s: cache_file(cache_dir, s).exists())
    if (~cached).any():
        print(f"WARNING: {int((~cached).sum())} labelled studies are not in the cache and are skipped")
    label_table = label_table[cached].reset_index(drop=True)

    positive_counts(label_table).to_csv(out_dir / "label_counts.csv", index=False)
    if previous_split is None:
        split = make_split(label_table, config.n_val_gold, config.weak_val_fraction, config.seed)
    else:
        split = reuse_split(previous_split, label_table, config.weak_val_fraction, config.seed)
    split.to_csv(out_dir / "splits.csv", index=False)
    train_table = label_table[split["split"].eq("train").to_numpy()]
    val_table = label_table[split["split"].eq("val").to_numpy()]
    val_gold = val_table[val_table["label_source"] == "gold"]
    val_weak = val_table[val_table["label_source"] == "weak"]
    print(f"train: {len(train_table)} studies, val: {len(val_table)} ({len(val_gold)} gold)")

    train_labels, train_mask = label_arrays(train_table)
    pos_weight = compute_pos_weight(train_labels, train_mask, config.pos_weight_cap).to(device)

    loader_args = {"batch_size": config.batch_size, "num_workers": config.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(KneeDataset(train_table, cache_dir, augment=True, seed=config.seed), shuffle=True, **loader_args)
    val_loader = DataLoader(KneeDataset(val_table, cache_dir), shuffle=False, **loader_args)

    model = KneeNet(pretrained=config.pretrained, pooling=config.pooling).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(config.epochs, 1))
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    sample = np.load(cache_file(cache_dir, train_table["study_id"].iloc[0]))["array"]
    shape_meta = {"n_slices": int(sample.shape[1]), "image_size": int(sample.shape[2])}

    # Everything needed to reproduce or describe this run in the report.
    run_info = {
        "config": asdict(config), **shape_meta,
        "n_train": len(train_table), "n_val": len(val_table), "n_val_gold": len(val_gold), "n_val_weak": len(val_weak),
        "train_by_source": train_table["label_source"].value_counts().to_dict(),
        "pos_weight": {name: round(float(w), 3) for name, w in zip(FINDINGS, pos_weight.cpu().tolist())},
        "split_reused": previous_split is not None, "mixed_precision": use_amp,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_info, indent=2))

    history, best_score, best_epoch, stale = [], -np.inf, -1, 0
    for epoch in range(config.epochs):
        start = time.perf_counter()
        model.set_backbone_trainable(epoch >= config.freeze_epochs)
        model.train()
        running, batches = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                study_logits, plane_logits = model(batch["image"], batch["plane_mask"])
            loss = total_loss(study_logits, plane_logits, batch, pos_weight, config.plane_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss became {loss.item()} in epoch {epoch + 1}; lower --lr or check the cache")
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            running, batches = running + loss.item(), batches + 1
        scheduler.step()

        predictions = predict(model, val_loader, device, amp=use_amp)
        auc_gold = per_finding_auc(predictions, val_gold)
        mean_auc = _mean_auc(auc_gold)
        entry = {"epoch": epoch + 1, "train_loss": running / max(batches, 1), "val_mean_auc_gold": mean_auc,
                 "lr": optimiser.param_groups[0]["lr"], "backbone_frozen": epoch < config.freeze_epochs,
                 "seconds": round(time.perf_counter() - start, 1)}
        if len(val_weak):
            entry["val_mean_auc_weak"] = _mean_auc(per_finding_auc(predictions, val_weak))
        history.append(entry)
        print(json.dumps(entry))

        # Early stopping on gold validation AUC.
        score = mean_auc if not np.isnan(mean_auc) else epoch
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            save_checkpoint(out_dir / f"knee_{config.version}.pt", model, {
                "version": config.version, "epoch": epoch + 1, "val_mean_auc_gold": mean_auc,
                "split_seed": config.seed, "config": asdict(config), **shape_meta,
            })
            predictions.to_csv(out_dir / "val_predictions.csv", index=False)
            auc_gold.to_csv(out_dir / "val_auc.csv", index=False)
        else:
            stale += 1
            if stale >= config.patience:
                print(f"stopping early: no improvement for {config.patience} epochs")
                break

    summary = {"best_epoch": best_epoch, "best_val_mean_auc_gold": best_score if best_epoch > 0 else None,
               "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "history": history}
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the knee abnormality model.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weak-labels", type=Path, default=None, help="Person 2's weak label CSV; leave out for a gold-only run")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--version", default="v0")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--freeze-epochs", type=int, default=TrainConfig.freeze_epochs)
    parser.add_argument("--n-val-gold", type=int, default=TrainConfig.n_val_gold)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--split-csv", type=Path, default=None,
                        help="an earlier run's splits.csv; reuses its gold validation studies (use v0's for v1)")
    parser.add_argument("--weak-weight", type=float, default=TrainConfig.weak_weight)
    parser.add_argument("--pos-weight-cap", type=float, default=TrainConfig.pos_weight_cap)
    parser.add_argument("--plane-loss-weight", type=float, default=TrainConfig.plane_loss_weight)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--pooling", choices=["max", "mean"], default=TrainConfig.pooling)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    weak = pd.read_csv(args.weak_labels) if args.weak_labels else None
    table = build_label_table(pd.read_csv(args.train_csv), manifest, weak, weak_weight=args.weak_weight)
    previous_split = pd.read_csv(args.split_csv) if args.split_csv else None
    config = TrainConfig(version=args.version, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                         freeze_epochs=args.freeze_epochs, n_val_gold=args.n_val_gold,
                         num_workers=args.num_workers, pretrained=not args.no_pretrained,
                         weak_weight=args.weak_weight, pos_weight_cap=args.pos_weight_cap,
                         plane_loss_weight=args.plane_loss_weight, weight_decay=args.weight_decay,
                         pooling=args.pooling, patience=args.patience, seed=args.seed,
                         grad_clip=args.grad_clip, amp=not args.no_amp)
    summary = train(config, table, args.cache_dir, args.out_dir, previous_split=previous_split)
    print(f"best epoch {summary['best_epoch']}, gold val mean AUC {summary['best_val_mean_auc_gold']}")


if __name__ == "__main__":
    main()