#!/usr/bin/env python3
"""
Baseline training script. Wires the other modules together; holds no data or model logic itself.

Example:
    python train.py --train-series /data/train_series --findings findings.csv \
        --pretrained --epochs 20 --cache-dir cache
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ABNORMALITIES
from dataset import KneeStudyDataset
from labels import build_label_table
from metrics import format_report, mean_auc, per_label_auc
from model import KneeNet
from splits import split_by_label_source


def pos_weight_from(table) -> torch.Tensor:
    """Up-weights rare labels in the loss. Clipped so a label with 0 positives does not explode."""
    pos = table[ABNORMALITIES].sum().to_numpy(dtype=np.float32)
    neg = len(table) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        probs.append(torch.sigmoid(model(batch["images"].to(device))).cpu().numpy())
        ys.append(batch["labels"].numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_one_epoch(model, loader, optimiser, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimiser.zero_grad()
        loss = loss_fn(model(batch["images"].to(device)), batch["labels"].to(device))
        loss.backward()
        optimiser.step()
        total += loss.item() * len(batch["labels"])
        n += len(batch["labels"])
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-series", required=True, help="root folder of the DICOM series")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--series-selection", default="series_selection.csv")
    ap.add_argument("--train-csv", default="train.csv")
    ap.add_argument("--findings", default="findings.csv", help="output of extract_knee_findings.py (weak labels)")
    ap.add_argument("--cache-dir", default=None, help="cache preprocessed volumes here (much faster epochs)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained ResNet18 (downloads weights)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="best_model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    table = build_label_table(args.manifest, args.train_csv, args.findings)
    train_t, val_t, test_t = split_by_label_source(table, seed=args.seed)
    print(f"train(weak)={len(train_t)}  val(gold)={len(val_t)}  test(gold)={len(test_t)}  device={device}")

    def make(t, augment):
        ds = KneeStudyDataset(t, args.series_selection, args.train_series, args.depth, args.size,
                              args.cache_dir, augment)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=augment, num_workers=args.workers)

    train_dl, val_dl, test_dl = make(train_t, True), make(val_t, False), make(test_t, False)

    model = KneeNet(pretrained=args.pretrained).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(train_t).to(device))

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_dl, optimiser, loss_fn, device)
        auc = mean_auc(per_label_auc(*predict(model, val_dl, device)))
        print(f"epoch {epoch:>3}  train loss {loss:.4f}  val mean AUC {auc:.3f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device))
    print("\nTest (gold) results for the best-validation checkpoint:")
    print(format_report(per_label_auc(*predict(model, test_dl, device))))


if __name__ == "__main__":
    main()
