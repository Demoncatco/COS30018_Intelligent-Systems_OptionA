"""
Train/validation/test split. Whole studies only, never individual slices.

Weak studies are used only for training. Gold studies are held out for validation and testing,
so reported accuracy is measured against real labels rather than the extraction model's errors.
"""

import pandas as pd


def split_by_label_source(table: pd.DataFrame, gold_val_frac: float = 0.5, seed: int = 0):
    """Returns (train, val, test). train = all weak studies; gold studies are divided into val/test."""
    train = table[table.label_source == "weak"].reset_index(drop=True)
    gold = table[table.label_source == "gold"].sample(frac=1.0, random_state=seed)
    n_val = int(round(len(gold) * gold_val_frac))
    val = gold.iloc[:n_val].reset_index(drop=True)
    test = gold.iloc[n_val:].reset_index(drop=True)
    return train, val, test
