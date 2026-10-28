"""Finds the folder of .dcm files for a series under the train_series root."""

from pathlib import Path


def resolve_series_dir(root, study_id: str, series_id: str) -> Path:
    """
    Accepts either layout:
        <root>/<study_id>/<series_id>/*.dcm
        <root>/<series_id>/*.dcm
    """
    root = Path(root)
    for cand in (root / study_id / series_id, root / series_id):
        if cand.is_dir() and any(cand.glob("*.dcm")):
            return cand
    raise FileNotFoundError(f"no .dcm files for study {study_id}, series {series_id} under {root}")
