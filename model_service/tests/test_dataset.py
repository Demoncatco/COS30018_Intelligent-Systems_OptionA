import torch

from model_service.knee_model.dataset import KneeDataset, augment
from model_service.tests._fakes import N_SLICES, SIZE, label_table, write_cache


def test_example_shapes_and_masks(tmp_path):
    table = label_table()
    write_cache(tmp_path, table["study_id"].tolist(), missing_axial={"weak0"})
    dataset = KneeDataset(table, tmp_path)
    gold = dataset[0]
    assert gold["image"].shape == (3, N_SLICES, SIZE, SIZE) and gold["image"].dtype == torch.float32
    assert gold["label_mask"].sum() == 12
    weak = dataset[len(table) - 6]
    assert weak["study_id"] == "weak0"
    assert weak["label_mask"][:3].sum() == 0  # the unknown findings are masked
    assert weak["plane_mask"].tolist() == [True, True, False]


def test_augmentation_keeps_missing_plane_black_and_values_in_range():
    volume = torch.rand(3, N_SLICES, SIZE, SIZE)
    volume[2] = 0
    out = augment(volume, torch.tensor([True, True, False]), torch.Generator().manual_seed(0))
    assert out[2].abs().sum() == 0
    assert out.min() >= 0 and out.max() <= 1
    assert out.shape == volume.shape


def test_augmentation_never_mirrors_left_and_right():
    # Left-bright input must stay left-bright.
    volume = torch.zeros(3, N_SLICES, SIZE, SIZE)
    volume[:, :, :, : SIZE // 2] = 1.0
    present = torch.tensor([True, True, True])
    for seed in range(20):
        out = augment(volume, present, torch.Generator().manual_seed(seed))
        assert (out[..., : SIZE // 2].mean() > out[..., SIZE // 2 :].mean()).item()