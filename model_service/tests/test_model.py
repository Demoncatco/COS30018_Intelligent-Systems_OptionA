import torch

from model_service.knee_model.model import KneeNet, load_checkpoint, logits_to_scores, save_checkpoint
from shared.schema import FINDINGS, PLANES, PlaneScores


def _model():
    torch.manual_seed(0)
    return KneeNet(pretrained=False).eval()


def test_output_shapes():
    study, planes = _model()(torch.rand(2, 3, 4, 32, 32))
    assert study.shape == (2, len(FINDINGS))
    assert planes.shape == (2, len(PLANES), len(FINDINGS))


def test_first_layer_takes_one_channel():
    assert _model().backbone.conv1.in_channels == 1


def test_missing_plane_is_left_out_of_the_study_score():
    model = _model()
    x = torch.rand(1, 3, 4, 32, 32)
    mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        study, planes = model(x, mask)
    assert torch.allclose(study[0], planes[0, :2].mean(dim=0), atol=1e-6)


def test_freezing_keeps_conv1_trainable_only():
    model = _model()
    model.set_backbone_trainable(False)
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert trainable == ["conv1.weight"]
    assert all(p.requires_grad for p in model.plane_heads.parameters())
    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    path = tmp_path / "knee.pt"
    save_checkpoint(path, model, {"version": "test", "n_slices": 4, "image_size": 32})
    loaded, meta = load_checkpoint(path)
    x = torch.rand(1, 3, 4, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x)[0], loaded(x)[0])
    assert meta["findings"] == list(FINDINGS) and meta["version"] == "test"


def test_scores_match_the_api_schema():
    study, planes = _model()(torch.rand(1, 3, 4, 32, 32))
    findings, per_plane = logits_to_scores(study[0], planes[0])
    assert set(findings) == set(FINDINGS)
    for entry in per_plane:
        PlaneScores(**entry)  # raises if a score is missing or outside 0 - 1