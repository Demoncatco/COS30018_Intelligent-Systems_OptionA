"""KneeNet: ResNet18 run on every slice, pooled per plane, one head per plane, averaged over present planes.

Input (batch, 3, 20, 224, 224) -> study logits (batch, 12) and per-plane logits (batch, 3, 12).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torchvision

from shared.schema import FINDINGS, PLANES

# ImageNet normalisation, averaged over RGB, for greyscale input.
IMAGENET_GREY_MEAN = 0.449
IMAGENET_GREY_STD = 0.226

Pooling = Literal["max", "mean"]


def _greyscale_resnet18(pretrained: bool) -> tuple[nn.Module, int]:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = torchvision.models.resnet18(weights=weights)

    old = backbone.conv1
    # Average the pretrained RGB filters so conv1 takes 1 channel.
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    backbone.conv1 = new

    n_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, n_features


class KneeNet(nn.Module):
    def __init__(
        self,
        n_findings: int = len(FINDINGS),
        n_planes: int = len(PLANES),
        pretrained: bool = True,
        pooling: Pooling = "max",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if pooling not in ("max", "mean"):
            raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")
        self.n_findings = n_findings
        self.n_planes = n_planes
        self.pooling = pooling

        self.backbone, n_features = _greyscale_resnet18(pretrained)
        self.plane_heads = nn.ModuleList(
            [nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, n_findings)) for _ in range(n_planes)]
        )

    def forward(self, x: torch.Tensor, plane_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, P, S, H, W). plane_mask: (B, P). Returns study (B, F) and plane (B, P, F) logits.
        if x.dim() != 5:
            raise ValueError(f"expected input of shape (batch, planes, slices, H, W), got {tuple(x.shape)}")
        batch, planes, slices, height, width = x.shape
        if planes != self.n_planes:
            raise ValueError(f"model was built for {self.n_planes} planes, input has {planes}")

        x = (x - IMAGENET_GREY_MEAN) / IMAGENET_GREY_STD
        features = self.backbone(x.reshape(batch * planes * slices, 1, height, width))
        features = features.reshape(batch, planes, slices, -1)

        pooled = features.amax(dim=2) if self.pooling == "max" else features.mean(dim=2)

        plane_logits = torch.stack([head(pooled[:, p]) for p, head in enumerate(self.plane_heads)], dim=1)

        if plane_mask is None:
            plane_mask = torch.ones(batch, planes, dtype=torch.bool, device=x.device)
        weights = plane_mask.to(plane_logits.dtype).unsqueeze(-1)
        # Average over present planes only.
        study_logits = (plane_logits * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return study_logits, plane_logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        # conv1 always stays trainable since it was adapted to 1 channel.
        for name, param in self.backbone.named_parameters():
            param.requires_grad = trainable or name.startswith("conv1")
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(trainable and self.training)

    def train(self, mode: bool = True) -> "KneeNet":
        super().train(mode)
        # Keep frozen batch-norm layers in eval mode.
        frozen = not any(p.requires_grad for n, p in self.backbone.named_parameters() if not n.startswith("conv1"))
        if mode and frozen:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


# --- Checkpoints ---

def save_checkpoint(path: str | Path, model: KneeNet, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_meta = {
        "findings": list(FINDINGS),
        "planes": list(PLANES),
        "pooling": model.pooling,
        "n_findings": model.n_findings,
        "n_planes": model.n_planes,
        **meta,
    }
    torch.save({"state_dict": model.state_dict(), "meta": full_meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[KneeNet, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    if meta["findings"] != list(FINDINGS):
        raise ValueError(f"checkpoint findings {meta['findings']} do not match shared/schema.py FINDINGS")
    model = KneeNet(
        n_findings=meta["n_findings"], n_planes=meta["n_planes"], pretrained=False, pooling=meta["pooling"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, meta


def logits_to_scores(study_logits: torch.Tensor, plane_logits: torch.Tensor) -> tuple[dict, list[dict]]:
    # One study's logits -> the findings / per_plane dicts PredictionResponse expects.
    study = torch.sigmoid(study_logits.float()).tolist()
    planes = torch.sigmoid(plane_logits.float()).tolist()
    findings = {name: float(p) for name, p in zip(FINDINGS, study)}
    per_plane = [
        {"plane": plane, "scores": {name: float(p) for name, p in zip(FINDINGS, scores)}}
        for plane, scores in zip(PLANES, planes)
    ]
    return findings, per_plane