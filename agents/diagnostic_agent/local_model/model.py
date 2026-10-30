"""
The network the Kaggle notebook trains: a ResNet18 run over every slice, max-pooled over slices, shared across the three
planes, with the three 512-d plane features joined into one Linear(1536 -> 12) head.

Layer and parameter names match the notebook, so its `best_model.pt` state_dict loads unchanged.
`plane_logits` is the only addition: it splits the final layer into per-plane scores without any retraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from shared.schema import FINDINGS, PLANES


class SliceEncoder(nn.Module):
    """(B, D, H, W) stack of slices -> (B, feat): a 2D CNN per slice, max-pooled over D."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        x = x.reshape(b * d, 1, h, w).expand(-1, 3, -1, -1)  # grey -> 3 channels
        return self.net(x).reshape(b, d, -1).max(dim=1).values


class NotebookKneeNet(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = SliceEncoder(pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.feat_dim * len(PLANES), len(FINDINGS)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3 planes, D, H, W) -> logits (B, 12)."""
        feats = [self.encoder(images[:, p]) for p in range(images.shape[1])]
        return self.head(torch.cat(feats, dim=1))

    @torch.no_grad()
    def plane_logits(self, plane_images: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        plane_images: {plane index (0 sagittal, 1 coronal, 2 axial): (B, D, H, W)} for the planes that are present.
        Returns (fused (B, 12), per_plane (B, n_present, 12)) as logits. Call model.eval() first.

        The head is linear, so each plane's contribution to a logit is exact:
          per-plane logit = bias + 3 * W_p . f_p
          fused logit     = bias + (3 / n_present) * sum_p W_p . f_p
        With all three planes the fused logit equals forward() and is the mean of the three per-plane logits. With a
        missing plane the present ones are rescaled instead of feeding a blank image. These are a split of one jointly
        trained head, not three separately trained models.
        """
        lin = self.head[1]
        f_dim, n_planes = self.encoder.feat_dim, len(PLANES)
        contrib = [self.encoder(x) @ lin.weight[:, p * f_dim:(p + 1) * f_dim].T for p, x in plane_images.items()]
        c = torch.stack(contrib, dim=1)
        fused = lin.bias + c.sum(dim=1) * (n_planes / len(contrib))
        return fused, lin.bias + c * n_planes
