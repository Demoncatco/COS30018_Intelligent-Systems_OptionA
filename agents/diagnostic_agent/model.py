"""
Baseline model: a 2D CNN run over every slice, max-pooled over slices (MRNet-style),
one pass per plane with shared weights, plane features concatenated into 12 logits.
"""

import torch
import torch.nn as nn
import torchvision

from config import ABNORMALITIES, PLANES


class SliceEncoder(nn.Module):
    """(B, D, H, W) stack of slices -> (B, feat) by running a 2D CNN per slice and max-pooling over D."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        x = x.reshape(b * d, 1, h, w).expand(-1, 3, -1, -1)  # grey -> 3 channels
        f = self.net(x).reshape(b, d, -1)
        return f.max(dim=1).values


class KneeNet(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        self.encoder = SliceEncoder(pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.feat_dim * len(PLANES), len(ABNORMALITIES)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3 planes, D, H, W) -> logits (B, 12)."""
        feats = [self.encoder(images[:, p]) for p in range(images.shape[1])]
        return self.head(torch.cat(feats, dim=1))
