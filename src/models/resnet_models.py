"""B2 and M1 deep-learning models, both built on a pretrained ResNet-50.

B2 -- ResNetLinear: frozen backbone + one linear head -> single score.
       The "basic transfer-learning" baseline.

M1 -- ResNetMultiHead: shared backbone + four small regression heads
       (composition, exposure, sharpness, background), each -> a 1..5 score.
       This is the main contribution. The backbone can be frozen (fast, for a
       1-week MVP) or partially unfrozen (layer4) if data and time allow.

Outputs are squashed to the [1, 5] range with a scaled sigmoid so predictions
are always valid scores.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from ..config import AXES, AXIS_SCALE


def _backbone(trainable_layer4: bool = False) -> tuple[nn.Module, int]:
    """Return a pretrained ResNet-50 with the classifier removed."""
    net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feat_dim = net.fc.in_features            # 2048
    net.fc = nn.Identity()                   # we attach our own heads

    for p in net.parameters():
        p.requires_grad = False
    if trainable_layer4:                     # optional light fine-tuning
        for p in net.layer4.parameters():
            p.requires_grad = True
    return net, feat_dim


class _ScaledSigmoid(nn.Module):
    """Map raw logits to the [lo, hi] interval."""

    def __init__(self, lo: float, hi: float):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(x)


class ResNetLinear(nn.Module):
    """B2: frozen backbone + a single linear layer regressing all 4 axes.

    The simplest possible transfer-learning head: one Linear(2048 -> 4).
    Contrasts with M1's deeper per-axis heads.
    """

    def __init__(self, n_outputs: int = len(AXES), score_range=AXIS_SCALE):
        super().__init__()
        self.backbone, feat = _backbone(trainable_layer4=False)
        self.head = nn.Linear(feat, n_outputs)
        self.squash = _ScaledSigmoid(*score_range)

    def forward(self, x):
        f = self.backbone(x)
        return self.squash(self.head(f))      # [B, 4]


class ResNetMultiHead(nn.Module):
    """M1: shared backbone + one regression head per axis (1..5)."""

    def __init__(self, trainable_layer4: bool = False, dropout: float = 0.3):
        super().__init__()
        self.backbone, feat = _backbone(trainable_layer4)
        self.axes = AXES
        self.heads = nn.ModuleDict({
            axis: nn.Sequential(
                nn.Linear(feat, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            for axis in AXES
        })
        self.squash = _ScaledSigmoid(*AXIS_SCALE)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        """Return [B, 4] tensor of axis scores in axis order."""
        f = self.backbone(x)
        outs = [self.squash(self.heads[a](f)) for a in self.axes]
        return torch.cat(outs, dim=1)
