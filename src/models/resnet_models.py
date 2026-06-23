"""B2 and M1 deep-learning models, both built on a pretrained ResNet-50.

B2 -- ResNetLinear: frozen backbone + one linear head -> single score.
       The "basic transfer-learning" baseline.

M1 -- ResNetMultiHead: shared backbone + four small regression heads
       (composition, exposure, sharpness, background), each -> a 1..5 score.
       This is the main contribution. The backbone can be frozen (fast, for a
       1-week MVP) or partially unfrozen (layer4) if data and time allow.

Outputs are squashed to the [1, 5] range with a scaled sigmoid so predictions
are always valid scores. For AVA pretraining (1..10) we expose a `head_10`
linear layer used only during the pretraining phase.
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
    """B2: frozen backbone + single linear head."""

    def __init__(self, score_range=(1.0, 10.0)):
        super().__init__()
        self.backbone, feat = _backbone(trainable_layer4=False)
        self.head = nn.Linear(feat, 1)
        self.squash = _ScaledSigmoid(*score_range)

    def forward(self, x):
        f = self.backbone(x)
        return self.squash(self.head(f))      # [B, 1]


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
        # Pretraining head: predicts a single AVA mean score (1..10).
        self.head_10 = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(True),
                                     nn.Linear(256, 1))
        self.squash_10 = _ScaledSigmoid(1.0, 10.0)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        """Return [B, 4] tensor of axis scores in axis order."""
        f = self.backbone(x)
        outs = [self.squash(self.heads[a](f)) for a in self.axes]
        return torch.cat(outs, dim=1)

    def forward_pretrain(self, x):
        """Single-score path used during AVA pretraining."""
        f = self.backbone(x)
        return self.squash_10(self.head_10(f))   # [B, 1]
