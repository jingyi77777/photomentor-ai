"""Grad-CAM explainability + a convenient single-image inference wrapper.

`PhotoMentor` loads a trained M1 model and, for one image, returns:
  * the four axis scores (1..5)
  * a Grad-CAM heatmap per axis (numpy HxW in 0..1)
  * the rule-based feedback sentences

Grad-CAM targets the last convolutional block of ResNet-50 (layer4[-1]).
Each axis head is a separate regression output, so we build one CAM per head
by treating that head's scalar output as the target.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .config import AXES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEVICE
from .dataset import build_transforms
from .models.resnet_models import ResNetMultiHead
from .feedback import generate_feedback


class _SingleAxisWrapper(torch.nn.Module):
    """Expose a single axis output so Grad-CAM has a scalar target."""

    def __init__(self, model: ResNetMultiHead, axis_idx: int):
        super().__init__()
        self.model = model
        self.axis_idx = axis_idx

    def forward(self, x):
        return self.model(x)[:, self.axis_idx: self.axis_idx + 1]


class PhotoMentor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = device or (DEVICE if torch.cuda.is_available() else "cpu")
        self.model = ResNetMultiHead().to(self.device).eval()
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        # Grad-CAM needs gradients to reach the target conv layer. The backbone
        # was frozen for training; re-enable grad here. This object is inference
        # only, so flipping requires_grad has no training side-effects and is
        # what lets Grad-CAM capture activations' gradients on layer4.
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.tf = build_transforms(train=False)
        self.target_layer = self.model.backbone.layer4[-1]

    def _denorm(self, x: torch.Tensor) -> np.ndarray:
        """Tensor -> HxWx3 float image in 0..1 for overlaying CAMs."""
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = (x.cpu() * std + mean).clamp(0, 1)
        return img.permute(1, 2, 0).numpy()

    @torch.no_grad()
    def _scores(self, x: torch.Tensor) -> dict[str, float]:
        out = self.model(x).squeeze(0).cpu().numpy()
        return {a: float(out[i]) for i, a in enumerate(AXES)}

    def analyse(self, img: Image.Image, make_heatmaps: bool = True) -> dict:
        img = img.convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        scores = self._scores(x)
        result = {"scores": scores,
                  "feedback": generate_feedback(scores, img),
                  "heatmaps": {}}

        if make_heatmaps:
            rgb = self._denorm(x[0])
            for i, axis in enumerate(AXES):
                wrapper = _SingleAxisWrapper(self.model, i).to(self.device).eval()
                cam = GradCAM(model=wrapper, target_layers=[self.target_layer])
                grayscale = cam(input_tensor=x,
                                targets=[RawScoresOutputTarget()])[0]
                overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
                result["heatmaps"][axis] = overlay        # HxWx3 uint8
        return result
