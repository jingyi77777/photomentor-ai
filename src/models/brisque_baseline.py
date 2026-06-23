"""B1 -- BRISQUE baseline (classical, no machine learning of our own).

BRISQUE is a no-reference image-quality metric: lower = better perceptual
quality. It establishes the "without deep learning" floor the report contrasts
against. We use `piq` for a robust pure-PyTorch implementation (the older
`imagequality`/libsvm route is brittle on modern environments).

BRISQUE outputs ONE number, not four axes -- that is the point. It cannot give
per-axis or spatial feedback, which motivates B2 and M1.
"""
from __future__ import annotations
import torch
import piq
from torchvision import transforms
from PIL import Image

_to_tensor = transforms.ToTensor()


def brisque_score(img: Image.Image) -> float:
    """Return the BRISQUE score for a PIL image (lower = better quality)."""
    x = _to_tensor(img.convert("RGB")).unsqueeze(0)        # [1,3,H,W] in 0..1
    with torch.no_grad():
        score = piq.brisque(x, data_range=1.0, reduction="none")
    return float(score.item())


def brisque_to_quality_1_5(score: float) -> float:
    """Map a BRISQUE score (~0 good .. ~100 bad) onto a 1..5 quality scale.

    This is a simple monotonic mapping so B1 can be compared on the same axis
    as the other models. Calibrated loosely; not meant to be precise.
    """
    q = 5.0 - (score / 100.0) * 4.0
    return float(max(1.0, min(5.0, q)))
