"""End-to-end smoke test with throwaway data.

Generates a dozen random images + random 1..5 labels in a SEPARATE temp dir
(data/_smoke/), then runs the FULL pipeline: data loader -> B2 & M1
forward/backward -> save a temp checkpoint -> Grad-CAM + rule-based feedback.
If it prints "PIPELINE OK", every moving part works.

This never touches data/custom/ or the real results/checkpoints, and it cleans
up after itself.

Run in Colab:   !python scripts/smoke_test.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import AXES, DATA_ROOT, CHECKPOINT_DIR
from src.dataset import CustomAxisDataset
from src.models.resnet_models import ResNetLinear, ResNetMultiHead
from torch.utils.data import DataLoader

SMOKE_DIR = DATA_ROOT / "_smoke"
SMOKE_CKPT = CHECKPOINT_DIR / "_smoke_m1.pt"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

N = 12
print(f"[1/5] Writing {N} random throwaway images + labels to {SMOKE_DIR} ...")
rng = np.random.default_rng(0)
rows = []
for i in range(N):
    arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    fname = f"img_{i:02d}.jpg"
    Image.fromarray(arr).save(SMOKE_DIR / fname)
    rows.append({"filename": fname,
                 **{a: int(rng.integers(1, 6)) for a in AXES}})
pd.DataFrame(rows).to_csv(SMOKE_DIR / "labels.csv", index=False)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[2/5] Device: {dev}")

ds = CustomAxisDataset(labels_csv=SMOKE_DIR / "labels.csv",
                       img_dir=SMOKE_DIR, train=True)
loader = DataLoader(ds, batch_size=4, shuffle=True)


def tiny_train(model, name, epochs=2):
    model = model.to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        print(f"   [{name}] epoch {ep+1}/{epochs}  loss {loss.item():.3f}")
    return model


try:
    print("[3/5] B2 forward/backward ...")
    tiny_train(ResNetLinear(), "B2")

    print("[4/5] M1 forward/backward + save temp checkpoint ...")
    m1 = tiny_train(ResNetMultiHead(), "M1")
    torch.save(m1.state_dict(), SMOKE_CKPT)

    print("[5/5] Grad-CAM + feedback on one image ...")
    from src.gradcam_utils import PhotoMentor
    mentor = PhotoMentor(SMOKE_CKPT, device=dev)
    res = mentor.analyse(Image.open(SMOKE_DIR / "img_00.jpg"))
    print("   scores:", {k: round(v, 2) for k, v in res["scores"].items()})
    print("   heatmap shapes:", {k: v.shape for k, v in res["heatmaps"].items()})
    print("   feedback:", res["feedback"])

    print("\nPIPELINE OK  - data loading, both models, Grad-CAM and feedback all run.")
finally:
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    SMOKE_CKPT.unlink(missing_ok=True)
    print("(cleaned up temp smoke files)")
