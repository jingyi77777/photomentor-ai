"""End-to-end smoke test with throwaway data.

Generates a dozen random images + random 1..5 labels, then runs the FULL
pipeline: data loader -> B2 & M1 forward/backward -> save checkpoint ->
Grad-CAM + rule-based feedback on one image. If it prints "PIPELINE OK",
every moving part works and you can swap in real photos later.

Run in Colab:   !python scripts/smoke_test.py
NOTE: this writes random labels into data/custom/. Delete them before adding
real data, or just overwrite labels.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import AXES, CUSTOM_DIR, CHECKPOINT_DIR
from src.dataset import CustomAxisDataset
from src.models.resnet_models import ResNetLinear, ResNetMultiHead
from torch.utils.data import DataLoader

N = 12
print(f"[1/5] Writing {N} random throwaway images + labels to {CUSTOM_DIR} ...")
rng = np.random.default_rng(0)
rows = []
for i in range(N):
    arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    fname = f"_smoke_{i:02d}.jpg"
    Image.fromarray(arr).save(CUSTOM_DIR / fname)
    rows.append({"filename": fname,
                 **{a: int(rng.integers(1, 6)) for a in AXES}})
pd.DataFrame(rows).to_csv(CUSTOM_DIR / "labels.csv", index=False)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[2/5] Device: {dev}")

ds = CustomAxisDataset(train=True)
loader = DataLoader(ds, batch_size=4, shuffle=True)

def tiny_train(model, name, epochs=2):
    model = model.to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        print(f"   [{name}] epoch {ep+1}/{epochs}  loss {loss.item():.3f}")
    return model

print("[3/5] B2 forward/backward ...")
tiny_train(ResNetLinear(), "B2")

print("[4/5] M1 forward/backward + save checkpoint ...")
m1 = tiny_train(ResNetMultiHead(), "M1")
ckpt = CHECKPOINT_DIR / "m1_multihead.pt"
torch.save(m1.state_dict(), ckpt)

print("[5/5] Grad-CAM + feedback on one image ...")
from src.gradcam_utils import PhotoMentor
mentor = PhotoMentor(ckpt, device=dev)
res = mentor.analyse(Image.open(CUSTOM_DIR / "_smoke_00.jpg"))
print("   scores:", {k: round(v, 2) for k, v in res["scores"].items()})
print("   heatmap shapes:", {k: v.shape for k, v in res["heatmaps"].items()})
print("   feedback:", res["feedback"])

print("\nPIPELINE OK  - data loading, both models, Grad-CAM and feedback all run.")
print("Remember to delete the _smoke_* files + overwrite labels.csv before real data.")
