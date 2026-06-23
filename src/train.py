"""Training: pretrain on AVA, then fine-tune the 4 axes on the custom set.

Two trainable models:
  B2 (ResNetLinear)     -- trained on AVA only (single score), baseline.
  M1 (ResNetMultiHead)  -- pretrained on AVA via head_10, then its four axis
                           heads are fine-tuned on the custom labelled set.

Run order in Colab:
    python -m src.train --stage ava        # trains B2 + M1 backbone-head on AVA
    python -m src.train --stage finetune   # fine-tunes M1's 4 axis heads

Checkpoints land in results/checkpoints/.
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import (
    SEED, BATCH_SIZE, NUM_WORKERS, DEVICE, CHECKPOINT_DIR,
    AVA_EPOCHS, AVA_LR, FINETUNE_EPOCHS, FINETUNE_LR, VAL_FRACTION,
)
from .dataset import AVADataset, CustomAxisDataset
from .models.resnet_models import ResNetLinear, ResNetMultiHead


def _device() -> str:
    return DEVICE if torch.cuda.is_available() else "cpu"


def _seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def _loaders(dataset, val_fraction):
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(SEED)
    tr, va = random_split(dataset, [n_train, n_val], generator=g)
    return (DataLoader(tr, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
            DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS))


def _run_epoch(model, loader, loss_fn, dev, opt=None, forward="default"):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model.forward_pretrain(x) if forward == "pretrain" else model(x)
        loss = loss_fn(pred, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / n


# ----------------------------------------------------------------------------
# Stage 1: AVA pretraining
# ----------------------------------------------------------------------------
def train_ava():
    _seed(); dev = _device()
    train_ds = AVADataset(train=True)
    tr, va = _loaders(train_ds, VAL_FRACTION)
    loss_fn = nn.MSELoss()

    # --- B2 baseline ---
    print("=== Training B2 (ResNetLinear) on AVA ===")
    b2 = ResNetLinear(score_range=(1.0, 10.0)).to(dev)
    opt = torch.optim.Adam([p for p in b2.parameters() if p.requires_grad], lr=AVA_LR)
    for ep in range(AVA_EPOCHS):
        tl = _run_epoch(b2, tr, loss_fn, dev, opt)
        vl = _run_epoch(b2, va, loss_fn, dev)
        print(f"[B2] epoch {ep+1}/{AVA_EPOCHS}  train {tl:.3f}  val {vl:.3f}")
    torch.save(b2.state_dict(), CHECKPOINT_DIR / "b2_resnet_linear.pt")

    # --- M1 backbone-head pretraining (single AVA score) ---
    print("=== Pretraining M1 (ResNetMultiHead) head_10 on AVA ===")
    m1 = ResNetMultiHead().to(dev)
    opt = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], lr=AVA_LR)
    for ep in range(AVA_EPOCHS):
        tl = _run_epoch(m1, tr, loss_fn, dev, opt, forward="pretrain")
        vl = _run_epoch(m1, va, loss_fn, dev, forward="pretrain")
        print(f"[M1-pre] epoch {ep+1}/{AVA_EPOCHS}  train {tl:.3f}  val {vl:.3f}")
    torch.save(m1.state_dict(), CHECKPOINT_DIR / "m1_pretrained.pt")
    print(f"Saved B2 and M1-pretrained checkpoints to {CHECKPOINT_DIR}")


# ----------------------------------------------------------------------------
# Stage 2: custom 4-axis fine-tuning
# ----------------------------------------------------------------------------
def finetune():
    _seed(); dev = _device()
    m1 = ResNetMultiHead().to(dev)
    ckpt = CHECKPOINT_DIR / "m1_pretrained.pt"
    if ckpt.exists():
        m1.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"Loaded pretrained weights from {ckpt}")
    else:
        print("WARNING: no AVA-pretrained checkpoint found; fine-tuning from ImageNet only.")

    ds = CustomAxisDataset(train=True)
    tr, va = _loaders(ds, VAL_FRACTION)
    loss_fn = nn.MSELoss()
    params = [p for p in m1.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=FINETUNE_LR)

    best = float("inf")
    for ep in range(FINETUNE_EPOCHS):
        tl = _run_epoch(m1, tr, loss_fn, dev, opt)
        vl = _run_epoch(m1, va, loss_fn, dev)
        print(f"[M1-ft] epoch {ep+1}/{FINETUNE_EPOCHS}  train {tl:.3f}  val {vl:.3f}")
        if vl < best:
            best = vl
            torch.save(m1.state_dict(), CHECKPOINT_DIR / "m1_multihead.pt")
    print(f"Best val MSE {best:.3f}. Saved -> {CHECKPOINT_DIR / 'm1_multihead.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["ava", "finetune"], required=True)
    args = p.parse_args()
    if args.stage == "ava":
        train_ava()
    else:
        finetune()
