"""Training: train B2 and M1 directly on the custom 4-axis dataset.

AVA large-scale pretraining was descoped (the available AVA mirror ships
images only, no score labels). Both models start from an ImageNet-pretrained
ResNet-50 backbone and learn the four 1..5 axes from the custom labelled set.

  B2 (ResNetLinear)    -- frozen backbone + one linear layer -> 4 axes.
  M1 (ResNetMultiHead) -- frozen backbone + a deeper head per axis -> 4 axes.

Run in Colab:
    python -m src.train --model both     # trains B2 and M1
    python -m src.train --model m1        # just M1

Needs data/custom/labels.csv + the images. Checkpoints -> results/checkpoints/.
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import (
    SEED, BATCH_SIZE, NUM_WORKERS, DEVICE, CHECKPOINT_DIR,
    FINETUNE_EPOCHS, FINETUNE_LR, VAL_FRACTION,
)
from .dataset import CustomAxisDataset
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


def _run_epoch(model, loader, loss_fn, dev, opt=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x)
        loss = loss_fn(pred, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / n


def _train_one(model, name, tr, va, dev, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR):
    loss_fn = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    best, best_path = float("inf"), CHECKPOINT_DIR / f"{name}.pt"
    print(f"=== Training {name} on custom 4-axis data ===")
    for ep in range(epochs):
        tl = _run_epoch(model, tr, loss_fn, dev, opt)
        vl = _run_epoch(model, va, loss_fn, dev)
        print(f"[{name}] epoch {ep+1}/{epochs}  train {tl:.3f}  val {vl:.3f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), best_path)
    print(f"[{name}] best val MSE {best:.3f}  ->  {best_path}\n")
    return best


def main(which: str):
    _seed(); dev = _device()
    ds = CustomAxisDataset(train=True)
    if len(ds) < 8:
        print(f"WARNING: only {len(ds)} labelled photos found. Add more to "
              f"data/custom/ (aim for 30-50 for a usable MVP).")
    tr, va = _loaders(ds, VAL_FRACTION)

    if which in ("both", "b2"):
        _train_one(ResNetLinear().to(dev), "b2_resnet_linear", tr, va, dev)
    if which in ("both", "m1"):
        _train_one(ResNetMultiHead().to(dev), "m1_multihead", tr, va, dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["both", "b2", "m1"], default="both")
    args = p.parse_args()
    main(args.model)
