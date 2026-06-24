"""Evaluation metrics on the HELD-OUT validation split.

Reproduces the exact train/val split used in src/train.py (same SEED and
VAL_FRACTION), then evaluates M1 on the validation portion only -- not on the
whole dataset -- so the numbers are not inflated by training leakage.

Reports per axis: MAE, Spearman rho (tests H2: rho >= 0.5), and 95% bootstrap
confidence intervals. Writes results/metrics.csv.

Usage:  python -m src.evaluate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, random_split

from .config import (
    AXES, DEVICE, CHECKPOINT_DIR, RESULTS_DIR, N_BOOTSTRAP, CI, BATCH_SIZE,
    SEED, VAL_FRACTION,
)
from .dataset import CustomAxisDataset
from .models.resnet_models import ResNetMultiHead


def _device() -> str:
    return DEVICE if torch.cuda.is_available() else "cpu"


def held_out_split():
    """Recreate train.py's split and return the validation Subset.

    random_split assigns indices from the dataset length and the generator
    seed only (the transform is irrelevant), so using train=False here yields
    exactly the same validation indices the training run held out.
    """
    full = CustomAxisDataset(train=False)
    n_val = max(1, int(len(full) * VAL_FRACTION))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val = random_split(full, [n_train, n_val], generator=g)
    return val


def bootstrap_ci(fn, a, b, n=N_BOOTSTRAP, ci=CI, seed=0):
    rng = np.random.default_rng(seed)
    stats, idx = [], np.arange(len(a))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        val = fn(a[s], b[s])
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return float("nan"), float("nan")
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def _spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return spearmanr(a, b).correlation


@torch.no_grad()
def collect_predictions(model, loader, dev):
    preds, gts = [], []
    model.eval()
    for x, y in loader:
        preds.append(model(x.to(dev)).cpu().numpy())
        gts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def evaluate_m1():
    dev = _device()
    model = ResNetMultiHead().to(dev)
    ckpt = CHECKPOINT_DIR / "m1_multihead.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)

    val = held_out_split()
    loader = DataLoader(val, BATCH_SIZE, shuffle=False)
    preds, gts = collect_predictions(model, loader, dev)
    n_held = len(val)
    print(f"Evaluating M1 on the held-out validation split (N={n_held}).\n")

    rows = []
    for i, axis in enumerate(AXES):
        p, g = preds[:, i], gts[:, i]
        m = mae(p, g)
        rho = _spearman(p, g)
        m_lo, m_hi = bootstrap_ci(mae, p, g)
        rho_lo, rho_hi = bootstrap_ci(_spearman, p, g)
        rows.append({
            "axis": axis, "N": n_held, "MAE": round(m, 3),
            "MAE_CI": f"[{m_lo:.3f}, {m_hi:.3f}]",
            "Spearman_rho": (round(rho, 3) if not np.isnan(rho) else "nan"),
            "rho_CI": f"[{rho_lo:.3f}, {rho_hi:.3f}]",
            "H2_pass(rho>=0.5)": (bool(rho >= 0.5) if not np.isnan(rho) else "n/a"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {RESULTS_DIR / 'metrics.csv'}")
    return df


def rater_agreement(rater_a, rater_b):
    """Cohen's quadratic-weighted kappa on two raters' 1..5 labels (one axis)."""
    return cohen_kappa_score(np.round(rater_a), np.round(rater_b),
                             weights="quadratic")


if __name__ == "__main__":
    evaluate_m1()
