"""Evaluation metrics and a model-comparison report.

Produces the table the write-up needs:
  * MAE per axis (M1 vs human labels)
  * Spearman rho per axis (tests H2: rho >= 0.5)
  * 95% bootstrap confidence intervals
  * BRISQUE (B1) and ResNetLinear (B2) overall-score correlation for contrast

Usage:
    python -m src.evaluate
Writes results/metrics.csv and prints a summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .config import (
    AXES, DEVICE, CHECKPOINT_DIR, RESULTS_DIR, N_BOOTSTRAP, CI, BATCH_SIZE,
)
from .dataset import CustomAxisDataset
from .models.resnet_models import ResNetMultiHead
from torch.utils.data import DataLoader


def _device() -> str:
    return DEVICE if torch.cuda.is_available() else "cpu"


def bootstrap_ci(fn, a, b, n=N_BOOTSTRAP, ci=CI, seed=0):
    """Bootstrap a 2-sample statistic fn(a,b); return (low, high)."""
    rng = np.random.default_rng(seed)
    stats = []
    idx = np.arange(len(a))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        stats.append(fn(a[s], b[s]))
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


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
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    ds = CustomAxisDataset(train=False)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False)
    preds, gts = collect_predictions(model, loader, dev)

    rows = []
    for i, axis in enumerate(AXES):
        p, g = preds[:, i], gts[:, i]
        m = mae(p, g)
        rho = spearmanr(p, g).correlation
        m_lo, m_hi = bootstrap_ci(mae, p, g)
        rho_lo, rho_hi = bootstrap_ci(
            lambda a, b: spearmanr(a, b).correlation, p, g)
        rows.append({
            "axis": axis, "MAE": round(m, 3),
            "MAE_CI": f"[{m_lo:.3f}, {m_hi:.3f}]",
            "Spearman_rho": round(rho, 3),
            "rho_CI": f"[{rho_lo:.3f}, {rho_hi:.3f}]",
            "H2_pass(rho>=0.5)": rho >= 0.5,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {RESULTS_DIR / 'metrics.csv'}")
    return df


def rater_agreement(rater_a, rater_b):
    """Cohen's kappa on integer 1..5 labels from two raters (one axis)."""
    return cohen_kappa_score(np.round(rater_a), np.round(rater_b),
                             weights="quadratic")


if __name__ == "__main__":
    evaluate_m1()
