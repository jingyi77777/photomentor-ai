"""Download a stratified 5,000-image slice of the AVA dataset.

Full AVA is ~255k images. For transfer learning a balanced 5k slice is plenty
and fits comfortably in Colab. We stream from the HuggingFace mirror
`Iceclear/AVA` so we never download the whole thing.

AVA ground truth = a distribution of votes for scores 1..10. We collapse that
to a single *mean aesthetic score* per image, then stratify the subsample
across 9 score bins so the model sees the full quality range.

Usage (in Colab or locally):
    python -m data.download_ava --n 5000

If the HuggingFace mirror is unavailable, see data/README.md for Kaggle
fallbacks (e.g. nicolacarrassi/ava-aesthetic-visual-assessment).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import AVA_DIR, AVA_SUBSAMPLE_N, SEED


def mean_score_from_votes(votes: list[int]) -> float:
    """AVA stores 10 vote counts (for scores 1..10). Return the weighted mean."""
    votes = np.asarray(votes, dtype=np.float64)
    if votes.sum() == 0:
        return float("nan")
    scores = np.arange(1, 11)
    return float((votes * scores).sum() / votes.sum())


def main(n: int = AVA_SUBSAMPLE_N) -> None:
    from datasets import load_dataset

    print(f"[1/3] Streaming AVA from HuggingFace (Iceclear/AVA)...")
    ds = load_dataset("Iceclear/AVA", split="train", streaming=True)

    # Bin edges for stratification across the 1..10 mean-score range.
    n_bins = 9
    per_bin = n // n_bins
    bin_counts = {b: 0 for b in range(n_bins)}

    img_dir = AVA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"[2/3] Collecting ~{per_bin} images per score-bin...")
    pbar = tqdm(total=n)
    for ex in ds:
        if len(rows) >= n:
            break
        votes = ex.get("votes") or ex.get("rating_counts")
        if votes is None:
            continue
        mean = mean_score_from_votes(votes)
        if np.isnan(mean):
            continue
        b = min(int(mean) - 1, n_bins - 1)          # mean 1.x -> bin 0, etc.
        if bin_counts[b] >= per_bin:
            continue

        img = ex["image"]
        if not isinstance(img, Image.Image):
            continue
        img = img.convert("RGB")
        img_id = ex.get("id", len(rows))
        fname = f"{img_id}.jpg"
        img.save(img_dir / fname, quality=90)

        rows.append({"filename": fname, "mean_score": round(mean, 4), "bin": b})
        bin_counts[b] += 1
        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    manifest = AVA_DIR / "ava_subsample.csv"
    df.to_csv(manifest, index=False)

    print(f"[3/3] Saved {len(df)} images -> {img_dir}")
    print(f"      Manifest -> {manifest}")
    print("\nScore-bin distribution:")
    print(df["bin"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=AVA_SUBSAMPLE_N)
    args = p.parse_args()
    np.random.seed(SEED)
    main(args.n)
