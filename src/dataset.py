"""PyTorch Datasets and image transforms.

Two datasets:
  * AVADataset      -> (image, mean_score in 1..10)   for pretraining a head
  * CustomAxisDataset -> (image, [comp, expo, sharp, bg] in 1..5) for fine-tune

The custom set expects a CSV `labels.csv` in data/custom/ with columns:
    filename, composition, exposure, sharpness, background
(see data/custom/labels_template.csv). Images sit next to it in data/custom/.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import (
    AXES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, AVA_DIR, CUSTOM_DIR,
)


def build_transforms(train: bool) -> transforms.Compose:
    """Standard ImageNet preprocessing; light augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class AVADataset(Dataset):
    """AVA images with a single mean aesthetic score target (1..10)."""

    def __init__(self, manifest_csv: Path | None = None, train: bool = True):
        manifest_csv = manifest_csv or (AVA_DIR / "ava_subsample.csv")
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.img_dir = AVA_DIR / "images"
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([row["mean_score"]], dtype=torch.float32)
        return x, y


class CustomAxisDataset(Dataset):
    """Beginner photos labelled on the four 1..5 axes."""

    def __init__(self, labels_csv: Path | None = None, train: bool = True):
        labels_csv = labels_csv or (CUSTOM_DIR / "labels.csv")
        self.df = pd.read_csv(labels_csv).reset_index(drop=True)
        self.img_dir = CUSTOM_DIR
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([row[a] for a in AXES], dtype=torch.float32)
        return x, y
