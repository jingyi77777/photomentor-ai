"""PyTorch Dataset and image transforms.

CustomAxisDataset -> (image, [comp, expo, sharp, bg] in 1..5).

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

from .config import AXES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CUSTOM_DIR


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


class CustomAxisDataset(Dataset):
    """Photos labelled on the four 1..5 axes."""

    def __init__(self, labels_csv: Path | None = None,
                 img_dir: Path | None = None, train: bool = True):
        labels_csv = labels_csv or (CUSTOM_DIR / "labels.csv")
        self.df = pd.read_csv(labels_csv).reset_index(drop=True)
        self.img_dir = img_dir or CUSTOM_DIR
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([row[a] for a in AXES], dtype=torch.float32)
        return x, y
