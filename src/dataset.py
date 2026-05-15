"""
Dataset module: automatic label extraction, data loading, and augmentation.

Labels are automatically extracted from filenames (e.g., "aqua (1).jpg" → class "aqua").
This eliminates the need for manual annotation — a key strategy for reducing labeling effort.
"""

import os
import re
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.config import (
    DATASET_DIR, OUTPUT_DIR, CLASS_TO_IDX, CLASS_NAMES,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    COLOR_JITTER, ROTATION_DEGREES, RANDOM_ERASING_P, AFFINE_TRANSLATE,
)


# ============================================================
# Automatic Label Extraction

def extract_labels_from_filenames(dataset_dir: str = DATASET_DIR) -> list:
    """
    Automatically extract labels from image filenames.
    
    Naming convention: 'category (N).jpg' → label = 'category'
    This removes the need for manual annotation entirely.
    
    Returns:
        List of (filepath, label_idx) tuples
    """
    samples = []
    pattern = re.compile(r"^([a-zA-Z]+)\s*\(\d+\)\.jpg$", re.IGNORECASE)

    for fname in sorted(os.listdir(dataset_dir)):
        match = pattern.match(fname)
        if match:
            category = match.group(1).lower()
            if category in CLASS_TO_IDX:
                filepath = os.path.join(dataset_dir, fname)
                label_idx = CLASS_TO_IDX[category]
                samples.append((filepath, label_idx))
            else:
                print(f"   Unknown category '{category}' in file: {fname}")

    print(f"  Auto-labeled {len(samples)} images across {len(CLASS_NAMES)} classes")
    return samples


def save_labels_csv(samples: list, output_path: str = None):
    """Save the auto-generated labels to a CSV file for documentation."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "labels.csv")
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label_idx", "label_name"])
        for filepath, label_idx in samples:
            writer.writerow([
                os.path.basename(filepath),
                label_idx,
                CLASS_NAMES[label_idx],
            ])
    print(f"  Saved labels to: {output_path}")


# ============================================================
# Data Augmentation Transforms

def get_train_transforms():
    """
    Aggressive augmentation pipeline for training.
    
    These transforms effectively multiply the dataset by simulating
    real-world variations: different angles, lighting, occlusions, etc.
    This is critical for achieving high accuracy with only 25 images/class.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.RandomAffine(
            degrees=0,
            translate=AFFINE_TRANSLATE,
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(**COLOR_JITTER),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=RANDOM_ERASING_P, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """Minimal transforms for validation — no augmentation, just resize + normalize."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# Dataset Class

class FMCGDataset(Dataset):
    """
    PyTorch Dataset for FMCG product images.
    
    Supports automatic label extraction and configurable augmentation.
    """

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples: List of (filepath, label_idx) tuples
            transform: torchvision transforms to apply
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Mixup Augmentation

def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation: creates virtual training examples by blending
    pairs of images and their labels. This regularizes the model and
    improves generalization — especially important with small datasets.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
    
    Args:
        x: batch of images (B, C, H, W)
        y: batch of labels (B,)
        alpha: mixup interpolation strength
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss as weighted combination of two targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
