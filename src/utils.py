"""
Utility functions for reproducibility, visualization, and logging.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import SEED, PLOTS_DIR, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


def set_seed(seed: int = SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device (MPS for M1 Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M1 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slower)")
    return device


def plot_training_history(history: dict, fold: int, save: bool = True):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"Fold {fold + 1} — Loss Curves", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="95% Target")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(f"Fold {fold + 1} — Accuracy Curves", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"training_fold_{fold + 1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved training plot: {path}")
    plt.close()


def plot_confusion_matrix(cm, save: bool = True, filename: str = "confusion_matrix.png"):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 14},
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved confusion matrix: {path}")
    plt.close()


def plot_sample_predictions(images, true_labels, pred_labels, pred_probs,
                            save: bool = True, filename: str = "sample_predictions.png"):
    """Plot a grid of sample predictions with true/predicted labels."""
    n = min(len(images), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        img = img * std + mean  # Denormalize
        img = np.clip(img, 0, 1)

        true_name = CLASS_NAMES[true_labels[i]]
        pred_name = CLASS_NAMES[pred_labels[i]]
        confidence = pred_probs[i] * 100

        correct = true_labels[i] == pred_labels[i]
        color = "green" if correct else "red"

        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name} ({confidence:.1f}%)",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        axes[i].axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved sample predictions: {path}")
    plt.close()


def plot_class_distribution(labels, save: bool = True, filename: str = "class_distribution.png"):
    """Plot the class distribution of the dataset."""
    counts = [labels.count(i) for i in range(len(CLASS_NAMES))]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(CLASS_NAMES, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                   edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha="center", va="bottom", fontsize=13, fontweight="bold")
    plt.xlabel("Product Category", fontsize=13)
    plt.ylabel("Number of Images", fontsize=13)
    plt.title("Dataset Class Distribution", fontsize=15, fontweight="bold")
    plt.ylim(0, max(counts) + 5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f" Saved class distribution: {path}")
    plt.close()
