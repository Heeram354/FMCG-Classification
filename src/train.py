"""
Training pipeline with Stratified K-Fold Cross Validation.

Training Strategy:
    Phase 1 — Classifier Head Training (backbone frozen):
        - Only the custom classification head is trained
        - Higher learning rate (1e-3) since these are random weights
        - 10 epochs to converge the head
    
    Phase 2 — Fine-tuning (last 2 backbone blocks unfrozen):
        - Lower learning rate (1e-4) to avoid destroying pre-trained features
        - Cosine annealing LR schedule for smooth convergence
        - 20 epochs with early stopping
        - Mixup augmentation for regularization

    5-Fold Stratified CV ensures every image is used for both training
    and validation, giving a robust accuracy estimate.
"""

import os
import sys
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SEED, K_FOLDS, BATCH_SIZE, NUM_WORKERS,
    PHASE1_EPOCHS, PHASE1_LR, PHASE1_WEIGHT_DECAY,
    PHASE2_EPOCHS, PHASE2_LR, PHASE2_WEIGHT_DECAY,
    LABEL_SMOOTHING, MIXUP_ALPHA, EARLY_STOPPING_PATIENCE,
    MODEL_DIR, REPORTS_DIR, CLASS_NAMES,
)
from src.dataset import (
    extract_labels_from_filenames, save_labels_csv,
    get_train_transforms, get_val_transforms,
    FMCGDataset, mixup_data, mixup_criterion,
)
from src.model import build_model
from src.utils import set_seed, get_device, plot_training_history, plot_class_distribution


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup and MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            # For accuracy, use the dominant label
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item()
                        + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model. Returns average loss, accuracy, and predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_fold(fold, train_indices, val_indices, all_samples, device):
    """Train a single fold with Phase 1 + Phase 2."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold + 1}/{K_FOLDS}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_indices)} samples | Val: {len(val_indices)} samples")

    # Create datasets with appropriate transforms
    train_dataset = FMCGDataset(
        [all_samples[i] for i in train_indices],
        transform=get_train_transforms(),
    )
    val_dataset = FMCGDataset(
        [all_samples[i] for i in val_indices],
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # Build fresh model for this fold
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    # --------------------------------------------------------
    # Phase 1: Train classifier head only (backbone frozen)
    # --------------------------------------------------------
    print(f"\n  📌 Phase 1: Training classifier head ({PHASE1_EPOCHS} epochs)")
    model.freeze_backbone()
    model.count_parameters()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=PHASE1_WEIGHT_DECAY,
    )

    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=False
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"    Epoch {epoch+1:2d}/{PHASE1_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    # --------------------------------------------------------
    # Phase 2: Fine-tune with unfrozen backbone
    # --------------------------------------------------------
    print(f"\n  📌 Phase 2: Fine-tuning backbone ({PHASE2_EPOCHS} epochs)")
    model.unfreeze_backbone(unfreeze_from=-2)
    model.count_parameters()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR,
        weight_decay=PHASE2_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)
    patience_counter = 0

    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=True
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " ⭐ best"
        else:
            patience_counter += 1

        print(f"    Epoch {epoch+1:2d}/{PHASE2_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}{marker}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"    ⏹ Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
            break

    # Save best model for this fold
    model_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold+1}.pth")
    torch.save(best_model_state, model_path)
    print(f"\n  💾 Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  💾 Saved model: {model_path}")

    # Plot training curves
    plot_training_history(history, fold)

    return best_val_acc, history


def main():
    """Main training pipeline with K-Fold Cross Validation."""
    start_time = time.time()

    print("=" * 60)
    print("  FMCG Product Classification — Training Pipeline")
    print("=" * 60)

    # 1. Set seed for reproducibility
    set_seed(SEED)
    device = get_device()

    # 2. Load and auto-label dataset
    print("\n📁 Loading dataset...")
    all_samples = extract_labels_from_filenames()
    save_labels_csv(all_samples)

    # Extract labels for stratification
    all_labels = [label for _, label in all_samples]
    all_paths = [path for path, _ in all_samples]

    # Plot class distribution
    plot_class_distribution(all_labels)

    # Print dataset summary
    print(f"\n📊 Dataset Summary:")
    for cls_name in CLASS_NAMES:
        count = sum(1 for _, l in all_samples if CLASS_NAMES[l] == cls_name)
        print(f"    {cls_name}: {count} images")

    # 3. K-Fold Cross Validation
    print(f"\n🔄 Starting {K_FOLDS}-Fold Stratified Cross Validation...")
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    fold_accuracies = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        fold_acc, history = train_fold(fold, train_idx, val_idx, all_samples, device)
        fold_accuracies.append(fold_acc)
        all_histories.append(history)

    # 4. Summary
    elapsed = time.time() - start_time
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Per-Fold Accuracies:")
    for i, acc in enumerate(fold_accuracies):
        print(f"    Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  📈 Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"  📊 Std Deviation: {std_acc:.4f}")
    print(f"  ⏱  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    target_met = "✅ TARGET MET" if mean_acc >= 0.95 else "❌ BELOW TARGET"
    print(f"\n  🎯 95% Target: {target_met}")

    # Save results
    results = {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "training_time_seconds": elapsed,
        "k_folds": K_FOLDS,
        "target_met": bool(mean_acc >= 0.95),
    }
    results_path = os.path.join(REPORTS_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Saved results: {results_path}")


if __name__ == "__main__":
    main()
