"""
Evaluation module: generates metrics, confusion matrix, and sample predictions.

Uses ensemble prediction from all K-fold models for robust evaluation.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    K_FOLDS, BATCH_SIZE, NUM_WORKERS, MODEL_DIR, REPORTS_DIR,
    CLASS_NAMES, NUM_CLASSES, SEED,
)
from src.dataset import (
    extract_labels_from_filenames, get_val_transforms, FMCGDataset,
)
from src.model import build_model
from src.utils import (
    set_seed, get_device, plot_confusion_matrix, plot_sample_predictions,
)


def load_fold_models(device):
    """Load all saved fold models."""
    models = []
    for fold in range(K_FOLDS):
        model_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold + 1}.pth")
        if os.path.exists(model_path):
            model = build_model()
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)
            print(f"  ✅ Loaded model: Fold {fold + 1}")
        else:
            print(f"  ⚠️  Model not found: {model_path}")
    return models


@torch.no_grad()
def ensemble_predict(models, loader, device):
    """
    Ensemble prediction: average softmax probabilities across all fold models.
    
    This ensemble approach further improves accuracy by combining the
    strengths of models trained on different data splits.
    """
    all_probs = []
    all_labels = []
    all_images = []

    for images, labels in loader:
        images = images.to(device)
        batch_probs = torch.zeros(images.size(0), NUM_CLASSES).to(device)

        for model in models:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            batch_probs += probs

        batch_probs /= len(models)  # Average probabilities

        all_probs.append(batch_probs.cpu())
        all_labels.extend(labels.numpy())
        all_images.append(images.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_labels = np.array(all_labels)

    pred_labels = all_probs.argmax(dim=1).numpy()
    pred_confidences = all_probs.max(dim=1).values.numpy()

    return all_images, all_labels, pred_labels, pred_confidences, all_probs


def main():
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("  FMCG Product Classification — Evaluation")
    print("=" * 60)

    set_seed(SEED)
    device = get_device()

    # Load all fold models
    print("\n📦 Loading ensemble models...")
    models = load_fold_models(device)

    if not models:
        print("❌ No models found. Run training first!")
        return

    # Load full dataset with val transforms (no augmentation)
    print("\n📁 Loading dataset...")
    all_samples = extract_labels_from_filenames()

    dataset = FMCGDataset(all_samples, transform=get_val_transforms())
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # Ensemble predictions on full dataset
    print("\n🔮 Running ensemble predictions...")
    images, true_labels, pred_labels, pred_confidences, all_probs = ensemble_predict(
        models, loader, device
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n  🎯 Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  📊 F1 Score (macro):     {f1_macro:.4f}")
    print(f"  📊 F1 Score (weighted):  {f1_weighted:.4f}")
    print(f"  📊 Mean Confidence:      {pred_confidences.mean():.4f}")

    target_met = "✅ TARGET MET" if accuracy >= 0.95 else "❌ BELOW TARGET"
    print(f"\n  🎯 95% Target: {target_met}")

    # Classification report
    report = classification_report(
        true_labels, pred_labels,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(f"\n  📋 Classification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"  📋 Confusion Matrix:")
    print(f"  {cm}")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    # Save classification report
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("FMCG Product Classification — Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"F1 Score (macro):     {f1_macro:.4f}\n")
        f.write(f"F1 Score (weighted):  {f1_weighted:.4f}\n")
        f.write(f"Mean Confidence:      {pred_confidences.mean():.4f}\n")
        f.write(f"Ensemble Models:      {len(models)}\n")
        f.write(f"Total Samples:        {len(true_labels)}\n")
        f.write(f"\n95% Target: {target_met}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"\nClassification Report:\n{report}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"  💾 Saved report: {report_path}")

    # Save metrics as JSON
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "mean_confidence": float(pred_confidences.mean()),
        "num_models": len(models),
        "num_samples": len(true_labels),
        "target_met": accuracy >= 0.95,
        "per_class": {},
    }
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = true_labels == i
        cls_acc = (pred_labels[mask] == i).mean()
        metrics["per_class"][cls_name] = {
            "accuracy": float(cls_acc),
            "count": int(mask.sum()),
        }

    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  💾 Saved metrics: {metrics_path}")

    # Plot confusion matrix
    plot_confusion_matrix(cm)

    # Plot sample predictions
    plot_sample_predictions(images, true_labels, pred_labels, pred_confidences)

    print(f"\n✅ Evaluation complete! All outputs saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
