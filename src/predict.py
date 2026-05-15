"""
Single-image inference script.

Usage:
    python src/predict.py --image "path/to/image.jpg"
    python src/predict.py --image "FMCG dataset/dataset/images/aqua (1).jpg"
"""

import os
import sys
import argparse
import torch
from PIL import Image

# Adding project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, K_FOLDS, CLASS_NAMES, NUM_CLASSES
from src.dataset import get_val_transforms
from src.model import build_model
from src.utils import get_device


def load_ensemble(device):
    """Loading all fold models for ensemble prediction."""
    models = []
    for fold in range(K_FOLDS):
        model_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold + 1}.pth")
        if os.path.exists(model_path):
            model = build_model()
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)
    return models


@torch.no_grad()
def predict_single(image_path: str, models: list, device):
    """
    Predicting class for a single image using ensemble.
    
    Returns:
        predicted_class (str), confidence (float), all_probs (dict)
    """
    # Loading and preprocessing
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

    # Ensemble prediction
    avg_probs = torch.zeros(1, NUM_CLASSES).to(device)
    for model in models:
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        avg_probs += probs
    avg_probs /= len(models)

    # Get prediction
    confidence, pred_idx = avg_probs.max(dim=1)
    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence = confidence.item()

    # All class probabilities
    all_probs = {CLASS_NAMES[i]: avg_probs[0][i].item() for i in range(NUM_CLASSES)}

    return predicted_class, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description="FMCG Product Classifier — Single Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    device = get_device()

    print("Loading ensemble models...")
    models = load_ensemble(device)
    if not models:
        print(" No models found. Run training first!")
        return
    print(f" Loaded {len(models)} models")

    print(f"\nPredicting: {args.image}")
    predicted_class, confidence, all_probs = predict_single(args.image, models, device)

    print(f"\n{'='*40}")
    print(f"  Predicted Class: {predicted_class.upper()}")
    print(f"  Confidence:      {confidence*100:.2f}%")
    print(f"{'='*40}")
    print(f"\n  All probabilities:")
    for cls_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls_name:12s} {prob*100:6.2f}% {bar}")


if __name__ == "__main__":
    main()
