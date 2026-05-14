"""
Configuration file for FMCG Product Classification Project.
All hyperparameters, paths, and settings are centralized here.
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "FMCG dataset", "dataset", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Create output directories
for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Dataset
# ============================================================
CLASS_NAMES = ["aqua", "chitato", "pepsodent", "shampoo"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ============================================================
# Model
# ============================================================
MODEL_NAME = "efficientnet_b0"       # timm model name
PRETRAINED = True                     # Use ImageNet pretrained weights
DROPOUT_RATE = 0.3                    # Dropout in classifier head
HIDDEN_DIM = 512                      # Hidden layer size in classifier head

# ============================================================
# Training
# ============================================================
SEED = 42
IMAGE_SIZE = 224                      # EfficientNet-B0 input size
BATCH_SIZE = 8                        # Small batch for 100 images
NUM_WORKERS = 0                       # 0 for macOS compatibility
K_FOLDS = 5                           # Stratified K-Fold CV

# Phase 1: Train classifier head only (backbone frozen)
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4

# Phase 2: Fine-tune backbone + head (progressive unfreezing)
PHASE2_EPOCHS = 20
PHASE2_LR = 1e-4
PHASE2_WEIGHT_DECAY = 1e-4

# Regularization
LABEL_SMOOTHING = 0.1                 # Label smoothing for CrossEntropy
MIXUP_ALPHA = 0.2                     # Mixup augmentation alpha

# Early stopping
EARLY_STOPPING_PATIENCE = 7

# ============================================================
# Augmentation
# ============================================================
# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation parameters
COLOR_JITTER = {
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.3,
    "hue": 0.1,
}
ROTATION_DEGREES = 15
RANDOM_ERASING_P = 0.2
AFFINE_TRANSLATE = (0.1, 0.1)
