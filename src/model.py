"""
Model architecture: EfficientNet-B0 with custom classification head.

Transfer Learning Strategy:
- Use EfficientNet-B0 pre-trained on ImageNet (1.2M images, 1000 classes)
- The pre-trained backbone already knows edges, textures, shapes, colors
- Replace the classification head for our 4-class FMCG problem
- Progressive unfreezing: train head first, then fine-tune deeper layers
"""

import torch
import torch.nn as nn
import timm

from src.config import MODEL_NAME, PRETRAINED, NUM_CLASSES, DROPOUT_RATE, HIDDEN_DIM


class FMCGClassifier(nn.Module):
    """
    FMCG Product Classifier based on EfficientNet-B0.
    
    Architecture:
        EfficientNet-B0 backbone (frozen/unfrozen) →
        AdaptiveAvgPool →
        Linear(1280, 512) → BatchNorm → ReLU → Dropout(0.3) →
        Linear(512, 4)
    """

    def __init__(self):
        super().__init__()

        # Load pre-trained EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=0,         # Remove original classifier
            global_pool="avg",     # Keep global average pooling
        )

        # Get the feature dimension from backbone
        self.feature_dim = self.backbone.num_features  # 1280 for efficientnet_b0

        # Custom classification head with regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier head with Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass: backbone features → classifier."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """
        Freeze all backbone parameters (Phase 1).
        Only the classifier head will be trained.
        This prevents catastrophic forgetting of pre-trained features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  🧊 Backbone FROZEN — training classifier head only")

    def unfreeze_backbone(self, unfreeze_from: int = -2):
        """
        Unfreeze the last N blocks of the backbone (Phase 2).
        This allows fine-tuning of higher-level features while
        preserving low-level feature detectors.
        
        For EfficientNet-B0, blocks are in self.backbone.blocks
        """
        # First, keep everything frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[unfreeze_from:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the final batch norm and head-related layers
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        n_unfrozen = abs(unfreeze_from)
        print(f"  🔥 Backbone PARTIALLY UNFROZEN — last {n_unfrozen} blocks + head trainable")

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  📐 Parameters: {total:,} total | {trainable:,} trainable | {frozen:,} frozen")
        return total, trainable


def build_model():
    """Factory function to create and return the FMCG classifier."""
    model = FMCGClassifier()
    print(f"  ✅ Built {MODEL_NAME} with {NUM_CLASSES}-class head")
    return model
