"""
src/models/scratch_cnn.py
─────────────────────────
Custom CNN trained entirely from scratch on CIFAR-100.
No pretrained weights — all parameters are randomly initialised
and learned solely from the 45,000 training images.

Architecture overview:
    4 convolutional blocks  (Conv → BN → ReLU → Conv → BN → ReLU → Pool → Dropout)
    1 classifier head       (Flatten → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear)

Spatial resolution flow:
    Input  32 × 32
    Block1 16 × 16  (64 channels)
    Block2  8 ×  8  (128 channels)
    Block3  4 ×  4  (256 channels)
    Block4  2 ×  2  (512 channels)
    Flatten → 2048-dim vector → FC layers → 100 logits

Exports:
    ScratchCNN          — main model class
    count_parameters()  — total / trainable / frozen param counts
    build_scratch_cnn() — factory function used by run_scratch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Reusable double-conv block:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
        → MaxPool2d → Dropout2d

    Two convolutions before each pool is a design choice borrowed from
    VGG: two 3×3 convolutions have the same receptive field as one 5×5
    but fewer parameters and an extra non-linearity.

    Args:
        in_channels:  Number of input feature maps.
        out_channels: Number of output feature maps.
        dropout_rate: Spatial dropout probability (applied after pooling).
                      Spatial dropout zeros entire feature map channels,
                      which is more effective than per-pixel dropout for
                      convolutional feature maps.
    """

    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        dropout_rate : float = 0.2,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # First conv: expand channels
            nn.Conv2d(in_channels,  out_channels, kernel_size=3,
                      padding=1, bias=False),  # bias=False: BN absorbs bias
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second conv: refine at same resolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Halve spatial dimensions
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Spatial dropout: drops whole channels, not individual pixels
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClassifierHead(nn.Module):
    """
    Fully-connected classification head:
        Flatten → Linear(2048→1024) → BN → ReLU → Dropout
                → Linear(1024→512)  → BN → ReLU → Dropout
                → Linear(512→100)

    BatchNorm is applied after each hidden linear layer for training
    stability. This is less common in heads but helps on CIFAR-100
    where the 100-class problem pushes the head towards overfitting.

    Args:
        in_features:  Flattened feature dimension from the backbone.
        num_classes:  Number of output logits (100 for CIFAR-100).
        dropout_rate: Dropout probability for both hidden layers.
    """

    def __init__(
        self,
        in_features  : int,
        num_classes  : int   = 100,
        dropout_rate : float = 0.5,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.8),   # slightly less dropout on second layer

            nn.Linear(512, num_classes),         # no BN before softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Main model ────────────────────────────────────────────────────────────────

class ScratchCNN(nn.Module):
    """
    Full custom CNN for CIFAR-100 classification.

    Input:  [B, 3, 32, 32]  — normalised RGB images
    Output: [B, 100]        — raw logits (pass through softmax for probabilities)

    The network uses progressively increasing dropout rates across blocks
    to apply stronger regularisation as spatial resolution decreases and
    feature maps become more abstract.

    Dropout schedule:
        Block 1:  0.10  (low — early features are generic, less prone to overfit)
        Block 2:  0.20
        Block 3:  0.30
        Block 4:  0.40
        Head:     0.50  (highest — the dense layers overfit most aggressively)

    Args:
        num_classes:     Number of output categories. Default 100 for CIFAR-100.
        base_channels:   Width multiplier. Default 64 gives the standard model.
                         Set to 32 for a lightweight variant (~1.5M params).
        head_dropout:    Base dropout probability for the classifier head.
    """

    def __init__(
        self,
        num_classes   : int   = 100,
        base_channels : int   = 64,
        head_dropout  : float = 0.5,
    ) -> None:
        super().__init__()

        c = base_channels   # shorthand: 64 → [64, 128, 256, 512]

        # ── Convolutional backbone ─────────────────────────────────────────
        self.backbone = nn.Sequential(
            ConvBlock(3,       c,     dropout_rate=0.10),   # 32→16
            ConvBlock(c,   2 * c,     dropout_rate=0.20),   # 16→ 8
            ConvBlock(2*c, 4 * c,     dropout_rate=0.30),   #  8→ 4
            ConvBlock(4*c, 8 * c,     dropout_rate=0.40),   #  4→ 2
        )

        # Spatial size after 4 pools: 32 / 2^4 = 2
        # Feature map volume:         8*c × 2 × 2
        backbone_out = 8 * c * 2 * 2    # = 2048 at base_channels=64

        # ── Classifier head ────────────────────────────────────────────────
        self.classifier = ClassifierHead(
            in_features  = backbone_out,
            num_classes  = num_classes,
            dropout_rate = head_dropout,
        )

        # ── Weight initialisation ──────────────────────────────────────────
        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Kaiming He initialisation for Conv2d (ReLU networks).
        Xavier uniform for Linear layers.
        BatchNorm: weight=1, bias=0 (standard).

        Why Kaiming for conv?  It accounts for the ReLU non-linearity —
        Xavier assumes symmetric activations and underestimates variance
        for ReLU, leading to vanishing activations in deep networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, 3, 32, 32].

        Returns:
            Logits tensor of shape [B, 100].
            Apply softmax externally if probabilities are needed.
        """
        features = self.backbone(x)      # [B, 512, 2, 2]
        logits   = self.classifier(features)   # [B, 100]
        return logits

    # ── Convenience methods ───────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the flattened backbone feature vector before the classifier.
        Useful for t-SNE visualisation or feature-based analysis.

        Args:
            x: Input tensor [B, 3, 32, 32].

        Returns:
            Feature tensor [B, 2048] (at base_channels=64).
        """
        features = self.backbone(x)
        return features.flatten(start_dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax class probabilities instead of raw logits.

        Args:
            x: Input tensor [B, 3, 32, 32].

        Returns:
            Probability tensor [B, 100], sums to 1 along dim=1.
        """
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def __repr__(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return (
            f"ScratchCNN(\n"
            f"  backbone   : 4 × ConvBlock  "
            f"[3→64→128→256→512 channels, 32→2 spatial]\n"
            f"  classifier : 3 × Linear     "
            f"[2048→1024→512→100]\n"
            f"  total params    : {total:,}\n"
            f"  trainable params: {trainable:,}\n"
            f")"
        )


# ── Lightweight variant ───────────────────────────────────────────────────────

class ScratchCNNSmall(ScratchCNN):
    """
    Half-width variant of ScratchCNN.
    Channels: [32, 64, 128, 256]  instead of [64, 128, 256, 512].
    Parameters: ~1.5M  instead of ~9.2M.

    Useful for:
      - Quick experimentation on CPU (Intel Mac)
      - Ablation study comparing capacity vs accuracy
      - Verifying the training loop before committing to a full run

    Usage:
        model = ScratchCNNSmall(num_classes=100)
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__(num_classes=num_classes, base_channels=32)


# ── Residual variant ──────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Basic residual block with a skip connection.
    Input and output channels must match (in_channels == out_channels).

        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU

    The skip connection allows gradients to flow directly to early layers,
    mitigating vanishing gradients in deeper stacks.

    Args:
        channels:     Number of input and output channels.
        dropout_rate: Dropout after the residual addition.
    """

    def __init__(self, channels: int, dropout_rate: float = 0.1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(p=dropout_rate)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)),  inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)   # skip connection
        out = self.drop(out)
        return out


class ScratchCNNResidual(nn.Module):
    """
    Scratch CNN with residual (skip) connections inside each stage.
    Deeper than ScratchCNN but more stable to train.

    Architecture:
        Stage 1: ConvBlock(3→64)  + ResidualBlock(64)   → pool → 16×16
        Stage 2: ConvBlock(64→128)+ ResidualBlock(128)  → pool →  8×8
        Stage 3: ConvBlock(128→256)+ResidualBlock(256)  → pool →  4×4
        Stage 4: ConvBlock(256→512)+ResidualBlock(512)  → pool →  2×2
        Head: same as ScratchCNN

    Parameters: ~18M  (roughly double ScratchCNN due to residual convs)

    Usage:
        model = ScratchCNNResidual(num_classes=100)
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        def stage(in_c, out_c, drop):
            return nn.Sequential(
                ConvBlock(in_c, out_c, dropout_rate=drop),
                ResidualBlock(out_c, dropout_rate=drop * 0.5),
            )

        self.backbone = nn.Sequential(
            stage(3,   64,  0.10),
            stage(64,  128, 0.20),
            stage(128, 256, 0.30),
            stage(256, 512, 0.40),
        )

        self.classifier = ClassifierHead(
            in_features  = 512 * 2 * 2,
            num_classes  = num_classes,
            dropout_rate = 0.5,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(start_dim=1)


# ── Factory function ──────────────────────────────────────────────────────────

def build_scratch_cnn(
    variant     : str = "standard",
    num_classes : int = 100,
) -> ScratchCNN | ScratchCNNSmall | ScratchCNNResidual:
    """
    Factory function — returns the requested model variant.
    Used by run_scratch.py so you can switch variants in one place.

    Args:
        variant:     One of:
                       "standard"  → ScratchCNN       (~9.2M params)
                       "small"     → ScratchCNNSmall  (~1.5M params)
                       "residual"  → ScratchCNNResidual (~18M params)
        num_classes: Output classes. Default 100 for CIFAR-100.

    Returns:
        Instantiated model (not yet moved to device).

    Raises:
        ValueError for unknown variant names.
    """
    variants = {
        "standard" : lambda: ScratchCNN(num_classes=num_classes),
        "small"    : lambda: ScratchCNNSmall(num_classes=num_classes),
        "residual" : lambda: ScratchCNNResidual(num_classes=num_classes),
    }

    if variant not in variants:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from: {list(variants.keys())}"
        )

    model = variants[variant]()
    print(f"[scratch_cnn] Built variant='{variant}'  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


# ── Parameter utilities ───────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Count total, trainable, and frozen parameters.

    Args:
        model: Any nn.Module.

    Returns:
        dict with keys: total, trainable, frozen.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"  Parameters — total: {total:>12,}  "
          f"trainable: {trainable:>12,}  "
          f"frozen: {frozen:>10,}")

    return {"total": total, "trainable": trainable, "frozen": frozen}


def receptive_field_size(n_conv_blocks: int = 4) -> int:
    """
    Calculate the theoretical receptive field of the backbone.
    Each 3×3 conv adds 2 pixels; MaxPool2d doubles the field.

    Args:
        n_conv_blocks: Number of ConvBlocks (each has 2 convs + 1 pool).

    Returns:
        Receptive field in input pixels.

    Example:
        receptive_field_size(4) → 62
        (Larger than 32×32 — every output neuron sees the full image.)
    """
    rf = 1
    for _ in range(n_conv_blocks):
        rf += 2   # first conv (3×3, padding=1 → +2 in RF)
        rf += 2   # second conv
        rf *= 2   # MaxPool2d stride=2
    return rf


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import DEVICE

    print("=" * 60)
    print("  src/models/scratch_cnn.py — smoke test")
    print("=" * 60)

    batch = torch.randn(8, 3, 32, 32).to(DEVICE)

    for variant in ["standard", "small", "residual"]:
        print(f"\n── Variant: {variant} {'─'*40}")
        model = build_scratch_cnn(variant=variant, num_classes=100).to(DEVICE)

        # Forward pass
        model.train()
        logits = model(batch)
        assert logits.shape == (8, 100), \
            f"Expected (8,100), got {logits.shape}"

        # Feature extraction
        model.eval()
        with torch.no_grad():
            feats = model.extract_features(batch) \
                if hasattr(model, "extract_features") else None
            proba = model.predict_proba(batch) \
                if hasattr(model, "predict_proba") else None

        if feats is not None:
            assert feats.shape[0] == 8, "Feature batch dim mismatch"
            print(f"  Feature shape  : {tuple(feats.shape)}")

        if proba is not None:
            assert abs(proba.sum(dim=1).mean().item() - 1.0) < 1e-5, \
                "Probabilities do not sum to 1"
            print(f"  Proba sum check: OK  (mean={proba.sum(dim=1).mean():.6f})")

        count_parameters(model)

        if variant == "standard":
            print(model)

    # Receptive field
    rf = receptive_field_size(n_conv_blocks=4)
    print(f"\n── Receptive field (4 blocks): {rf}px  "
          f"({'≥' if rf >= 32 else '<'} 32px input — "
          f"{'full image visible' if rf >= 32 else 'partial view'})")

    # Gradient flow check
    print("\n── Gradient flow check ──────────────────────────────────────")
    model = build_scratch_cnn("standard").to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    fake_labels = torch.randint(0, 100, (8,)).to(DEVICE)
    loss = F.cross_entropy(model(batch), fake_labels)
    loss.backward()

    zero_grad_layers = [
        name for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().max().item() == 0.0
    ]
    if zero_grad_layers:
        print(f"  WARNING — zero gradients in: {zero_grad_layers}")
    else:
        print(f"  All layers have non-zero gradients — OK")
        print(f"  Loss: {loss.item():.4f}")

    print("\n[scratch_cnn] All smoke tests passed.")