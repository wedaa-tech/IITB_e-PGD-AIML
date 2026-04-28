"""
src/models/vgg_extractor.py
────────────────────────────
VGG16-BN feature extractor and MLP classifier for CIFAR-100.

Architecture overview:
    Stage 1 — Feature extraction (offline, run once):
        Input 32×32 → upsample → 224×224
        → VGG16-BN convolutional backbone (ALL 13 conv layers frozen)
        → Global Average Pooling (7×7 → 1×1)
        → 512-dimensional feature vector per image

    Stage 2 — MLP training (fast, features cached to disk):
        512-dim feature vector
        → Linear(512→512) → BN1d → ReLU → Dropout(0.5)
        → Linear(512→256) → BN1d → ReLU → Dropout(0.3)
        → Linear(256→100)  [raw logits]

Design justifications:
    VGG16-BN chosen over VGG16:
        Batch normalisation makes the conv features more stable and
        better distributed, producing higher-quality 512-dim vectors
        for the downstream MLP. VGG16 without BN produces noisier
        activation distributions that degrade MLP performance.

    Global Average Pooling (not flatten):
        VGG's avgpool outputs [B, 512, 7, 7]. Flattening gives 25,088
        dimensions — an MLP on 25k inputs would have millions of
        parameters in its first layer alone, defeating the "small MLP"
        requirement. GAP collapses spatial dimensions to [B, 512] by
        averaging each channel map, discarding spatial layout and
        keeping only channel-level feature presence. This produces a
        compact, spatially-invariant 512-dim descriptor.

    Why freeze ALL conv layers (not just early ones):
        The task is to use VGG purely as a feature extractor — the
        experiment explicitly tests what ImageNet features alone can
        achieve without any task-specific adaptation. Allowing any
        conv layer to update would partially convert this into fine-
        tuning, muddying the comparison. Full freezing guarantees the
        512-dim vectors are pure ImageNet representations.

    MLP depth (3 layers):
        A single linear layer (logistic regression) on 512 features
        cannot capture the non-linear structure needed to separate 100
        classes. Two hidden layers with ~500 parameters each provides
        sufficient non-linear capacity while remaining "small" in the
        sense that it trains in seconds rather than hours.

Exports:
    VGGFeatureExtractor     — frozen VGG16-BN backbone + GAP
    MLPClassifier           — small trainable MLP head
    VGGWithMLP              — combined model for end-to-end inference
    extract_and_cache()     — runs extraction, saves .npy files to disk
    load_cached_features()  — loads saved .npy files into DataLoaders
    count_parameters()      — parameter counts for both stages
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

from src.config import DEVICE, NUM_CLASSES


# ── VGG feature extractor ─────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """
    Frozen VGG16-BN backbone used purely as a feature extractor.

    Takes a normalised [B, 3, 224, 224] tensor and returns a
    [B, 512] feature vector per image. No parameters are updated
    during any training — this module is always in eval mode.

    Args:
        pretrained: Load ImageNet weights. False only for unit tests.
        vgg_variant: "vgg16_bn" (default), "vgg11_bn", or "vgg19_bn".
    """

    def __init__(
        self,
        pretrained  : bool = True,
        vgg_variant : str  = "vgg16_bn",
    ) -> None:
        super().__init__()

        # Load the full VGG model from torchvision
        print(f"  [VGGExtractor] Loading {vgg_variant} "
              f"({'pretrained ImageNet' if pretrained else 'random init'})")

        vgg = getattr(models, vgg_variant)(
            weights = "IMAGENET1K_V1" if pretrained else None
        )

        # Keep only the convolutional backbone — discard VGG's original
        # avgpool and classifier entirely
        self.features = vgg.features        # all 13 conv layers + maxpools

        # Our own Global Average Pooling: 512×7×7 → 512×1×1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Count conv parameters for reporting
        self._n_conv_params = sum(
            p.numel() for p in self.features.parameters()
        )
        self.feature_dim = 512              # output dimension

        # Freeze EVERY parameter — no gradients through VGG ever
        self._freeze_all()

        print(f"  [VGGExtractor] Conv parameters (frozen): "
              f"{self._n_conv_params:,}")
        print(f"  [VGGExtractor] Feature dimension: {self.feature_dim}")

    def _freeze_all(self) -> None:
        """Set requires_grad=False for every parameter in the backbone."""
        for param in self.features.parameters():
            param.requires_grad = False
        # Always keep in eval mode — BN running stats must not update
        self.features.eval()

    def train(self, mode: bool = True):
        """Override: backbone stays in eval regardless of mode."""
        super().train(mode)
        self.features.eval()       # always eval — critical for frozen BN
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] normalised tensor (ImageNet norm).

        Returns:
            [B, 512] feature vectors.
        """
        with torch.no_grad():               # no gradients through VGG
            feat = self.features(x)         # [B, 512, 7, 7]
        feat = self.gap(feat)               # [B, 512, 1, 1]
        feat = feat.flatten(start_dim=1)    # [B, 512]
        return feat

    def extract_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — used in extraction loops."""
        return self.forward(x)


# ── MLP classifier ────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Small 3-layer MLP trained on top of VGG features.

    Architecture:
        512 → Linear → BN1d → ReLU → Dropout(0.5)
            → Linear → BN1d → ReLU → Dropout(0.3)
            → Linear → [100 logits]

    Total trainable parameters: ~393,000
    Compare to scratch CNN: ~9,200,000  (23× fewer)
    Compare to EfficientNet head: ~1,280,000  (3× fewer)

    Design rationale:
        Three layers are the minimum for learning non-linear class
        boundaries in a 512-dim space with 100 classes.
        BatchNorm1d after each hidden layer stabilises training and
        reduces sensitivity to learning rate choice.
        Dropout prevents the MLP from overfitting to the finite set
        of 45,000 feature vectors — since features are fixed (no
        augmentation in feature space), overfitting is a real risk.

    Args:
        input_dim:    Dimension of input features (512 for VGG16-BN GAP).
        hidden_dims:  List of hidden layer sizes. Default [512, 256].
        num_classes:  Output classes. Default 100 for CIFAR-100.
        dropout_rates: Dropout probability for each hidden layer.
    """

    def __init__(
        self,
        input_dim    : int       = 512,
        hidden_dims  : list[int] = None,
        num_classes  : int       = 100,
        dropout_rates: list[float] = None,
    ) -> None:
        super().__init__()

        if hidden_dims   is None: hidden_dims   = [512, 256]
        if dropout_rates is None: dropout_rates = [0.5, 0.3]

        assert len(hidden_dims) == len(dropout_rates), \
            "hidden_dims and dropout_rates must have the same length"

        layers = []
        prev_dim = input_dim

        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers += [
                nn.Linear(prev_dim, h_dim, bias=False),  # bias=False: BN absorbs
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            ]
            prev_dim = h_dim

        # Output layer — no BN, no activation before loss
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        print(f"  [MLP] Architecture: {input_dim} → "
              f"{' → '.join(str(h) for h in hidden_dims)} → {num_classes}")
        print(f"  [MLP] Trainable parameters: {total:,}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 512] feature vectors.

        Returns:
            [B, 100] raw logits.
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities [B, 100]."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)


# ── Combined model for inference ──────────────────────────────────────────────

class VGGWithMLP(nn.Module):
    """
    Combined VGG feature extractor + MLP classifier for end-to-end
    evaluation on raw (upscaled) CIFAR-100 images.

    Used only at test time — MLP is already trained on cached features.
    VGG remains frozen throughout.

    Args:
        extractor: Trained VGGFeatureExtractor.
        mlp:       Trained MLPClassifier.
    """

    def __init__(
        self,
        extractor : VGGFeatureExtractor,
        mlp       : MLPClassifier,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.mlp       = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] → [B, 100] logits."""
        features = self.extractor(x)       # frozen, no grad
        return self.mlp(features)          # trainable

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)


# ── Feature extraction and caching ───────────────────────────────────────────

def extract_and_cache(
    extractor  : VGGFeatureExtractor,
    loaders    : dict[str, DataLoader],
    cache_dir  : Path,
) -> dict[str, Path]:
    """
    Run VGG feature extraction for all splits and save to .npy files.

    This is the most compute-intensive step and should be run only once.
    Subsequent MLP training epochs read from the fast numpy cache.

    Args:
        extractor: Frozen VGGFeatureExtractor on DEVICE.
        loaders:   {"train": ..., "val": ..., "test": ...} DataLoaders.
                   Images must already be 224×224 with ImageNet normalisation.
        cache_dir: Directory to save .npy files.

    Returns:
        dict mapping split name → path to features .npy file.

    Saved files:
        cache_dir/vgg_features_train.npy   [45000, 512]
        cache_dir/vgg_labels_train.npy     [45000]
        cache_dir/vgg_features_val.npy     [5000, 512]
        cache_dir/vgg_labels_val.npy       [5000]
        cache_dir/vgg_features_test.npy    [10000, 512]
        cache_dir/vgg_labels_test.npy      [10000]
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    extractor.eval()
    paths = {}

    for split, loader in loaders.items():
        feat_path  = cache_dir / f"vgg_features_{split}.npy"
        label_path = cache_dir / f"vgg_labels_{split}.npy"

        if feat_path.exists() and label_path.exists():
            print(f"  [cache] {split}: already cached at {feat_path.name}")
            paths[split] = feat_path
            continue

        print(f"\n  [cache] Extracting {split} features "
              f"({len(loader.dataset):,} images) …")

        all_features, all_labels = [], []
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(loader):
                imgs = imgs.to(DEVICE)
                feat = extractor(imgs)              # [B, 512]
                all_features.append(feat.cpu().numpy())
                all_labels.append(labels.numpy())

                if (batch_idx + 1) % 50 == 0:
                    done = (batch_idx + 1) * loader.batch_size
                    total = len(loader.dataset)
                    print(f"    {done:>6}/{total}  "
                          f"({done/total*100:.1f}%)", end="\r")

        features_np = np.vstack(all_features)       # [N, 512]
        labels_np   = np.concatenate(all_labels)    # [N]

        np.save(feat_path,  features_np)
        np.save(label_path, labels_np)

        elapsed = time.perf_counter() - t0
        print(f"    Saved {features_np.shape}  →  {feat_path.name}  "
              f"({elapsed:.1f}s)")

        paths[split] = feat_path

    return paths


def load_cached_features(
    cache_dir  : Path,
    batch_size : int = 256,
    splits     : list[str] = None,
) -> dict[str, DataLoader]:
    """
    Load cached .npy feature files into TensorDataset DataLoaders.

    Feature DataLoaders are much faster than image DataLoaders because:
    - No disk I/O for image decoding or resizing
    - No CPU transform pipeline
    - 512 floats per sample vs 3×224×224 = 150,528 floats

    Args:
        cache_dir:  Directory containing .npy files from extract_and_cache().
        batch_size: Mini-batch size for MLP training (can be large, e.g. 512).
        splits:     Which splits to load. Default ["train","val","test"].

    Returns:
        dict of split → DataLoader over TensorDataset(features, labels).
    """
    if splits is None:
        splits = ["train", "val", "test"]

    loaders = {}
    for split in splits:
        feat_path  = cache_dir / f"vgg_features_{split}.npy"
        label_path = cache_dir / f"vgg_labels_{split}.npy"

        if not feat_path.exists():
            raise FileNotFoundError(
                f"Cached features not found: {feat_path}\n"
                f"Run extract_and_cache() first."
            )

        features = torch.from_numpy(np.load(feat_path)).float()
        labels   = torch.from_numpy(np.load(label_path)).long()
        ds       = TensorDataset(features, labels)

        loaders[split] = DataLoader(
            ds,
            batch_size = batch_size,
            shuffle    = (split == "train"),
            num_workers = 0,
            pin_memory  = False,
        )
        print(f"  [cache] Loaded {split}: {features.shape}  "
              f"→  {len(loaders[split])} batches of {batch_size}")

    return loaders


# ── Parameter counting ────────────────────────────────────────────────────────

def count_parameters(
    extractor : VGGFeatureExtractor,
    mlp       : MLPClassifier,
) -> dict:
    """Report parameter counts for both stages."""
    vgg_total   = sum(p.numel() for p in extractor.features.parameters())
    mlp_total   = sum(p.numel() for p in mlp.parameters())
    mlp_train   = sum(p.numel() for p in mlp.parameters()
                      if p.requires_grad)

    print(f"\n  Parameter summary:")
    print(f"    VGG16-BN backbone (frozen) : {vgg_total:>12,}")
    print(f"    MLP classifier (trainable) : {mlp_train:>12,}")
    print(f"    Total (backbone + MLP)     : {vgg_total + mlp_total:>12,}")
    print(f"    Actually trained           : {mlp_train:>12,}  "
          f"({mlp_train/(vgg_total+mlp_total)*100:.1f}% of total)")

    return {
        "vgg_frozen"   : vgg_total,
        "mlp_trainable": mlp_train,
        "total"        : vgg_total + mlp_total,
    }