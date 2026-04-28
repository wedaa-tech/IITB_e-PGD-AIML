"""
src/dataset.py
──────────────
Handles all data loading, transformation, and splitting for CIFAR-100.

Exports:
    get_scratch_transforms()   → (train_tf, test_tf) for the scratch CNN
    get_transfer_transforms()  → (train_tf, test_tf) for transfer learning
    get_dataloaders()          → (train_loader, val_loader, test_loader)
    show_sample_grid()         → visualise a grid of sample images
    get_class_names()          → list of all 100 CIFAR-100 class names
"""

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR100

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    PLOT_DIR,
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
    NUM_CLASSES,
)


def _verify_data_dir() -> None:
    """
    Confirm the manually downloaded CIFAR-100 files are in the right place.
    Called once at the start of get_dataloaders().
    """
    expected = DATA_DIR / "cifar-100-python"
    required = ["train", "test", "meta"]
    missing  = [f for f in required if not (expected / f).exists()]

    if missing:
        raise FileNotFoundError(
            f"\n[dataset] Missing files inside {expected}: {missing}\n"
            f"  Make sure you extracted cifar-100-python.tar.gz into data/\n"
            f"  Run:  cd data && tar -xzf cifar-100-python.tar.gz"
        )
    print(f"[dataset] Data verified at: {expected}")

# ── Class names ────────────────────────────────────────────────────────────────

def get_class_names() -> list[str]:
    """
    Return all 100 CIFAR-100 fine-grained class names in label order.
    These are the same names torchvision uses internally.
    """
    # Temporarily load the dataset just to extract class names (no transform needed)
    ds = CIFAR100(root=DATA_DIR, train=True, download=False, transform=T.ToTensor())
    return ds.classes   # list of 100 strings, index == label


# ── Transforms ────────────────────────────────────────────────────────────────

def get_scratch_transforms():
    """
    Transforms for the custom scratch CNN.
    Input images stay at native 32×32 — no upscaling.

    Training augmentations applied:
      - RandomCrop with padding   → shifts the object slightly, prevents position bias
      - RandomHorizontalFlip      → doubles effective dataset size for symmetric objects
      - ColorJitter               → brightness / contrast / saturation / hue variation
      - RandomRotation ±15°       → slight rotation invariance
    Normalisation uses CIFAR-100 per-channel mean and std (not ImageNet values).
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomRotation(degrees=15),
        T.RandomGrayscale(p=0.05),          # occasionally strip colour → robustness
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    return train_transform, test_transform


def get_transfer_transforms(input_size: int = 224):
    """
    Transforms for the pretrained transfer learning model.
    Images are upscaled from 32×32 → input_size (default 224) to match
    the resolution ImageNet-pretrained models expect.

    Normalisation uses ImageNet mean and std — critical for pretrained weights
    to function correctly. Using CIFAR-100 stats here would degrade performance.

    Args:
        input_size: Target spatial size. EfficientNet-B0 expects 224.
                    Change to 240 for EfficientNet-B1, 260 for B2, etc.
    """
    train_transform = T.Compose([
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(input_size, padding=input_size // 8, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = T.Compose([
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),           # deterministic crop for evaluation
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


# ── Dataset splitting helper ───────────────────────────────────────────────────

def _make_val_subset(val_indices: list[int], test_transform) -> Subset:
    """
    Build a validation Subset that uses test_transform (no augmentation),
    independently of the augmented training dataset.

    This is necessary because random_split shares the underlying dataset object,
    so the val split would otherwise inherit the training augmentations.
    We create a second dataset object with test_transform applied and index
    into it with the same val_indices.
    """
    val_base = CIFAR100(
        root=DATA_DIR, train=True, download=False, transform=test_transform)
    return Subset(val_base, val_indices)


# ── Main dataloader factory ────────────────────────────────────────────────────

def get_dataloaders(
    mode        : str = "scratch",
    batch_size  : int = 128,
    num_workers : int = 0,
    input_size  : int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader) for CIFAR-100.

    Args:
        mode:        "scratch"  → 32×32, CIFAR-100 normalisation
                     "transfer" → upscaled to input_size, ImageNet normalisation
        batch_size:  Number of images per mini-batch.
        num_workers: Parallel workers for data loading.
                     Keep at 0 on macOS to avoid multiprocessing errors.
        input_size:  Target image size (only used when mode="transfer").

    Returns:
        train_loader : augmented, shuffled, 45,000 samples
        val_loader   : no augmentation, unshuffled, 5,000 samples
        test_loader  : no augmentation, unshuffled, 10,000 samples

    Notes:
        - CIFAR-100 is downloaded automatically to DATA_DIR on first call.
        - The train/val split is reproducible via RANDOM_SEED.
        - pin_memory is disabled — it is not supported on MPS (Apple Silicon).
    """

    _verify_data_dir()


    # Select transforms based on mode
    if mode == "scratch":
        train_tf, test_tf = get_scratch_transforms()
    elif mode == "transfer":
        train_tf, test_tf = get_transfer_transforms(input_size)
    else:
        raise ValueError(f"mode must be 'scratch' or 'transfer', got '{mode}'")

    # ── Load full training set with augmentation ───────────────────────────
    train_full = CIFAR100(
        root=DATA_DIR, train=True, download=False, transform=train_tf)

    # ── Reproducible train / val split ────────────────────────────────────
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_indices, val_indices = random_split(
        range(len(train_full)),          # split indices, not the dataset itself
        [TRAIN_SIZE, VAL_SIZE],
        generator=generator,
    )
    train_indices = list(train_indices)
    val_indices   = list(val_indices)

    train_subset = Subset(train_full, train_indices)

    # Val subset uses test_tf (no augmentation) — see docstring of helper above
    val_subset   = _make_val_subset(val_indices, test_tf)

    # ── Test set ──────────────────────────────────────────────────────────
    test_set = CIFAR100(
        root=DATA_DIR, train=False, download=False, transform=test_tf)

    # ── DataLoaders ───────────────────────────────────────────────────────
    shared_kwargs = dict(
        num_workers = num_workers,
        pin_memory  = False,          # must be False for MPS
    )

    train_loader = DataLoader(
        train_subset,
        batch_size = batch_size,
        shuffle    = True,            # shuffle every epoch
        **shared_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size = batch_size,
        shuffle    = False,
        **shared_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle    = False,
        **shared_kwargs,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"[dataset] Mode       : {mode}  |  Input size: {input_size}×{input_size}")
    print(f"[dataset] Train      : {len(train_subset):,} images")
    print(f"[dataset] Validation : {len(val_subset):,} images")
    print(f"[dataset] Test       : {len(test_set):,} images")
    print(f"[dataset] Batch size : {batch_size}  |  Workers: {num_workers}")
    print(f"[dataset] Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ── Dataset statistics ────────────────────────────────────────────────────────

def compute_dataset_stats(loader: DataLoader) -> tuple[list, list]:
    """
    Compute per-channel mean and standard deviation over a DataLoader.
    Useful for verifying or recomputing CIFAR100_MEAN / CIFAR100_STD.

    Args:
        loader: A DataLoader whose images are in [0, 1] range (ToTensor only,
                no Normalize applied yet).

    Returns:
        mean: list of 3 floats [R_mean, G_mean, B_mean]
        std:  list of 3 floats [R_std,  G_std,  B_std]
    """
    channel_sum     = torch.zeros(3)
    channel_sq_sum  = torch.zeros(3)
    n_pixels        = 0

    for imgs, _ in loader:
        # imgs shape: [B, C, H, W]
        channel_sum    += imgs.sum(dim=[0, 2, 3])
        channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3])
        n_pixels       += imgs.size(0) * imgs.size(2) * imgs.size(3)

    mean = (channel_sum    / n_pixels).tolist()
    std  = ((channel_sq_sum / n_pixels) - torch.tensor(mean) ** 2).sqrt().tolist()

    print(f"[dataset] Computed mean : {[round(m, 4) for m in mean]}")
    print(f"[dataset] Computed std  : {[round(s, 4) for s in std]}")
    return mean, std


def class_distribution(dataset) -> dict[str, int]:
    """
    Count how many samples exist per class in a dataset or Subset.
    CIFAR-100 training set is perfectly balanced: 500 samples per class.

    Returns:
        dict mapping class_name → count
    """
    class_names = get_class_names()

    if isinstance(dataset, Subset):
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = dataset.targets

    counts = {}
    for label in labels:
        name = class_names[label]
        counts[name] = counts.get(name, 0) + 1

    print(f"[dataset] Unique classes: {len(counts)}")
    print(f"[dataset] Min samples   : {min(counts.values())}")
    print(f"[dataset] Max samples   : {max(counts.values())}")
    return counts


# ── Visualisation ─────────────────────────────────────────────────────────────

def show_sample_grid(
    mode        : str = "scratch",
    n_classes   : int = 10,
    n_per_class : int = 5,
    save        : bool = True,
) -> None:
    """
    Display a grid of sample images: n_classes rows × n_per_class columns.
    Images are shown before normalisation (raw pixel values) for readability.

    Args:
        mode:        "scratch" or "transfer" — determines which transforms are shown
        n_classes:   Number of classes to show (rows).
        n_per_class: Samples per class (columns).
        save:        If True, saves the figure to outputs/plots/.
    """
    # Load raw (un-normalised) dataset for display
    raw_ds      = CIFAR100(root=DATA_DIR, train=True,
                           download=False, transform=T.ToTensor())
    class_names = raw_ds.classes

    # Collect sample indices per class
    class_to_indices: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(raw_ds):
        class_to_indices[label].append(idx)

    # Pick n_classes evenly spaced across the 100 classes for variety
    chosen_classes = np.linspace(0, NUM_CLASSES - 1, n_classes, dtype=int).tolist()

    fig = plt.figure(figsize=(n_per_class * 1.6, n_classes * 1.6))
    fig.suptitle(f"CIFAR-100 sample images  ({mode} mode)",
                 fontsize=12, y=1.01)
    gs = gridspec.GridSpec(n_classes, n_per_class, hspace=0.6, wspace=0.15)

    rng = np.random.default_rng(RANDOM_SEED)

    for row, cls_idx in enumerate(chosen_classes):
        indices = rng.choice(class_to_indices[cls_idx],
                             size=n_per_class, replace=False)
        for col, img_idx in enumerate(indices):
            img, _ = raw_ds[img_idx]            # shape [3, 32, 32], range [0, 1]
            img_np  = img.permute(1, 2, 0).numpy()   # → [32, 32, 3]

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_np, interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.set_title(class_names[cls_idx], fontsize=8,
                             loc="left", pad=2)

    plt.tight_layout()

    if save:
        path = PLOT_DIR / "sample_grid.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[dataset] Sample grid saved → {path}")

    plt.show()


def show_augmentation_comparison(class_idx: int = 0, n_samples: int = 6) -> None:
    """
    Show the same image with and without training augmentation side by side.
    Helpful for verifying that augmentations are sensible and not too aggressive.

    Args:
        class_idx: CIFAR-100 class label to sample from (0–99).
        n_samples: Number of augmented versions to show.
    """
    raw_ds      = CIFAR100(root=DATA_DIR, train=True,
                           download=False, transform=T.ToTensor())
    aug_tf, _   = get_scratch_transforms()
    aug_ds      = CIFAR100(root=DATA_DIR, train=True,
                           download=False, transform=aug_tf)
    class_names = raw_ds.classes

    # Find first image of the requested class
    img_idx = next(i for i, (_, lbl) in enumerate(raw_ds) if lbl == class_idx)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.8, 4))
    fig.suptitle(f"Augmentation comparison — class: {class_names[class_idx]}",
                 fontsize=11)

    for col in range(n_samples):
        # Top row: original (no augmentation)
        raw_img, _ = raw_ds[img_idx]
        axes[0, col].imshow(raw_img.permute(1, 2, 0).numpy(), interpolation="nearest")
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=9)

        # Bottom row: augmented version (random each time)
        aug_img, _ = aug_ds[img_idx]
        # Un-normalise for display: x = x*std + mean
        mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        std  = torch.tensor(CIFAR100_STD).view(3, 1, 1)
        aug_display = (aug_img * std + mean).clamp(0, 1)
        axes[1, col].imshow(aug_display.permute(1, 2, 0).numpy(),
                            interpolation="nearest")
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("Augmented", fontsize=9)

    plt.tight_layout()
    path = PLOT_DIR / "augmentation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[dataset] Augmentation comparison saved → {path}")
    plt.show()


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Scratch loaders ──────────────────────────────────────────")
    train_l, val_l, test_l = get_dataloaders(mode="scratch", batch_size=128)

    imgs, labels = next(iter(train_l))
    print(f"Batch shape : {imgs.shape}")        # [128, 3, 32, 32]
    print(f"Label range : {labels.min()}–{labels.max()}")
    print(f"Pixel range : {imgs.min():.3f} – {imgs.max():.3f}")

    print("\n── Transfer loaders ─────────────────────────────────────────")
    train_l2, val_l2, test_l2 = get_dataloaders(
        mode="transfer", batch_size=64, input_size=224)
    imgs2, _ = next(iter(train_l2))
    print(f"Batch shape : {imgs2.shape}")       # [64, 3, 224, 224]

    print("\n── Class distribution (first 5 classes) ─────────────────────")
    dist = class_distribution(train_l.dataset)
    for name, count in list(dist.items())[:5]:
        print(f"  {name:<20} {count}")

    print("\n── Sample grid ──────────────────────────────────────────────")
    show_sample_grid(n_classes=8, n_per_class=5)

    print("\n── Augmentation comparison ──────────────────────────────────")
    show_augmentation_comparison(class_idx=3, n_samples=6)