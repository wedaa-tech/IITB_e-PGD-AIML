"""
src/noise.py
────────────
Reusable Gaussian noise injection module.

All noise operations work in [0,1] pixel space as specified in the
assignment: "zero-mean Gaussian noise with variance σ²=0.05, assuming
input images are normalised to the range [0,1]".

Workflow for each image tensor:
    1. Un-normalise  →  restore to [0,1] pixel space
    2. Add noise     →  x + N(0, σ²)
    3. Clip          →  clamp to [0,1]
    4. Re-normalise  →  restore to model-input space

The noise schedule is documented and saved to JSON so subsequent
experiments can reproduce the exact noise parameters.

Exports:
    NoiseConfig              — dataclass holding all noise parameters
    save_noise_schedule()    — serialise NoiseConfig to JSON
    load_noise_schedule()    — restore NoiseConfig from JSON
    unnormalise()            — tensor [0,1] from normalised tensor
    inject_noise()           — add Gaussian noise to a single batch
    NoisyDataLoader          — wraps any DataLoader, injects noise per batch
    evaluate_noisy()         — top-1/top-k evaluation on a NoisyDataLoader
    verify_noise_statistics()— confirm injected noise has correct mean/variance
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESULT_DIR,
    DEVICE,
)


# ── Noise configuration dataclass ─────────────────────────────────────────────

@dataclass
class NoiseConfig:
    """
    Complete specification of the noise injection experiment.
    Stored to JSON so every downstream experiment uses identical parameters.

    Args:
        variance:         Noise variance σ² — assignment specifies 0.05.
        pixel_range:      Assumed pixel value range before noise ("0_1").
        distribution:     Noise distribution name ("gaussian").
        clip_after_noise: Whether to clamp noisy pixels to [0,1].
        normalisation:    Which normalisation stats the model expects
                          ("cifar100" or "imagenet").
        seed:             RNG seed for reproducibility. None = unseeded.
        description:      Free-text description for the experiment log.
    """
    variance         : float  = 0.05
    pixel_range      : str    = "0_1"
    distribution     : str    = "gaussian"
    clip_after_noise : bool   = True
    normalisation    : str    = "cifar100"
    seed             : int | None = 42
    description      : str    = (
        "Additive zero-mean Gaussian noise, σ²=0.05, "
        "applied in [0,1] pixel space before re-normalisation. "
        "Assignment specification: CIFAR-100 robustness evaluation."
    )

    # ── Derived read-only properties ──────────────────────────────────────────

    @property
    def sigma(self) -> float:
        """Standard deviation σ = √σ²."""
        return math.sqrt(self.variance)

    @property
    def mean_param(self) -> float:
        """Mean of the noise distribution (always zero for this experiment)."""
        return 0.0

    def summary(self) -> str:
        return (
            f"NoiseConfig | distribution={self.distribution} | "
            f"σ²={self.variance} | σ={self.sigma:.6f} | "
            f"clip={self.clip_after_noise} | norm={self.normalisation} | "
            f"seed={self.seed}"
        )


# ── Noise schedule serialisation ──────────────────────────────────────────────

def save_noise_schedule(
    config   : NoiseConfig,
    filename : str = "noise_schedule.json",
) -> Path:
    """
    Serialise a NoiseConfig to JSON in RESULT_DIR.
    Adds a timestamp and the derived sigma value for documentation.

    Args:
        config:   NoiseConfig instance to save.
        filename: Output filename inside outputs/results/.

    Returns:
        Path to the saved JSON file.
    """
    payload = asdict(config)
    payload["sigma"]      = config.sigma               # derived — for documentation
    payload["mean_param"] = config.mean_param
    payload["saved_at"]   = time.strftime("%Y-%m-%d %H:%M:%S")

    path = RESULT_DIR / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[noise] Schedule saved → {path}")
    print(f"[noise] {config.summary()}")
    return path


def load_noise_schedule(filename: str = "noise_schedule.json") -> NoiseConfig:
    """
    Restore a NoiseConfig from a previously saved JSON file.
    Use this in subsequent experiments to guarantee identical parameters.

    Args:
        filename: JSON filename inside outputs/results/.

    Returns:
        Reconstructed NoiseConfig.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    path = RESULT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"[noise] Schedule not found: {path}\n"
            f"  Run save_noise_schedule() first."
        )
    with open(path) as f:
        data = json.load(f)

    # Strip derived fields before reconstructing the dataclass
    for derived in ("sigma", "mean_param", "saved_at"):
        data.pop(derived, None)

    config = NoiseConfig(**data)
    print(f"[noise] Schedule loaded ← {path}")
    print(f"[noise] {config.summary()}")
    return config


# ── Normalisation lookup ───────────────────────────────────────────────────────

def _get_norm_stats(normalisation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (mean, std) tensors shaped [3,1,1] for broadcasting over [B,3,H,W].

    Args:
        normalisation: "cifar100" or "imagenet".

    Returns:
        (mean_tensor, std_tensor) both shape [3,1,1].
    """
    if normalisation == "cifar100":
        mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        std  = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    elif normalisation == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    else:
        raise ValueError(
            f"Unknown normalisation '{normalisation}'. "
            f"Choose 'cifar100' or 'imagenet'."
        )
    return mean, std


# ── Core noise operations ─────────────────────────────────────────────────────

def unnormalise(
    x             : torch.Tensor,
    normalisation : str = "cifar100",
) -> torch.Tensor:
    """
    Reverse the channel-wise normalisation to restore pixel values to [0,1].

    Formula: x_pixel = x_normalised * std + mean

    Args:
        x:             Normalised tensor, shape [B,3,H,W] or [3,H,W].
        normalisation: Which stats to use ("cifar100" or "imagenet").

    Returns:
        Tensor in approximately [0,1] range (may slightly exceed due to
        floating-point, will be clamped in inject_noise).
    """
    mean, std = _get_norm_stats(normalisation)
    mean = mean.to(x.device)
    std  = std.to(x.device)
    return x * std + mean


def inject_noise(
    x             : torch.Tensor,
    config        : NoiseConfig,
    generator     : torch.Generator | None = None,
) -> torch.Tensor:
    """
    Inject additive Gaussian noise in [0,1] pixel space.

    Full pipeline per batch:
        1. Un-normalise  →  [0,1] pixel space
        2. Generate noise ε ~ N(0, σ²) with same shape as x
        3. Add:  x_noisy = x_pixel + ε
        4. Clip: x_noisy = clamp(x_noisy, 0, 1)
        5. Re-normalise → model input space

    Args:
        x:         Batch of normalised images, shape [B,3,H,W].
        config:    NoiseConfig with variance, clip, and normalisation settings.
        generator: Optional torch.Generator for seeded reproducibility.
                   Pass None to use global RNG state.

    Returns:
        Noisy batch in normalised model-input space, same shape as x.
    """
    mean, std = _get_norm_stats(config.normalisation)
    mean = mean.to(x.device)
    std  = std.to(x.device)

    # Step 1: un-normalise to [0,1] pixel space
    x_pixel = x * std + mean                         # [B, 3, H, W]

    # Step 2: generate ε ~ N(0, σ²) = N(0, variance)
    #   torch.normal takes std (σ), not variance (σ²)
    # Sample on CPU with the seeded generator, then move to the tensor's
    # device. This is required because MPS/CUDA tensors reject a CPU
    # generator, and it also keeps the noise sequence reproducible
    # regardless of the compute device.
    noise_cpu = torch.empty(x_pixel.shape, dtype=x_pixel.dtype).normal_(
        mean=0.0,
        std=config.sigma,                            # σ = √σ² = √0.05
        generator=generator,
    )
    noise = noise_cpu.to(x_pixel.device, non_blocking=False)

    # Step 3: add noise
    x_noisy = x_pixel + noise                        # still in pixel space

    # Step 4: clip to valid pixel range
    if config.clip_after_noise:
        x_noisy = x_noisy.clamp(0.0, 1.0)

    # Step 5: re-normalise to model-input space
    x_model = (x_noisy - mean) / std

    return x_model


# ── NoisyDataLoader ────────────────────────────────────────────────────────────

class NoisyDataLoader:
    """
    Wraps any existing DataLoader and injects Gaussian noise into every batch.

    Usage:
        noisy_loader = NoisyDataLoader(test_loader, config)
        for imgs, labels in noisy_loader:
            # imgs already have noise applied
            logits = model(imgs.to(DEVICE))

    Args:
        loader: Source DataLoader (test or val, no augmentation).
        config: NoiseConfig specifying variance, clip, normalisation.
        device: Device to move images to before noise injection.
                If None, images stay on CPU.

    Notes:
        - A new seeded Generator is created per epoch iteration if
          config.seed is not None. This makes every evaluation
          deterministic and reproducible.
        - Images are NOT moved to device by this wrapper — the caller's
          training loop should do .to(DEVICE) as usual.
    """

    def __init__(
        self,
        loader : DataLoader,
        config : NoiseConfig,
        device : torch.device | None = None,
    ) -> None:
        self.loader = loader
        self.config = config
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.config.seed)

        for imgs, labels in self.loader:
            noisy_imgs = inject_noise(imgs, self.config, generator=generator)
            yield noisy_imgs, labels

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


# ── Evaluate one model on a NoisyDataLoader ───────────────────────────────────

def evaluate_noisy(
    model        : torch.nn.Module,
    noisy_loader : "NoisyDataLoader",
    label        : str,
    top_k        : int = 5,
) -> dict:
    """
    Run top-1 / top-k accuracy evaluation over a NoisyDataLoader.

    Args:
        model:        Trained model already on DEVICE.
        noisy_loader: NoisyDataLoader wrapping the test set.
        label:        Short name for logging.
        top_k:        k for top-k accuracy.

    Returns:
        dict with top1, top5, loss, correct_1, correct_5, total,
        avg_confidence_correct, avg_confidence_incorrect.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total         = 0
    correct_1     = 0
    correct_5     = 0
    running_loss  = 0.0
    conf_correct  = []
    conf_wrong    = []

    with torch.no_grad():
        for imgs, labels in noisy_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, labels)
            probs  = F.softmax(logits, dim=1)

            # Top-1
            preds_1      = logits.argmax(dim=1)
            correct_mask = (preds_1 == labels)
            correct_1   += correct_mask.sum().item()

            # Top-k
            top_k_idx = logits.topk(top_k, dim=1).indices
            for i in range(len(labels)):
                if labels[i].item() in top_k_idx[i].tolist():
                    correct_5 += 1

            # Loss
            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)

            # Confidence split between correct / incorrect predictions
            max_probs = probs.max(dim=1).values.cpu()
            conf_correct.extend(max_probs[correct_mask.cpu()].tolist())
            conf_wrong.extend(max_probs[~correct_mask.cpu()].tolist())

    top1     = correct_1 / total
    top5     = correct_5 / total
    avg_loss = running_loss / total
    avg_conf_correct = float(np.mean(conf_correct)) if conf_correct else 0.0
    avg_conf_wrong   = float(np.mean(conf_wrong))   if conf_wrong   else 0.0

    print(f"\n  [{label}] Noisy test results:")
    print(f"    Top-1 accuracy        : {top1:.4f}  ({top1*100:.2f}%)")
    print(f"    Top-{top_k} accuracy        : {top5:.4f}  ({top5*100:.2f}%)")
    print(f"    Test loss             : {avg_loss:.4f}")
    print(f"    Avg confidence (correct)   : {avg_conf_correct:.4f}")
    print(f"    Avg confidence (incorrect) : {avg_conf_wrong:.4f}")
    print(f"    Total samples         : {total:,}")

    return {
        "label"                   : label,
        "top1"                    : float(top1),
        "top5"                    : float(top5),
        "loss"                    : float(avg_loss),
        "correct_1"               : correct_1,
        "correct_5"               : correct_5,
        "total"                   : total,
        "avg_confidence_correct"  : avg_conf_correct,
        "avg_confidence_incorrect": avg_conf_wrong,
    }


# ── Noise verification ────────────────────────────────────────────────────────

def verify_noise_statistics(
    loader : DataLoader,
    config : NoiseConfig,
    n_batches : int = 10,
) -> dict:
    """
    Empirically verify that the injected noise has the correct statistics.

    Collects pixel-space differences (x_noisy - x_original) over n_batches
    and reports the empirical mean and variance, which should match
    config.mean_param (≈ 0) and config.variance (≈ 0.05).

    Args:
        loader:    Clean test DataLoader.
        config:    NoiseConfig to verify.
        n_batches: Number of batches to sample.

    Returns:
        dict with keys: empirical_mean, empirical_variance, empirical_sigma,
                        target_variance, target_sigma, variance_error_pct.
    """
    mean_t, std_t = _get_norm_stats(config.normalisation)

    all_diffs = []
    generator = None
    if config.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(config.seed + 9999)   # separate seed from eval

    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break

        # Un-normalise original
        x_orig = imgs * std_t + mean_t

        # Generate noisy version
        x_noisy_norm = inject_noise(imgs, config, generator=generator)
        x_noisy      = x_noisy_norm * std_t + mean_t

        # Compute pixel-space difference  (noise = noisy - original before clip)
        diff = x_noisy - x_orig                     # clipping may reduce variance
        all_diffs.append(diff.flatten())

    diffs   = torch.cat(all_diffs)
    emp_mean = diffs.mean().item()
    emp_var  = diffs.var().item()
    emp_sig  = math.sqrt(abs(emp_var))
    var_err  = abs(emp_var - config.variance) / config.variance * 100

    print(f"\n[noise] Noise verification over {n_batches} batches:")
    print(f"  Target  mean     : {config.mean_param:.6f}")
    print(f"  Empirical mean   : {emp_mean:.6f}  "
          f"{'✓' if abs(emp_mean) < 0.01 else '⚠ check seed'}")
    print(f"  Target  variance : {config.variance:.6f}")
    print(f"  Empirical var    : {emp_var:.6f}  (error: {var_err:.1f}%)")
    print(f"  Target  σ        : {config.sigma:.6f}")
    print(f"  Empirical σ      : {emp_sig:.6f}")
    print(f"  Note: variance error > 0 is expected — clamp(0,1) truncates")
    print(f"        the tails of the Gaussian, reducing empirical variance.")

    return {
        "empirical_mean"     : emp_mean,
        "empirical_variance" : emp_var,
        "empirical_sigma"    : emp_sig,
        "target_variance"    : config.variance,
        "target_sigma"       : config.sigma,
        "variance_error_pct" : var_err,
    }