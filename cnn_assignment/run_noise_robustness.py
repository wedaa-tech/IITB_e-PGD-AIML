"""
run_noise_robustness.py
────────────────────────
Evaluate both trained models on Gaussian-noisy test images.

Noise specification (assignment):
    Distribution : zero-mean Gaussian
    Variance     : σ² = 0.05
    σ            : √0.05 ≈ 0.2236
    Applied in   : [0,1] pixel space (before re-normalisation)
    Clipping     : yes — clamp(0,1) after noise addition

The noise schedule is saved to outputs/results/noise_schedule.json
and can be reloaded in subsequent experiments.

Usage:
    python run_noise_robustness.py                     # full evaluation
    python run_noise_robustness.py --dry-run           # 2 batches, fast check
    python run_noise_robustness.py --no-plots          # metrics only
    python run_noise_robustness.py --save-noisy-images # save sample images

Outputs:
    outputs/results/noise_schedule.json
    outputs/results/noise_results_scratch.json
    outputs/results/noise_results_transfer.json
    outputs/results/noise_combined_report.json
    outputs/plots/noise_sample_images.png
    outputs/plots/noise_accuracy_comparison.png
    outputs/plots/noise_confidence_shift.png
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import (
    DEVICE,
    SCRATCH,
    TRANSFER,
    NUM_CLASSES,
    RESULT_DIR,
    PLOT_DIR,
    CKPT_DIR,
)
from src.utils import (
    set_seed,
    check_outputs_dir,
    log_system_info,
    format_time,
)
from src.dataset import get_dataloaders
from src.models.scratch_cnn import build_scratch_cnn
from src.models.transfer_model import build_transfer_model
from src.train import load_checkpoint
from src.noise import (
    NoiseConfig,
    save_noise_schedule,
    load_noise_schedule,
    inject_noise,
    NoisyDataLoader,
    verify_noise_statistics,
    unnormalise,
)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness to Gaussian noise (σ²=0.05).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scratch-variant", default="standard",
                        choices=["standard","small","residual"])
    parser.add_argument("--backbone",        default=TRANSFER["backbone"])
    parser.add_argument("--input-size",      type=int, default=TRANSFER["input_size"])
    parser.add_argument("--scratch-ckpt",    type=Path,
                        default=SCRATCH["checkpoint"])
    parser.add_argument("--transfer-ckpt",   type=Path,
                        default=TRANSFER["checkpoint"])
    parser.add_argument("--variance",        type=float, default=0.05,
                        help="Noise variance σ² (assignment: 0.05)")
    parser.add_argument("--seed",            type=int, default=42,
                        help="RNG seed for reproducible noise")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Evaluate 2 batches only (pipeline check)")
    parser.add_argument("--no-plots",        action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--save-noisy-images", action="store_true",
                        help="Save a grid of clean vs noisy sample images")
    parser.add_argument("--verify-noise",    action="store_true",
                        help="Empirically verify noise statistics before eval")
    parser.add_argument("--no-transfer",     action="store_true",
                        help="Evaluate scratch model only")
    return parser.parse_args()


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def preflight(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  Pre-flight — Noise Robustness Evaluation")
    print("═" * 60)
    check_outputs_dir()

    for label, path in [
        ("Scratch checkpoint",  args.scratch_ckpt),
        ("Transfer checkpoint", args.transfer_ckpt),
    ]:
        if label == "Transfer checkpoint" and args.no_transfer:
            continue
        if not path.exists():
            raise FileNotFoundError(
                f"  {label} not found: {path}\n"
                f"  Run the corresponding training script first."
            )
        print(f"  {label:<24} {path.name}  ✓")

    cifar = Path("data") / "cifar-100-python"
    if not cifar.exists():
        raise FileNotFoundError(f"  CIFAR-100 data not found at {cifar}")
    print(f"  {'CIFAR-100 data':<24} found  ✓")
    print(f"  {'Device':<24} {DEVICE}")
    print(f"  {'Noise variance σ²':<24} {args.variance}")
    print(f"  {'Noise σ':<24} {args.variance**0.5:.6f}")
    print()


# ── Evaluate one model on noisy loader ────────────────────────────────────────

def evaluate_noisy(
    model        : torch.nn.Module,
    noisy_loader : NoisyDataLoader,
    label        : str,
    top_k        : int = 5,
) -> dict:
    """
    Run top-1 and top-k accuracy evaluation over a NoisyDataLoader.

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
    criterion    = torch.nn.CrossEntropyLoss()

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
            preds_1  = logits.argmax(dim=1)
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

            # Confidence split
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
    print(f"    Avg confidence (✓)   : {avg_conf_correct:.4f}")
    print(f"    Avg confidence (✗)   : {avg_conf_wrong:.4f}")
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


# ── Plot: noisy sample image grid ────────────────────────────────────────────

def plot_noisy_samples(
    clean_loader : torch.utils.data.DataLoader,
    config       : NoiseConfig,
    n_images     : int = 8,
    save_name    : str = "noise_sample_images.png",
) -> None:
    """
    Show clean vs noisy images side by side for visual verification.
    Images are displayed in [0,1] pixel space before re-normalisation.

    Args:
        clean_loader: Test DataLoader (no augmentation, no noise).
        config:       NoiseConfig used in the experiment.
        n_images:     Number of image pairs to show.
        save_name:    Filename in outputs/plots/.
    """
    imgs, labels = next(iter(clean_loader))
    imgs   = imgs[:n_images]
    labels = labels[:n_images]

    mean_t, std_t = torch.tensor(CIFAR100_MEAN).view(3,1,1), \
                    torch.tensor(CIFAR100_STD).view(3,1,1)

    # Un-normalise clean images to [0,1] for display
    clean_pixels = (imgs * std_t + mean_t).clamp(0, 1)

    # Generate noisy versions (inject_noise works on normalised tensors)
    generator = None
    if config.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
    noisy_norm   = inject_noise(imgs, config, generator=generator)
    noisy_pixels = (noisy_norm * std_t + mean_t).clamp(0, 1)

    # Load class names
    try:
        import torchvision
        from src.config import DATA_DIR
        ds = torchvision.datasets.CIFAR100(
            root=DATA_DIR, train=False, download=False,
            transform=torchvision.transforms.ToTensor())
        class_names = ds.classes
    except Exception:
        class_names = [str(i) for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.8, 4.2))
    fig.suptitle(
        f"Clean vs Noisy images  —  Gaussian noise  σ²={config.variance}  "
        f"(σ={config.sigma:.3f})",
        fontsize=10, y=1.02,
    )

    for col in range(n_images):
        # Top row: clean
        axes[0, col].imshow(
            clean_pixels[col].permute(1, 2, 0).numpy(),
            interpolation="nearest",
        )
        axes[0, col].set_title(class_names[labels[col]], fontsize=7, pad=2)
        axes[0, col].axis("off")

        # Bottom row: noisy
        axes[1, col].imshow(
            noisy_pixels[col].permute(1, 2, 0).numpy(),
            interpolation="nearest",
        )
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Clean",  fontsize=8)
    axes[1, 0].set_ylabel("Noisy",  fontsize=8)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot] Noisy samples → {path}")
    plt.show()
    plt.close()


# ── Plot: accuracy comparison bar ────────────────────────────────────────────

def plot_accuracy_comparison(
    clean_results : dict[str, float],
    noisy_results : dict[str, dict],
    config        : NoiseConfig,
    save_name     : str = "noise_accuracy_comparison.png",
) -> None:
    """
    Grouped bar chart: clean accuracy vs noisy accuracy for each model.

    Args:
        clean_results: {model_label: clean_top1_accuracy}
        noisy_results: {model_label: evaluate_noisy result dict}
        config:        NoiseConfig (for title annotation).
        save_name:     Filename in outputs/plots/.
    """
    model_labels  = list(noisy_results.keys())
    clean_accs    = [clean_results.get(lbl, 0.0) for lbl in model_labels]
    noisy_accs    = [noisy_results[lbl]["top1"]   for lbl in model_labels]
    drops         = [c - n for c, n in zip(clean_accs, noisy_accs)]

    x   = np.arange(len(model_labels))
    w   = 0.35
    colours_clean = ["#534AB7", "#0F6E56"]
    colours_noisy = ["#AFA9EC", "#5DCAA5"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_c = ax.bar(x - w/2, clean_accs, w,
                    label="Clean test set",
                    color=[colours_clean[i] for i in range(len(model_labels))],
                    alpha=0.90)
    bars_n = ax.bar(x + w/2, noisy_accs, w,
                    label=f"Noisy (σ²={config.variance})",
                    color=[colours_noisy[i] for i in range(len(model_labels))],
                    alpha=0.90)

    # Annotate accuracy values on bars
    for bar, acc in zip(list(bars_c) + list(bars_n),
                        clean_accs + noisy_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    # Annotate drop arrows
    for i, (c, n, d) in enumerate(zip(clean_accs, noisy_accs, drops)):
        mid_x = x[i]
        ax.annotate(
            f"−{d:.1%}",
            xy=(mid_x, n + 0.01),
            ha="center", va="bottom",
            fontsize=8, color="#D85A30",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Top-1 accuracy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(
        f"Accuracy drop under Gaussian noise  "
        f"σ²={config.variance}  (σ={config.sigma:.3f})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plot] Accuracy comparison → {path}")
    plt.show()
    plt.close()


# ── Plot: confidence distribution shift ──────────────────────────────────────

def plot_confidence_shift(
    model        : torch.nn.Module,
    clean_loader : torch.utils.data.DataLoader,
    noisy_loader : NoisyDataLoader,
    model_label  : str,
    config       : NoiseConfig,
    save_name    : str = "noise_confidence_shift.png",
) -> None:
    """
    Overlay histogram: distribution of max softmax probability (confidence)
    on clean images vs noisy images.

    A well-calibrated model shifts confidence downward under noise.
    A brittle model maintains overconfident wrong predictions.

    Args:
        model:        Trained model on DEVICE.
        clean_loader: Clean test DataLoader.
        noisy_loader: NoisyDataLoader wrapping the same test set.
        model_label:  Name for the plot title.
        config:       NoiseConfig for title annotation.
        save_name:    Filename in outputs/plots/.
    """
    model.eval()
    clean_confs, noisy_confs = [], []

    with torch.no_grad():
        for (clean_imgs, _), (noisy_imgs, _) in zip(clean_loader, noisy_loader):
            clean_imgs = clean_imgs.to(DEVICE)
            noisy_imgs = noisy_imgs.to(DEVICE)

            clean_probs = F.softmax(model(clean_imgs), dim=1).max(dim=1).values
            noisy_probs = F.softmax(model(noisy_imgs), dim=1).max(dim=1).values

            clean_confs.extend(clean_probs.cpu().tolist())
            noisy_confs.extend(noisy_probs.cpu().tolist())

    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(clean_confs, bins=bins, alpha=0.65, color="#534AB7",
            label=f"Clean  (mean={np.mean(clean_confs):.3f})",
            density=True)
    ax.hist(noisy_confs, bins=bins, alpha=0.65, color="#D85A30",
            label=f"Noisy σ²={config.variance}  (mean={np.mean(noisy_confs):.3f})",
            density=True)

    ax.axvline(np.mean(clean_confs), color="#534AB7", lw=1.5,
               linestyle="--", alpha=0.8)
    ax.axvline(np.mean(noisy_confs), color="#D85A30", lw=1.5,
               linestyle="--", alpha=0.8)

    ax.set_title(
        f"{model_label} — confidence distribution shift under noise",
        fontsize=11,
    )
    ax.set_xlabel("Max softmax probability (predicted class)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plot] Confidence shift → {path}")
    plt.show()
    plt.close()


# ── Save full combined report ─────────────────────────────────────────────────

def save_combined_report(
    scratch_clean  : dict,
    scratch_noisy  : dict,
    transfer_clean : dict | None,
    transfer_noisy : dict | None,
    config         : NoiseConfig,
) -> Path:
    """
    Write a single JSON report combining clean baseline, noisy results,
    accuracy drops, and noise parameters for all models.
    """
    def drop(clean: float, noisy: float) -> float:
        return round(clean - noisy, 6)

    report = {
        "noise_parameters": {
            "variance"    : config.variance,
            "sigma"       : config.sigma,
            "distribution": config.distribution,
            "pixel_range" : config.pixel_range,
            "clipping"    : config.clip_after_noise,
            "seed"        : config.seed,
        },
        "scratch_cnn": {
            "clean_top1"  : scratch_clean.get("top1", 0.0),
            "noisy_top1"  : scratch_noisy["top1"],
            "top1_drop"   : drop(scratch_clean.get("top1", 0.0),
                                 scratch_noisy["top1"]),
            "clean_top5"  : scratch_clean.get("top5", 0.0),
            "noisy_top5"  : scratch_noisy["top5"],
            "top5_drop"   : drop(scratch_clean.get("top5", 0.0),
                                 scratch_noisy["top5"]),
            "noisy_loss"  : scratch_noisy["loss"],
            "avg_conf_correct"  : scratch_noisy["avg_confidence_correct"],
            "avg_conf_incorrect": scratch_noisy["avg_confidence_incorrect"],
        },
    }

    if transfer_noisy is not None and transfer_clean is not None:
        report["transfer_model"] = {
            "clean_top1"  : transfer_clean.get("top1", 0.0),
            "noisy_top1"  : transfer_noisy["top1"],
            "top1_drop"   : drop(transfer_clean.get("top1", 0.0),
                                 transfer_noisy["top1"]),
            "clean_top5"  : transfer_clean.get("top5", 0.0),
            "noisy_top5"  : transfer_noisy["top5"],
            "top5_drop"   : drop(transfer_clean.get("top5", 0.0),
                                 transfer_noisy["top5"]),
            "noisy_loss"  : transfer_noisy["loss"],
            "avg_conf_correct"  : transfer_noisy["avg_confidence_correct"],
            "avg_conf_incorrect": transfer_noisy["avg_confidence_incorrect"],
        }
        report["comparison"] = {
            "transfer_advantage_top1": round(
                transfer_noisy["top1"] - scratch_noisy["top1"], 6),
            "transfer_drop_vs_scratch_drop": round(
                drop(transfer_clean.get("top1",0), transfer_noisy["top1"]) -
                drop(scratch_clean.get("top1",0),  scratch_noisy["top1"]), 6),
            "interpretation": (
                "Positive transfer_advantage = transfer model more accurate "
                "under noise. Negative transfer_drop_vs_scratch_drop = "
                "transfer model loses less accuracy due to noise."
            ),
        }

    path = RESULT_DIR / "noise_combined_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [noise] Combined report → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    preflight(args)
    set_seed(args.seed)

    wall_start = time.perf_counter()

    # ── 1. Define and save noise schedule ─────────────────────────────────
    print("\n── Step 1: Define and save noise schedule ───────────────────")
    config = NoiseConfig(
        variance         = args.variance,
        pixel_range      = "0_1",
        distribution     = "gaussian",
        clip_after_noise = True,
        normalisation    = "cifar100",
        seed             = args.seed,
        description      = (
            f"Additive zero-mean Gaussian noise, sigma^2={args.variance}, "
            f"sigma={args.variance**0.5:.6f}. Applied in [0,1] pixel space "
            f"before re-normalisation. CIFAR-100 robustness evaluation. "
            f"Seed={args.seed}."
        ),
    )
    save_noise_schedule(config, filename="noise_schedule.json")

    # ── 2. Load data ───────────────────────────────────────────────────────
    print("\n── Step 2: Load test data ───────────────────────────────────")
    _, _, scratch_test = get_dataloaders(
        mode="scratch", batch_size=128, num_workers=0, input_size=32)

    if not args.no_transfer:
        # Transfer model needs its own test loader (224×224, ImageNet norm)
        # But noise must still be in [0,1] pixel space — handled inside
        # inject_noise using config.normalisation="cifar100".
        # For the transfer model we use a separate config with imagenet norm.
        transfer_config = NoiseConfig(
            variance         = args.variance,
            normalisation    = "imagenet",
            seed             = args.seed,
            clip_after_noise = True,
        )
        _, _, tl_test = get_dataloaders(
            mode="transfer", batch_size=64, num_workers=0,
            input_size=args.input_size)

    # ── 3. Verify noise statistics ─────────────────────────────────────────
    if args.verify_noise:
        print("\n── Step 3: Verify noise statistics ──────────────────────")
        verify_noise_statistics(scratch_test, config, n_batches=20)

    # ── 4. Save sample images ──────────────────────────────────────────────
    if args.save_noisy_images and not args.no_plots:
        print("\n── Step 4: Save clean vs noisy sample images ────────────")
        plot_noisy_samples(scratch_test, config, n_images=8)

    # ── 5. Load models ─────────────────────────────────────────────────────
    print("\n── Step 5: Load trained models ──────────────────────────────")
    scratch_model = build_scratch_cnn(
        variant=args.scratch_variant, num_classes=NUM_CLASSES).to(DEVICE)
    load_checkpoint(scratch_model, args.scratch_ckpt)
    scratch_model.eval()
    print(f"  Scratch CNN loaded from {args.scratch_ckpt.name}")

    if not args.no_transfer:
        tl_model = build_transfer_model(
            backbone=args.backbone, num_classes=NUM_CLASSES,
            pretrained=False).to(DEVICE)
        load_checkpoint(tl_model, args.transfer_ckpt)
        tl_model.eval()
        print(f"  Transfer model loaded from {args.transfer_ckpt.name}")

    # ── 6. Load clean baseline results ────────────────────────────────────
    print("\n── Step 6: Load clean baseline results ──────────────────────")
    scratch_clean, transfer_clean = {}, {}
    try:
        import json as _json
        with open(RESULT_DIR / "scratch_eval.json") as f:
            scratch_clean = _json.load(f)
        print(f"  Clean scratch top-1: {scratch_clean.get('top1',0):.4f}")
    except FileNotFoundError:
        print("  Clean scratch results not found — run compare.py first.")
        print("  Proceeding without baseline (drop cannot be computed).")

    if not args.no_transfer:
        try:
            with open(RESULT_DIR / "transfer_eval.json") as f:
                transfer_clean = _json.load(f)
            print(f"  Clean transfer top-1: {transfer_clean.get('top1',0):.4f}")
        except FileNotFoundError:
            print("  Clean transfer results not found.")

    # ── 7. Evaluate scratch CNN on noisy test set ──────────────────────────
    print("\n── Step 7: Evaluate scratch CNN on noisy test set ───────────")
    print(f"  Injecting Gaussian noise: σ²={config.variance}  σ={config.sigma:.4f}")

    scratch_noisy_loader = NoisyDataLoader(scratch_test, config)

    # Dry run: limit to 2 batches
    if args.dry_run:
        from itertools import islice
        class LimitedLoader:
            def __init__(self, loader, n): self.loader=loader; self.n=n
            def __iter__(self): return islice(iter(self.loader), self.n)
            def __len__(self): return min(len(self.loader), self.n)
            @property
            def dataset(self): return self.loader.dataset
        scratch_noisy_loader = LimitedLoader(scratch_noisy_loader, 2)
        print("  DRY RUN — 2 batches only")

    scratch_noisy = evaluate_noisy(
        scratch_model, scratch_noisy_loader,
        label="Scratch CNN", top_k=5)

    # Save scratch noisy results
    results_path = RESULT_DIR / "noise_results_scratch.json"
    with open(results_path, "w") as f:
        json.dump(scratch_noisy, f, indent=2)
    print(f"  Results saved → {results_path}")

    # ── 8. Evaluate transfer model on noisy test set ───────────────────────
    transfer_noisy = None
    if not args.no_transfer:
        print("\n── Step 8: Evaluate transfer model on noisy test set ────")

        tl_noisy_loader = NoisyDataLoader(tl_test, transfer_config)
        if args.dry_run:
            tl_noisy_loader = LimitedLoader(tl_noisy_loader, 2)

        transfer_noisy = evaluate_noisy(
            tl_model, tl_noisy_loader,
            label=f"Transfer ({args.backbone})", top_k=5)

        results_path = RESULT_DIR / "noise_results_transfer.json"
        with open(results_path, "w") as f:
            json.dump(transfer_noisy, f, indent=2)
        print(f"  Results saved → {results_path}")

    # ── 9. Plots ───────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n── Step 9: Generate plots ───────────────────────────────")

        # Accuracy comparison bar chart
        clean_map = {
            "Scratch CNN": scratch_clean.get("top1", 0.0),
        }
        noisy_map = {"Scratch CNN": scratch_noisy}
        if transfer_noisy:
            clean_map[f"Transfer ({args.backbone})"] = transfer_clean.get("top1", 0.0)
            noisy_map[f"Transfer ({args.backbone})"] = transfer_noisy

        plot_accuracy_comparison(clean_map, noisy_map, config)

        # Confidence distribution shift
        print("\n  Plotting confidence shift — Scratch CNN …")
        scratch_noisy_loader2 = NoisyDataLoader(scratch_test, config)
        plot_confidence_shift(
            scratch_model, scratch_test, scratch_noisy_loader2,
            model_label="Scratch CNN", config=config,
            save_name="noise_confidence_shift_scratch.png")

        if not args.no_transfer and transfer_noisy:
            print("  Plotting confidence shift — Transfer model …")
            tl_noisy_loader2 = NoisyDataLoader(tl_test, transfer_config)
            plot_confidence_shift(
                tl_model, tl_test, tl_noisy_loader2,
                model_label=f"Transfer ({args.backbone})", config=config,
                save_name="noise_confidence_shift_transfer.png")

    # ── 10. Save combined report ───────────────────────────────────────────
    print("\n── Step 10: Save combined report ────────────────────────────")
    save_combined_report(
        scratch_clean, scratch_noisy,
        transfer_clean if not args.no_transfer else None,
        transfer_noisy,
        config,
    )

    # ── 11. Final summary ──────────────────────────────────────────────────
    wall_total = time.perf_counter() - wall_start

    print("\n" + "═" * 60)
    print("  Noise Robustness Evaluation — Final Results")
    print("═" * 60)
    print(f"  Noise: σ²={config.variance}  σ={config.sigma:.6f}  "
          f"distribution=Gaussian  clip=True")
    print()
    print(f"  {'Metric':<35} {'Scratch CNN':>12} "
          + (f"{'Transfer':>14}" if not args.no_transfer else ""))
    print("  " + "─" * (35 + 12 + (14 if not args.no_transfer else 0) + 2))

    s_clean_t1 = scratch_clean.get("top1", float("nan"))
    s_clean_t5 = scratch_clean.get("top5", float("nan"))
    t_clean_t1 = transfer_clean.get("top1", float("nan")) if not args.no_transfer else float("nan")
    t_clean_t5 = transfer_clean.get("top5", float("nan")) if not args.no_transfer else float("nan")

    rows = [
        ("Clean top-1 accuracy",          s_clean_t1,             t_clean_t1),
        (f"Noisy top-1 (σ²={config.variance})", scratch_noisy["top1"], transfer_noisy["top1"] if transfer_noisy else float("nan")),
        ("Top-1 accuracy drop",           s_clean_t1 - scratch_noisy["top1"], t_clean_t1 - (transfer_noisy["top1"] if transfer_noisy else 0)),
        ("",                              None, None),
        ("Clean top-5 accuracy",          s_clean_t5,             t_clean_t5),
        (f"Noisy top-5 (σ²={config.variance})", scratch_noisy["top5"], transfer_noisy["top5"] if transfer_noisy else float("nan")),
        ("",                              None, None),
        ("Noisy test loss",               scratch_noisy["loss"],  transfer_noisy["loss"] if transfer_noisy else float("nan")),
        ("Avg confidence (correct pred)", scratch_noisy["avg_confidence_correct"],  transfer_noisy["avg_confidence_correct"] if transfer_noisy else float("nan")),
        ("Avg confidence (wrong pred)",   scratch_noisy["avg_confidence_incorrect"], transfer_noisy["avg_confidence_incorrect"] if transfer_noisy else float("nan")),
    ]

    for label, s_val, t_val in rows:
        if label == "":
            print()
            continue
        s_str = f"{s_val:.4f}" if s_val is not None and not (isinstance(s_val, float) and s_val != s_val) else "—"
        t_str = f"{t_val:.4f}" if t_val is not None and not (isinstance(t_val, float) and t_val != t_val) and not args.no_transfer else ("—" if args.no_transfer else f"{t_val:.4f}")
        print(f"  {label:<35} {s_str:>12}" + (f" {t_str:>14}" if not args.no_transfer else ""))

    print()
    print(f"  Total wall time : {format_time(wall_total)}")
    print()
    print("  Saved files:")
    print("    outputs/results/noise_schedule.json")
    print("    outputs/results/noise_results_scratch.json")
    if not args.no_transfer:
        print("    outputs/results/noise_results_transfer.json")
    print("    outputs/results/noise_combined_report.json")
    if not args.no_plots:
        print("    outputs/plots/noise_accuracy_comparison.png")
        print("    outputs/plots/noise_confidence_shift_scratch.png")
        if not args.no_transfer:
            print("    outputs/plots/noise_confidence_shift_transfer.png")
    print("═" * 60)


if __name__ == "__main__":
    main()