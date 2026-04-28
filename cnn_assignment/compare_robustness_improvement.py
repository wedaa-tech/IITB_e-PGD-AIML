"""
compare_robustness_improvement.py
───────────────────────────────────
Compare baseline (clean-trained) vs noise-augmented model on:
    1. Clean test set       — ensure clean accuracy is not sacrificed
    2. Noisy test set       — measure robustness improvement
    3. Accuracy retention   — noisy/clean ratio (fair cross-model comparison)

Usage:
    python compare_robustness_improvement.py               # scratch
    python compare_robustness_improvement.py --model transfer
    python compare_robustness_improvement.py --dry-run
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import (
    DEVICE, SCRATCH, TRANSFER, NUM_CLASSES,
    RESULT_DIR, PLOT_DIR, CKPT_DIR,
)
from src.utils import (
    set_seed, check_outputs_dir, format_time, load_history,
)
from src.dataset import get_dataloaders
from src.models.scratch_cnn import build_scratch_cnn
from src.models.transfer_model import build_transfer_model
from src.train import load_checkpoint
from src.noise import (
    NoiseConfig,
    load_noise_schedule,
    NoisyDataLoader,
    evaluate_noisy,
)
from src.noise_augment import load_augment_config


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",        default="scratch",
                        choices=["scratch", "transfer"])
    parser.add_argument("--scratch-variant", default="standard",
                        choices=["standard","small","residual"])
    parser.add_argument("--backbone",     default=TRANSFER["backbone"])
    parser.add_argument("--input-size",   type=int, default=TRANSFER["input_size"])
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    return parser.parse_args()


def evaluate_clean(model, loader) -> dict:
    """Evaluate top-1, top-5, and loss on a clean loader."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    t1, t5, loss_sum, total = 0, 0, 0.0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            t1    += (logits.argmax(1) == labels).sum().item()
            top5   = logits.topk(5, dim=1).indices
            t5    += sum(labels[i].item() in top5[i].tolist()
                         for i in range(len(labels)))
            loss_sum += criterion(logits, labels).item() * imgs.size(0)
            total    += imgs.size(0)

    return {"top1": t1/total, "top5": t5/total, "loss": loss_sum/total}


def plot_comparison(results: dict, noise_config: NoiseConfig, save: bool = True):
    """
    Four-panel figure comparing baseline vs noise-augmented model:
        Panel 1: Clean accuracy comparison
        Panel 2: Noisy accuracy comparison
        Panel 3: Accuracy drop (clean → noisy)
        Panel 4: Accuracy retention (noisy/clean %)
    """
    labels   = list(results.keys())
    colours  = ["#534AB7", "#1D9E75"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(
        f"Baseline vs Noise-Augmented Training  —  "
        f"Gaussian noise σ²={noise_config.variance}",
        fontsize=12, y=1.02,
    )

    metrics = [
        ("clean_top1",   "Clean top-1 accuracy"),
        ("noisy_top1",   f"Noisy top-1 (σ²={noise_config.variance})"),
        ("drop",         "Accuracy drop (clean − noisy)"),
        ("retention_pct","Accuracy retention (noisy/clean %)"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        vals = [results[lbl][key] for lbl in labels]

        if key == "retention_pct":
            display_vals = vals
            fmt = lambda v: f"{v:.1f}%"
        elif key == "drop":
            display_vals = vals
            fmt = lambda v: f"{v:.1%}"
            # Lower drop = better → invert colour assignment
            colours_used = [colours[1], colours[0]]
        else:
            display_vals = vals
            fmt = lambda v: f"{v:.1%}"
            colours_used = colours

        if key != "drop":
            colours_used = colours

        bars = ax.bar(labels, display_vals,
                      color=[colours_used[i] for i in range(len(labels))],
                      alpha=0.85, width=0.5)
        for bar, val in zip(bars, display_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(display_vals) * 0.02,
                    fmt(val), ha="center", va="bottom", fontsize=9)

        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(display_vals) * 1.25 + 0.05)
        if key not in ("drop", "retention_pct"):
            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelsize=8, rotation=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    plt.tight_layout()
    if save:
        path = PLOT_DIR / "robustness_improvement_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [plot] Comparison → {path}")
    plt.show()
    plt.close()


def main():
    args = parse_args()
    check_outputs_dir()
    set_seed(42)

    # ── Load noise schedule ────────────────────────────────────────────────
    try:
        noise_config = load_noise_schedule("noise_schedule.json")
    except FileNotFoundError:
        print("  noise_schedule.json not found — using defaults (σ²=0.05)")
        noise_config = NoiseConfig(variance=0.05, normalisation="cifar100")

    # ── Load data ──────────────────────────────────────────────────────────
    mode       = "scratch" if args.model == "scratch" else "transfer"
    input_size = 32 if args.model == "scratch" else args.input_size
    _, _, test_loader = get_dataloaders(
        mode=mode, batch_size=128, num_workers=0, input_size=input_size)

    if args.dry_run:
        from itertools import islice
        class OneLoader:
            def __init__(self, l): self._l = l
            def __iter__(self): return islice(iter(self._l), 2)
            def __len__(self): return 2
            @property
            def dataset(self): return self._l.dataset
        test_loader = OneLoader(test_loader)

    # Noisy loader
    norm = "cifar100" if args.model == "scratch" else "imagenet"
    noisy_cfg    = NoiseConfig(
        variance=noise_config.variance, normalisation=norm, seed=42)
    noisy_loader = NoisyDataLoader(test_loader, noisy_cfg)

    results = {}

    # ── Baseline model ─────────────────────────────────────────────────────
    print("\n── Evaluating BASELINE model ────────────────────────────────")
    if args.model == "scratch":
        baseline = build_scratch_cnn(
            variant=args.scratch_variant, num_classes=NUM_CLASSES).to(DEVICE)
        load_checkpoint(baseline, SCRATCH["checkpoint"])
        aug_ckpt = CKPT_DIR / "scratch_noisy_augmented_best.pth"
    else:
        baseline = build_transfer_model(
            args.backbone, NUM_CLASSES, pretrained=False).to(DEVICE)
        load_checkpoint(baseline, TRANSFER["checkpoint"])
        aug_ckpt = CKPT_DIR / "transfer_noisy_augmented_best.pth"

    baseline.eval()
    b_clean = evaluate_clean(baseline, test_loader)
    b_noisy = evaluate_noisy(baseline, noisy_loader, label="Baseline", top_k=5)
    results["Baseline"] = {
        "clean_top1"    : b_clean["top1"],
        "noisy_top1"    : b_noisy["top1"],
        "drop"          : b_clean["top1"] - b_noisy["top1"],
        "retention_pct" : b_noisy["top1"] / b_clean["top1"] * 100,
        "clean_top5"    : b_clean["top5"],
        "noisy_top5"    : b_noisy["top5"],
    }

    # ── Noise-augmented model ──────────────────────────────────────────────
    print("\n── Evaluating NOISE-AUGMENTED model ─────────────────────────")
    if not aug_ckpt.exists():
        print(f"  Augmented checkpoint not found: {aug_ckpt}")
        print(f"  Run:  python run_noise_augment_training.py --model {args.model}")
        return

    if args.model == "scratch":
        aug_model = build_scratch_cnn(
            variant=args.scratch_variant, num_classes=NUM_CLASSES).to(DEVICE)
    else:
        aug_model = build_transfer_model(
            args.backbone, NUM_CLASSES, pretrained=False).to(DEVICE)

    load_checkpoint(aug_model, aug_ckpt)
    aug_model.eval()

    # Re-initialise noisy loader with same seed
    noisy_loader2 = NoisyDataLoader(test_loader, noisy_cfg)

    a_clean = evaluate_clean(aug_model, test_loader)
    a_noisy = evaluate_noisy(aug_model, noisy_loader2,
                              label="Noise-Augmented", top_k=5)
    results["Noise-Augmented"] = {
        "clean_top1"    : a_clean["top1"],
        "noisy_top1"    : a_noisy["top1"],
        "drop"          : a_clean["top1"] - a_noisy["top1"],
        "retention_pct" : a_noisy["top1"] / a_clean["top1"] * 100,
        "clean_top5"    : a_clean["top5"],
        "noisy_top5"    : a_noisy["top5"],
    }

    # ── Save results ───────────────────────────────────────────────────────
    out_path = RESULT_DIR / f"robustness_improvement_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    if not args.no_plots:
        plot_comparison(results, noisy_cfg)

    # ── Final table ────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"  Robustness Improvement — {args.model.upper()}")
    print("═" * 62)
    print(f"  {'Metric':<35} {'Baseline':>12} {'Noise-Aug':>12}")
    print("  " + "─" * 60)

    rows = [
        ("Clean top-1 accuracy",       "clean_top1",    "{:.4f}"),
        ("Noisy top-1 accuracy",       "noisy_top1",    "{:.4f}"),
        ("Accuracy drop (pp)",         "drop",          "{:.4f}"),
        ("Accuracy retention (%)",     "retention_pct", "{:.2f}"),
        ("Clean top-5 accuracy",       "clean_top5",    "{:.4f}"),
        ("Noisy top-5 accuracy",       "noisy_top5",    "{:.4f}"),
    ]
    for label, key, fmt in rows:
        b = fmt.format(results["Baseline"][key])
        a = fmt.format(results["Noise-Augmented"][key])
        print(f"  {label:<35} {b:>12} {a:>12}")

    improvement = (results["Noise-Augmented"]["noisy_top1"]
                   - results["Baseline"]["noisy_top1"])
    ret_gain    = (results["Noise-Augmented"]["retention_pct"]
                   - results["Baseline"]["retention_pct"])
    clean_delta = (results["Noise-Augmented"]["clean_top1"]
                   - results["Baseline"]["clean_top1"])

    print("  " + "─" * 60)
    print(f"  {'Noisy accuracy improvement':35} {improvement:>+12.4f}")
    print(f"  {'Retention gain (pp)':35} {ret_gain:>+12.2f}")
    print(f"  {'Clean accuracy delta':35} {clean_delta:>+12.4f}")
    print(f"\n  Interpretation:")
    if improvement > 0:
        print(f"    Noise augmentation improved noisy accuracy by "
              f"{improvement*100:.2f}pp")
    if abs(clean_delta) < 0.02:
        print(f"    Clean accuracy was preserved "
              f"(delta={clean_delta*100:+.2f}pp — within acceptable range)")
    elif clean_delta < -0.02:
        print(f"    Clean accuracy degraded by {abs(clean_delta)*100:.2f}pp "
              f"— consider reducing noise_prob")
    print("═" * 62)


if __name__ == "__main__":
    main()