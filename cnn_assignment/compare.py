"""
compare.py
──────────
Final comparison entry point: load both trained models, run full
evaluation on the test set, and generate every plot and table
needed for the assignment report.

Must be run AFTER both training scripts have completed:
    python run_scratch.py      → outputs/checkpoints/scratch_best.pth
    python run_transfer.py     → outputs/checkpoints/transfer_best.pth

Usage:
    python compare.py                          # full comparison
    python compare.py --dry-run                # 1 test batch only (fast check)
    python compare.py --no-plots               # metrics only, skip figures
    python compare.py --noise-only             # robustness test only
    python compare.py --scratch-variant small  # if you trained a small variant
    python compare.py --backbone resnet34      # if you used a different backbone

    PYTORCH_ENABLE_MPS_FALLBACK=1 python compare.py   # if MPS errors occur

Outputs written to outputs/:
    plots/learning_curves.png
    plots/confusion_scratch.png
    plots/confusion_transfer.png
    plots/per_class_scratch.png
    plots/per_class_transfer.png
    plots/robustness.png
    plots/top_failures_scratch.png
    plots/top_failures_transfer.png
    plots/feature_tsne_scratch.png      (if --tsne flag set)
    plots/feature_tsne_transfer.png     (if --tsne flag set)
    results/comparison_table.csv
    results/scratch_eval.json
    results/transfer_eval.json
    results/robustness_scratch.json
    results/robustness_transfer.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Project imports ───────────────────────────────────────────────────────────
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
    save_results,
    load_history,
    export_results_csv,
)
from src.dataset import get_dataloaders
from src.models.scratch_cnn import build_scratch_cnn, count_parameters as scratch_count
from src.models.transfer_model import build_transfer_model, count_parameters as tl_count
from src.train import load_checkpoint
from src.evaluate import (
    evaluate,
    evaluate_per_class,
    evaluate_robustness,
    measure_inference_time,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_robustness,
    plot_top_failures,
    print_comparison_table,
)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare scratch CNN vs transfer learning on CIFAR-100 test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--scratch-variant",
        type    = str,
        default = "standard",
        choices = ["standard", "small", "residual"],
        help    = "Scratch CNN variant to load (must match what was trained).",
    )
    parser.add_argument(
        "--backbone",
        type    = str,
        default = TRANSFER["backbone"],
        help    = "Transfer learning backbone name (must match what was trained).",
    )
    parser.add_argument(
        "--input-size",
        type    = int,
        default = TRANSFER["input_size"],
        help    = "Input size used when training the transfer model.",
    )

    # Checkpoint overrides
    parser.add_argument(
        "--scratch-checkpoint",
        type    = Path,
        default = SCRATCH["checkpoint"],
        help    = "Path to the scratch CNN checkpoint.",
    )
    parser.add_argument(
        "--transfer-checkpoint",
        type    = Path,
        default = TRANSFER["checkpoint"],
        help    = "Path to the transfer learning checkpoint.",
    )

    # History file overrides
    parser.add_argument(
        "--scratch-history",
        type    = str,
        default = None,
        help    = "Scratch history JSON filename in results/. "
                  "Auto-detected from --scratch-variant if not set.",
    )
    parser.add_argument(
        "--transfer-history",
        type    = str,
        default = None,
        help    = "Transfer history JSON filename in results/. "
                  "Auto-detected from --backbone if not set.",
    )

    # Evaluation options
    parser.add_argument(
        "--noise-levels",
        type    = float,
        nargs   = "+",
        default = [0.0, 0.05, 0.1, 0.2, 0.3],
        help    = "Gaussian noise σ values for robustness evaluation.",
    )
    parser.add_argument(
        "--top-k",
        type    = int,
        default = 5,
        help    = "k for top-k accuracy (default 5).",
    )
    parser.add_argument(
        "--n-failures",
        type    = int,
        default = 25,
        help    = "Number of top failure images to display per model.",
    )

    # Feature visualisation
    parser.add_argument(
        "--tsne",
        action  = "store_true",
        help    = "Run t-SNE on backbone features and save scatter plots "
                  "(slow — adds ~5 min on CPU).",
    )
    parser.add_argument(
        "--tsne-samples",
        type    = int,
        default = 2000,
        help    = "Number of test samples to include in t-SNE plots.",
    )

    # Speed controls
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Evaluate on 1 batch only — verifies the pipeline fast.",
    )
    parser.add_argument(
        "--no-plots",
        action  = "store_true",
        help    = "Skip all figure generation — print metrics only.",
    )
    parser.add_argument(
        "--noise-only",
        action  = "store_true",
        help    = "Run robustness test only, skip other evaluations.",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Random seed (affects t-SNE initialisation).",
    )

    return parser.parse_args()


# ── Pre-flight ────────────────────────────────────────────────────────────────

def preflight(args: argparse.Namespace) -> None:
    """Verify checkpoints, data, and output directories exist."""
    print("\n" + "═" * 62)
    print("  Pre-flight checks")
    print("═" * 62)

    check_outputs_dir()

    # Checkpoints
    for label, path in [
        ("Scratch checkpoint",   args.scratch_checkpoint),
        ("Transfer checkpoint",  args.transfer_checkpoint),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"\n  {label} not found: {path}\n"
                f"  Run the corresponding training script first:\n"
                f"    Scratch  → python run_scratch.py\n"
                f"    Transfer → python run_transfer.py"
            )
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"  {label:<24} {path.name}  ({size_mb:.1f} MB)  ✓")

    # CIFAR-100 data
    cifar_path = Path("data") / "cifar-100-python"
    if not cifar_path.exists():
        raise FileNotFoundError(
            f"\n  CIFAR-100 data not found at: {cifar_path}\n"
            f"  Extract with:  cd data && tar -xzf cifar-100-python.tar.gz"
        )
    print(f"  {'CIFAR-100 data':<24} {cifar_path}  ✓")

    print(f"  {'Device':<24} {DEVICE}")
    print("\n  All pre-flight checks passed.\n")


# ── History loader ─────────────────────────────────────────────────────────────

def load_histories(args: argparse.Namespace) -> tuple[dict | None, dict | None]:
    """
    Load training histories for both models.
    Returns None for a model if its history file is not found
    (compare.py continues without the learning curve plot).
    """
    scratch_file  = (args.scratch_history
                     or f"scratch_{args.scratch_variant}_history.json")
    transfer_file = (args.transfer_history
                     or f"transfer_{args.backbone.replace('/', '_')}_history.json")

    histories = {}
    for label, filename in [
        ("scratch",  scratch_file),
        ("transfer", transfer_file),
    ]:
        try:
            histories[label] = load_history(filename)
        except FileNotFoundError:
            print(f"  [compare] History not found for {label}: {filename}")
            print(f"  [compare] Learning curve plot will be skipped for {label}.")
            histories[label] = None

    return histories["scratch"], histories["transfer"]


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_scratch_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build scratch CNN, load best checkpoint, set to eval mode."""
    print("\n── Loading scratch CNN ───────────────────────────────────────")
    model = build_scratch_cnn(
        variant     = args.scratch_variant,
        num_classes = NUM_CLASSES,
    ).to(DEVICE)
    load_checkpoint(model, args.scratch_checkpoint)
    model.eval()
    scratch_count(model)
    return model


def load_transfer_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build transfer backbone, load best checkpoint, set to eval mode."""
    print("\n── Loading transfer model ────────────────────────────────────")
    model = build_transfer_model(
        backbone    = args.backbone,
        num_classes = NUM_CLASSES,
        pretrained  = False,    # weights come from checkpoint, not ImageNet
    ).to(DEVICE)
    load_checkpoint(model, args.transfer_checkpoint)
    model.eval()
    tl_count(model)
    return model


# ── t-SNE visualisation ───────────────────────────────────────────────────────

def plot_tsne(
    model       : torch.nn.Module,
    loader      : torch.utils.data.DataLoader,
    n_samples   : int,
    save_name   : str,
    title       : str,
    has_extract : bool = True,
) -> None:
    """
    Collect backbone feature vectors, run t-SNE, and save a scatter plot.
    Colours 20 CIFAR-100 superclasses rather than all 100 classes so the
    plot remains readable.

    Args:
        model:       Trained model with extract_features() method.
        loader:      Test DataLoader (no augmentation, no shuffle).
        n_samples:   Max number of test images to embed.
        save_name:   Filename inside outputs/plots/.
        title:       Plot title.
        has_extract: If False, uses the full forward pass logits instead.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [tsne] scikit-learn not found — skipping t-SNE.")
        return

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    print(f"  [tsne] Collecting features for up to {n_samples} samples …")

    features_list, labels_list = [], []
    collected = 0

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            if collected >= n_samples:
                break
            imgs = imgs.to(DEVICE)
            if has_extract and hasattr(model, "extract_features"):
                feats = model.extract_features(imgs)
            else:
                feats = model(imgs)     # use logits as fallback
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            collected += imgs.size(0)

    features = np.vstack(features_list)[:n_samples]
    labels   = np.concatenate(labels_list)[:n_samples]

    print(f"  [tsne] Feature matrix: {features.shape}  "
          f"Running t-SNE (may take 2–5 min) …")
    t0      = time.perf_counter()
    tsne    = TSNE(n_components=2, perplexity=40, n_iter=1000,
                   random_state=42, n_jobs=-1)
    embedded = tsne.fit_transform(features)
    print(f"  [tsne] Done in {format_time(time.perf_counter() - t0)}")

    # Map fine-grained labels → superclass index (0–19)
    # CIFAR-100 superclass ordering matches the dataset's coarse_labels
    superclass_of = np.array([lbl // 5 for lbl in labels])
    n_super       = 20
    colours       = cm.tab20(np.linspace(0, 1, n_super))

    fig, ax = plt.subplots(figsize=(9, 8))
    for sc in range(n_super):
        mask = superclass_of == sc
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s        = 4,
            alpha    = 0.55,
            color    = colours[sc],
            label    = f"SC {sc:02d}",
            linewidths = 0,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.legend(
        markerscale = 3,
        fontsize    = 6.5,
        ncol        = 4,
        loc         = "upper right",
        framealpha  = 0.7,
    )
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [tsne] Saved → {path}")
    plt.show()
    plt.close()


# ── Phase boundary helper ─────────────────────────────────────────────────────

def add_phase_boundary(ax, history: dict) -> None:
    """
    Draw a vertical dashed line on a matplotlib Axes at the Phase A / Phase B
    boundary, if the history dict contains the 'phase_a_epochs' key written
    by merge_histories() in run_transfer.py.

    Args:
        ax:      Matplotlib Axes to annotate.
        history: History dict loaded from JSON.
    """
    boundary = history.get("phase_a_epochs")
    if boundary and boundary > 0:
        ax.axvline(
            x         = boundary + 0.5,
            color     = "#D85A30",
            lw        = 1.2,
            linestyle = "--",
            alpha     = 0.7,
            label     = "Phase A → B",
        )
        ax.legend(fontsize=8)


# ── Learning curve plot with phase boundary ───────────────────────────────────

def plot_curves_with_boundary(
    scratch_hist  : dict | None,
    transfer_hist : dict | None,
) -> None:
    """
    Plot learning curves for both models. If the transfer history contains
    a phase boundary marker, draw a vertical divider line.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    histories_present = {
        k: v for k, v in [
            ("Scratch CNN",                    scratch_hist),
            (f"Transfer (EfficientNet-B0)",    transfer_hist),
        ] if v is not None
    }

    if not histories_present:
        print("  [compare] No history files found — skipping learning curves.")
        return

    colours = ["#534AB7", "#0F6E56"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Training history — Scratch CNN vs Transfer Learning",
                 fontsize=12, y=1.01)

    for i, (name, h) in enumerate(histories_present.items()):
        c      = colours[i % len(colours)]
        epochs = range(1, len(h["val_acc"]) + 1)

        axes[0].plot(epochs, h["val_acc"],   color=c, lw=2,
                     label=f"{name} val")
        axes[0].plot(epochs, h["train_acc"], color=c, lw=1,
                     linestyle="--", alpha=0.5,
                     label=f"{name} train")
        axes[1].plot(epochs, h["val_loss"],   color=c, lw=2,
                     label=f"{name} val")
        axes[1].plot(epochs, h["train_loss"], color=c, lw=1,
                     linestyle="--", alpha=0.5,
                     label=f"{name} train")

        # Phase boundary for transfer model
        if "phase_a_epochs" in h:
            for ax in axes:
                add_phase_boundary(ax, h)

    for ax, ylabel, title in zip(
        axes,
        ["Top-1 accuracy", "Cross-entropy loss"],
        ["Accuracy (solid=val, dashed=train)",
         "Loss (solid=val, dashed=train)"],
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    axes[0].yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    path = PLOT_DIR / "learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot] Learning curves → {path}")
    plt.show()
    plt.close()


# ── Superclass accuracy breakdown ─────────────────────────────────────────────

def plot_superclass_accuracy(
    eval_scratch  : dict,
    eval_transfer : dict,
    save_name     : str = "superclass_accuracy.png",
) -> None:
    """
    Group the 100 per-class accuracies into 20 superclasses and plot a
    grouped bar chart comparing scratch vs transfer. Immediately reveals
    which semantic categories each model handles better.

    Args:
        eval_scratch:  evaluate() result dict for scratch CNN.
        eval_transfer: evaluate() result dict for transfer model.
        save_name:     Filename inside outputs/plots/.
    """
    import matplotlib.pyplot as plt

    from src.evaluate import SUPERCLASSES

    scratch_preds  = eval_scratch["preds"]
    scratch_labels = eval_scratch["labels"]
    tl_preds       = eval_transfer["preds"]
    tl_labels      = eval_transfer["labels"]

    super_names   = [sc[0] for sc in SUPERCLASSES]
    scratch_accs  = []
    transfer_accs = []

    for sc_idx, (_, fine_names) in enumerate(SUPERCLASSES):
        # Classes in this superclass: indices sc_idx*5 … sc_idx*5+4
        class_indices = list(range(sc_idx * 5, sc_idx * 5 + 5))

        for preds, labels, accs_list in [
            (scratch_preds,  scratch_labels, scratch_accs),
            (tl_preds,       tl_labels,      transfer_accs),
        ]:
            mask    = np.isin(labels, class_indices)
            correct = (preds[mask] == labels[mask]).sum()
            total   = mask.sum()
            accs_list.append(correct / total if total > 0 else 0.0)

    # Sort by transfer accuracy descending for readability
    order = np.argsort(transfer_accs)[::-1]
    names_sorted = [super_names[i] for i in order]
    s_sorted     = [scratch_accs[i]  for i in order]
    t_sorted     = [transfer_accs[i] for i in order]

    x    = np.arange(len(names_sorted))
    w    = 0.38
    fig, ax = plt.subplots(figsize=(14, 5))

    bars_s = ax.bar(x - w/2, s_sorted, w,
                    label="Scratch CNN",    color="#534AB7", alpha=0.85)
    bars_t = ax.bar(x + w/2, t_sorted, w,
                    label="Transfer (EfficientNet-B0)", color="#0F6E56", alpha=0.85)

    ax.set_title("Superclass accuracy — Scratch CNN vs Transfer Learning",
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 accuracy", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plot] Superclass accuracy → {path}")
    plt.show()
    plt.close()


# ── Confidence distribution plot ──────────────────────────────────────────────

def plot_confidence_distribution(
    eval_scratch  : dict,
    eval_transfer : dict,
    save_name     : str = "confidence_distribution.png",
) -> None:
    """
    Histogram of predicted confidence (max softmax probability) split by
    correct vs incorrect predictions for each model.
    Well-calibrated models should show high confidence for correct
    predictions and lower confidence for mistakes.

    Args:
        eval_scratch:  evaluate() result dict for scratch CNN.
        eval_transfer: evaluate() result dict for transfer model.
        save_name:     Filename inside outputs/plots/.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.suptitle("Prediction confidence distribution", fontsize=12, y=1.01)

    for ax, result, title, colour in zip(
        axes,
        [eval_scratch,   eval_transfer],
        ["Scratch CNN",  "Transfer (EfficientNet-B0)"],
        ["#534AB7",      "#0F6E56"],
    ):
        preds   = result["preds"]
        labels  = result["labels"]
        probs   = result["probs"]

        correct_conf   = probs[preds == labels]
        incorrect_conf = probs[preds != labels]

        bins = np.linspace(0, 1, 41)
        ax.hist(correct_conf,   bins=bins, alpha=0.7,
                color=colour,  label=f"Correct  (n={len(correct_conf):,})",
                density=True)
        ax.hist(incorrect_conf, bins=bins, alpha=0.55,
                color="#D85A30", label=f"Incorrect (n={len(incorrect_conf):,})",
                density=True)
        ax.axvline(correct_conf.mean(),   color=colour,   lw=1.5,
                   linestyle="--", alpha=0.8,
                   label=f"Mean correct: {correct_conf.mean():.2f}")
        ax.axvline(incorrect_conf.mean(), color="#D85A30", lw=1.5,
                   linestyle=":",  alpha=0.8,
                   label=f"Mean incorrect: {incorrect_conf.mean():.2f}")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted probability of top-1 class", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plot] Confidence distribution → {path}")
    plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Pre-flight ─────────────────────────────────────────────────────────
    preflight(args)
    set_seed(args.seed)
    log_system_info()

    wall_start = time.perf_counter()

    # ── Data ───────────────────────────────────────────────────────────────
    print("── Loading data ─────────────────────────────────────────────")

    # Scratch uses 32×32, no upscaling
    _, _, scratch_test = get_dataloaders(
        mode        = "scratch",
        batch_size  = 128,
        num_workers = 0,
        input_size  = 32,
    )

    # Transfer uses upscaled images (default 224×224)
    _, _, tl_test = get_dataloaders(
        mode        = "transfer",
        batch_size  = 64,
        num_workers = 0,
        input_size  = args.input_size,
    )

    # Dry-run: wrap loaders to yield 1 batch only
    if args.dry_run:
        print("\n  DRY RUN — evaluating on 1 batch only.")
        from itertools import islice
        from torch.utils.data import DataLoader

        def single_batch(loader):
            """Yield only the first batch of any DataLoader."""
            for batch in loader:
                yield batch
                return

        class OneBatchLoader:
            """Wraps a DataLoader to yield exactly one batch."""
            def __init__(self, loader): self._loader = loader
            def __iter__(self):         return single_batch(self._loader)
            def __len__(self):          return 1
            @property
            def dataset(self):          return self._loader.dataset

        scratch_test = OneBatchLoader(scratch_test)
        tl_test      = OneBatchLoader(tl_test)

    # ── Load models ────────────────────────────────────────────────────────
    scratch_model  = load_scratch_model(args)
    transfer_model = load_transfer_model(args)

    # ── Load training histories ────────────────────────────────────────────
    print("\n── Loading training histories ────────────────────────────────")
    scratch_hist, transfer_hist = load_histories(args)

    # ─────────────────────────────────────────────────────────────────────
    # ROBUSTNESS ONLY MODE
    # ─────────────────────────────────────────────────────────────────────
    if args.noise_only:
        print("\n── Robustness evaluation (--noise-only) ─────────────────")
        print("  Scratch CNN:")
        rob_scratch = evaluate_robustness(
            scratch_model, scratch_test, args.noise_levels)
        print("  Transfer model:")
        rob_transfer = evaluate_robustness(
            transfer_model, tl_test, args.noise_levels)

        if not args.no_plots:
            plot_robustness(
                {"Scratch CNN": rob_scratch,
                 f"Transfer ({args.backbone})": rob_transfer},
                save_name="robustness.png",
            )

        # Save JSON
        for label, rob in [("scratch", rob_scratch), ("transfer", rob_transfer)]:
            path = RESULT_DIR / f"robustness_{label}.json"
            with open(path, "w") as f:
                json.dump({str(k): v for k, v in rob.items()}, f, indent=2)
            print(f"  [compare] Robustness saved → {path}")

        print(f"\n  Total time: {format_time(time.perf_counter() - wall_start)}")
        return

    # ─────────────────────────────────────────────────────────────────────
    # FULL EVALUATION
    # ─────────────────────────────────────────────────────────────────────

    # ── Accuracy evaluation ────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  Accuracy evaluation")
    print("═" * 62)

    print("\n  Evaluating scratch CNN on test set …")
    t0            = time.perf_counter()
    eval_scratch  = evaluate(scratch_model,  scratch_test, top_k=args.top_k)
    scratch_eval_time = time.perf_counter() - t0
    print(f"  Done in {format_time(scratch_eval_time)}")
    print(f"  Top-1 : {eval_scratch['top1']:.4f}  "
          f"Top-{args.top_k} : {eval_scratch['top5']:.4f}  "
          f"Loss : {eval_scratch['loss']:.4f}")

    print(f"\n  Evaluating transfer model ({args.backbone}) on test set …")
    t0            = time.perf_counter()
    eval_transfer = evaluate(transfer_model, tl_test,      top_k=args.top_k)
    tl_eval_time  = time.perf_counter() - t0
    print(f"  Done in {format_time(tl_eval_time)}")
    print(f"  Top-1 : {eval_transfer['top1']:.4f}  "
          f"Top-{args.top_k} : {eval_transfer['top5']:.4f}  "
          f"Loss : {eval_transfer['loss']:.4f}")

    # ── Save scalar evaluation results ─────────────────────────────────────
    save_results(eval_scratch,  "scratch_eval.json")
    save_results(eval_transfer, "transfer_eval.json")

    # ── Per-class accuracy ─────────────────────────────────────────────────
    print("\n── Per-class accuracy ───────────────────────────────────────")
    pc_scratch  = evaluate_per_class(eval_scratch,  save=True)
    pc_transfer = evaluate_per_class(eval_transfer, save=True)

    # Save per-class to separate CSVs for report
    import csv
    for label, pc in [("scratch", pc_scratch), ("transfer", pc_transfer)]:
        path = RESULT_DIR / f"per_class_{label}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "accuracy"])
            for name, acc in pc.items():
                w.writerow([name, f"{acc:.4f}"])
        print(f"  [compare] Per-class saved → {path}")

    # Worst and best 5 classes for each model
    print("\n  Scratch CNN — 5 worst classes:")
    for name, acc in list(pc_scratch.items())[:5]:
        print(f"    {name:<25}  {acc:.3f}")
    print("  Scratch CNN — 5 best classes:")
    for name, acc in list(pc_scratch.items())[-5:]:
        print(f"    {name:<25}  {acc:.3f}")

    print(f"\n  Transfer ({args.backbone}) — 5 worst classes:")
    for name, acc in list(pc_transfer.items())[:5]:
        print(f"    {name:<25}  {acc:.3f}")
    print(f"  Transfer ({args.backbone}) — 5 best classes:")
    for name, acc in list(pc_transfer.items())[-5:]:
        print(f"    {name:<25}  {acc:.3f}")

    # ── Robustness evaluation ──────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  Robustness evaluation — Gaussian noise")
    print("═" * 62)

    print("\n  Scratch CNN:")
    rob_scratch  = evaluate_robustness(
        scratch_model, scratch_test, args.noise_levels)

    print(f"\n  Transfer ({args.backbone}):")
    rob_transfer = evaluate_robustness(
        transfer_model, tl_test, args.noise_levels)

    # Save robustness JSON
    for label, rob in [("scratch", rob_scratch), ("transfer", rob_transfer)]:
        path = RESULT_DIR / f"robustness_{label}.json"
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in rob.items()}, f, indent=2)
        print(f"  [compare] Robustness saved → {path}")

    # ── Inference time ─────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  Inference timing")
    print("═" * 62)

    print("\n  Scratch CNN:")
    timing_scratch  = measure_inference_time(
        scratch_model,  scratch_test, n_batches=20)

    print(f"\n  Transfer ({args.backbone}):")
    timing_transfer = measure_inference_time(
        transfer_model, tl_test,      n_batches=20)

    # ── Parameter counts ───────────────────────────────────────────────────
    scratch_params  = sum(p.numel() for p in scratch_model.parameters())
    transfer_params = sum(p.numel() for p in transfer_model.parameters())

    # ── Epoch timing from history ──────────────────────────────────────────
    def _avg_epoch_time(hist: dict | None) -> float:
        if hist is None:
            return 0.0
        times = hist.get("epoch_time", [])
        return float(np.mean(times)) if times else 0.0

    avg_scratch_ep  = _avg_epoch_time(scratch_hist)
    avg_transfer_ep = _avg_epoch_time(transfer_hist)

    # ── Comparison table ───────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  Final comparison table")
    print("═" * 62)

    comparison = {
        f"Scratch CNN ({args.scratch_variant})": {
            "top1"           : eval_scratch["top1"],
            "top5"           : eval_scratch["top5"],
            "loss"           : eval_scratch["loss"],
            "params"         : scratch_params,
            "avg_epoch_time" : avg_scratch_ep,
            "ms_per_image"   : timing_scratch["ms_per_image"],
        },
        f"Transfer ({args.backbone})": {
            "top1"           : eval_transfer["top1"],
            "top5"           : eval_transfer["top5"],
            "loss"           : eval_transfer["loss"],
            "params"         : transfer_params,
            "avg_epoch_time" : avg_transfer_ep,
            "ms_per_image"   : timing_transfer["ms_per_image"],
        },
    }

    print_comparison_table(comparison)
    export_results_csv(comparison, filename="comparison_table.csv")

    # ─────────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────────
    if args.no_plots:
        print("\n  --no-plots set — skipping all figures.")
    else:
        print("\n" + "═" * 62)
        print("  Generating plots")
        print("═" * 62)

        # 1. Learning curves
        print("\n  1/8  Learning curves …")
        plot_curves_with_boundary(scratch_hist, transfer_hist)

        # 2. Confusion matrices
        print("\n  2/8  Confusion matrix — Scratch CNN …")
        plot_confusion_matrix(
            eval_scratch["labels"],
            eval_scratch["preds"],
            title     = f"Scratch CNN ({args.scratch_variant}) — confusion matrix",
            save_name = "confusion_scratch.png",
            show      = True,
        )

        print("\n  3/8  Confusion matrix — Transfer model …")
        plot_confusion_matrix(
            eval_transfer["labels"],
            eval_transfer["preds"],
            title     = f"Transfer ({args.backbone}) — confusion matrix",
            save_name = "confusion_transfer.png",
            show      = True,
        )

        # 3. Per-class accuracy bars
        print("\n  4/8  Per-class accuracy — Scratch CNN …")
        plot_per_class_accuracy(
            pc_scratch,
            n_show    = 20,
            title     = f"Per-class accuracy — Scratch CNN ({args.scratch_variant})",
            save_name = "per_class_scratch.png",
            show      = True,
        )

        print("\n  5/8  Per-class accuracy — Transfer model …")
        plot_per_class_accuracy(
            pc_transfer,
            n_show    = 20,
            title     = f"Per-class accuracy — Transfer ({args.backbone})",
            save_name = "per_class_transfer.png",
            show      = True,
        )

        # 4. Robustness
        print("\n  6/8  Robustness plot …")
        plot_robustness(
            {
                f"Scratch CNN ({args.scratch_variant})": rob_scratch,
                f"Transfer ({args.backbone})"           : rob_transfer,
            },
            save_name = "robustness.png",
            show      = True,
        )

        # 5. Top failure grids
        print("\n  7/8  Top failure images — Scratch CNN …")
        plot_top_failures(
            eval_scratch,
            scratch_test,
            n_images  = args.n_failures,
            title     = f"Most confident wrong predictions — Scratch CNN ({args.scratch_variant})",
            save_name = "top_failures_scratch.png",
            show      = True,
        )

        print(f"\n  8/8  Top failure images — Transfer model …")
        plot_top_failures(
            eval_transfer,
            tl_test,
            n_images  = args.n_failures,
            title     = f"Most confident wrong predictions — Transfer ({args.backbone})",
            save_name = "top_failures_transfer.png",
            show      = True,
        )

        # 6. Superclass accuracy grouped bar
        print("\n  +    Superclass accuracy grouped bar …")
        plot_superclass_accuracy(eval_scratch, eval_transfer)

        # 7. Confidence distributions
        print("\n  +    Confidence distribution histograms …")
        plot_confidence_distribution(eval_scratch, eval_transfer)

        # 8. t-SNE (optional — slow)
        if args.tsne:
            print(f"\n  +    t-SNE feature scatter — Scratch CNN …")
            plot_tsne(
                model       = scratch_model,
                loader      = scratch_test,
                n_samples   = args.tsne_samples,
                save_name   = "feature_tsne_scratch.png",
                title       = f"t-SNE — Scratch CNN features ({args.tsne_samples} test samples)",
            )
            print(f"\n  +    t-SNE feature scatter — Transfer model …")
            plot_tsne(
                model       = transfer_model,
                loader      = tl_test,
                n_samples   = args.tsne_samples,
                save_name   = "feature_tsne_transfer.png",
                title       = f"t-SNE — Transfer ({args.backbone}) features ({args.tsne_samples} samples)",
            )

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    wall_total   = time.perf_counter() - wall_start
    gap_top1     = eval_transfer["top1"] - eval_scratch["top1"]
    gap_top5     = eval_transfer["top5"] - eval_scratch["top5"]
    rob_gap_01   = (rob_transfer.get(0.1, 0) - rob_scratch.get(0.1, 0))

    print("\n" + "═" * 62)
    print("  Summary for assignment report")
    print("═" * 62)
    print(f"  Scratch CNN ({args.scratch_variant})")
    print(f"    Top-1 test accuracy : {eval_scratch['top1']:.4f}  "
          f"({eval_scratch['top1']*100:.2f}%)")
    print(f"    Top-5 test accuracy : {eval_scratch['top5']:.4f}  "
          f"({eval_scratch['top5']*100:.2f}%)")
    print(f"    Parameters          : {scratch_params:,}")
    print(f"    Avg epoch time      : {format_time(avg_scratch_ep)}")
    print(f"    Inference ms/image  : {timing_scratch['ms_per_image']:.3f} ms")

    print(f"\n  Transfer — {args.backbone}")
    print(f"    Top-1 test accuracy : {eval_transfer['top1']:.4f}  "
          f"({eval_transfer['top1']*100:.2f}%)")
    print(f"    Top-5 test accuracy : {eval_transfer['top5']:.4f}  "
          f"({eval_transfer['top5']*100:.2f}%)")
    print(f"    Parameters          : {transfer_params:,}")
    print(f"    Avg epoch time      : {format_time(avg_transfer_ep)}")
    print(f"    Inference ms/image  : {timing_transfer['ms_per_image']:.3f} ms")

    print(f"\n  Transfer vs Scratch gains")
    print(f"    Top-1 accuracy gain : {gap_top1:+.4f}  ({gap_top1*100:+.2f} pp)")
    print(f"    Top-5 accuracy gain : {gap_top5:+.4f}  ({gap_top5*100:+.2f} pp)")
    print(f"    Robustness gap σ=0.1: {rob_gap_01:+.4f}  ({rob_gap_01*100:+.2f} pp)")

    print(f"\n  Total compare.py time : {format_time(wall_total)}")

    print("\n  Outputs written to:")
    outputs = [
        "outputs/results/comparison_table.csv",
        "outputs/results/scratch_eval.json",
        "outputs/results/transfer_eval.json",
        "outputs/results/robustness_scratch.json",
        "outputs/results/robustness_transfer.json",
        "outputs/plots/learning_curves.png",
        "outputs/plots/confusion_scratch.png",
        "outputs/plots/confusion_transfer.png",
        "outputs/plots/per_class_scratch.png",
        "outputs/plots/per_class_transfer.png",
        "outputs/plots/robustness.png",
        "outputs/plots/top_failures_scratch.png",
        "outputs/plots/top_failures_transfer.png",
        "outputs/plots/superclass_accuracy.png",
        "outputs/plots/confidence_distribution.png",
    ]
    for o in outputs:
        print(f"    {o}")
    print("═" * 62 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()