"""
src/evaluate.py
───────────────
All evaluation, metric computation, and visualisation for CIFAR-100.

Exports:
    evaluate()                  → top-1 / top-5 accuracy + predictions
    evaluate_per_class()        → per-class breakdown sorted by accuracy
    evaluate_robustness()       → accuracy under Gaussian noise levels
    plot_learning_curves()      → val accuracy + loss over epochs
    plot_confusion_matrix()     → 100×100 heatmap with superclass grid lines
    plot_per_class_accuracy()   → horizontal bar chart, worst/best classes
    plot_robustness()           → accuracy vs noise level for all models
    plot_top_failures()         → grid of most-confidently wrong predictions
    print_comparison_table()    → formatted side-by-side metrics
"""

import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)

from src.config import DEVICE, PLOT_DIR, RESULT_DIR, NUM_CLASSES, EVAL


# ── CIFAR-100 superclass mapping ───────────────────────────────────────────────
# Each of the 20 superclasses contains 5 fine-grained classes.
# Used to draw superclass boundary lines on the confusion matrix.

SUPERCLASSES = [
    ("aquatic mammals",        ["beaver","dolphin","otter","seal","whale"]),
    ("fish",                   ["aquarium_fish","flatfish","ray","shark","trout"]),
    ("flowers",                ["orchid","poppy","rose","sunflower","tulip"]),
    ("food containers",        ["bottle","bowl","can","cup","plate"]),
    ("fruit & vegetables",     ["apple","mushroom","orange","pear","sweet_pepper"]),
    ("household electrical",   ["clock","keyboard","lamp","telephone","television"]),
    ("household furniture",    ["bed","chair","couch","table","wardrobe"]),
    ("insects",                ["bee","beetle","butterfly","caterpillar","cockroach"]),
    ("large carnivores",       ["bear","leopard","lion","tiger","wolf"]),
    ("large man-made outdoor", ["bridge","castle","house","road","skyscraper"]),
    ("large natural outdoor",  ["cloud","forest","mountain","plain","sea"]),
    ("large omnivores",        ["camel","cattle","chimpanzee","elephant","kangaroo"]),
    ("medium mammals",         ["fox","porcupine","possum","raccoon","skunk"]),
    ("non-insect invertebrates",["crab","lobster","snail","spider","worm"]),
    ("people",                 ["baby","boy","girl","man","woman"]),
    ("reptiles",               ["crocodile","dinosaur","lizard","snake","turtle"]),
    ("small mammals",          ["hamster","mouse","rabbit","shrew","squirrel"]),
    ("trees",                  ["maple_tree","oak_tree","palm_tree","pine_tree","willow_tree"]),
    ("vehicles 1",             ["bicycle","bus","motorcycle","pickup_truck","train"]),
    ("vehicles 2",             ["lawn_mower","rocket","streetcar","tank","tractor"]),
]


def _get_class_names() -> list[str]:
    """
    Load CIFAR-100 class names in label-index order without
    re-downloading the dataset. Reads from the meta binary directly.
    Uses torchvision as a thin wrapper.
    """
    try:
        import torchvision
        from src.config import DATA_DIR
        ds = torchvision.datasets.CIFAR100(
            root=DATA_DIR, train=False, download=False,
            transform=torchvision.transforms.ToTensor())
        return ds.classes
    except Exception:
        # Fallback: return numeric strings if dataset not available
        return [str(i) for i in range(NUM_CLASSES)]


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(
    model  : nn.Module,
    loader : torch.utils.data.DataLoader,
    top_k  : int = 5,
) -> dict:
    """
    Run full evaluation on a DataLoader.

    Args:
        model:  Trained model already on DEVICE.
        loader: Test or validation DataLoader (no augmentation).
        top_k:  k for top-k accuracy (default 5).

    Returns:
        dict with keys:
            top1      float  — top-1 accuracy (0–1)
            top5      float  — top-k accuracy (0–1)
            loss      float  — cross-entropy loss (no label smoothing)
            preds     ndarray[int]  — predicted class per sample
            labels    ndarray[int]  — ground-truth class per sample
            probs     ndarray[float] — softmax probability of predicted class
            logits_all ndarray[float] — raw logits [N, 100] for further analysis
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_logits, all_labels = [], []
    running_loss = 0.0
    total        = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=False)
            labels = labels.to(DEVICE, non_blocking=False)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)   # [N, 100]
    all_labels = torch.cat(all_labels, dim=0)   # [N]

    # Top-1
    preds    = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    top1     = (preds == labels_np).mean()

    # Top-k using sklearn (handles edge cases cleanly)
    # Pass the full label universe so sklearn doesn't complain when the
    # evaluation batch happens to miss some classes (common on small/smoke sets).
    probs_np  = F.softmax(all_logits, dim=1).numpy()
    top5      = top_k_accuracy_score(
        labels_np,
        probs_np,
        k=top_k,
        labels=list(range(NUM_CLASSES)),
    )

    # Confidence of the predicted class
    pred_probs = probs_np[np.arange(len(preds)), preds]

    avg_loss = running_loss / total

    return {
        "top1"      : float(top1),
        "top5"      : float(top5),
        "loss"      : float(avg_loss),
        "preds"     : preds,
        "labels"    : labels_np,
        "probs"     : pred_probs,
        "logits_all": all_logits.numpy(),
    }


def evaluate_per_class(
    eval_result : dict,
    save        : bool = True,
) -> dict[str, float]:
    """
    Compute per-class top-1 accuracy from an evaluate() result dict.

    Args:
        eval_result: Output of evaluate().
        save:        Write per-class CSV to outputs/results/.

    Returns:
        OrderedDict {class_name: accuracy} sorted from worst to best.
    """
    class_names = _get_class_names()
    preds       = eval_result["preds"]
    labels      = eval_result["labels"]

    per_class = {}
    for cls_idx, name in enumerate(class_names):
        mask              = labels == cls_idx
        n                 = mask.sum()
        correct           = (preds[mask] == cls_idx).sum()
        per_class[name]   = float(correct / n) if n > 0 else 0.0

    # Sort worst → best
    sorted_pc = dict(sorted(per_class.items(), key=lambda x: x[1]))

    if save:
        import csv
        path = RESULT_DIR / "per_class_accuracy.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "accuracy"])
            for name, acc in sorted_pc.items():
                w.writerow([name, f"{acc:.4f}"])
        print(f"[evaluate] Per-class CSV saved → {path}")

    return sorted_pc


def evaluate_robustness(
    model        : nn.Module,
    loader       : torch.utils.data.DataLoader,
    noise_levels : list[float] | None = None,
) -> dict[float, float]:
    """
    Measure top-1 accuracy under increasing levels of Gaussian noise.

    Noise is added to normalised tensors, so σ=0.1 is relative to the
    normalised pixel range (roughly ±2.5), not raw [0,255] values.

    Args:
        model:        Trained model on DEVICE.
        loader:       Test DataLoader (no augmentation — noise is added here).
        noise_levels: List of σ values to test. Defaults to EVAL["noise_levels"].

    Returns:
        dict {sigma: accuracy}
    """
    if noise_levels is None:
        noise_levels = EVAL["noise_levels"]

    model.eval()
    results = {}

    for sigma in noise_levels:
        correct = 0
        total   = 0

        with torch.no_grad():
            for imgs, labels in loader:
                if sigma > 0.0:
                    noise = torch.randn_like(imgs) * sigma
                    imgs  = imgs + noise              # add in normalised space

                imgs   = imgs.to(DEVICE, non_blocking=False)
                labels = labels.to(DEVICE, non_blocking=False)

                preds   = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += imgs.size(0)

        acc           = correct / total
        results[sigma] = acc
        print(f"  [robustness] σ={sigma:.2f}  →  top-1 accuracy: {acc:.4f}")

    return results


def measure_inference_time(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    n_batches : int | None = None,
) -> dict:
    """
    Measure average inference latency per batch and per image.

    Args:
        model:     Trained model on DEVICE.
        loader:    Any DataLoader (batch size affects the measurement).
        n_batches: How many batches to time. Defaults to EVAL["n_inference_batch"].

    Returns:
        dict with keys:
            ms_per_batch  float
            ms_per_image  float
            batch_size    int
    """
    if n_batches is None:
        n_batches = EVAL["n_inference_batch"]

    def _sync():
        # MPS / CUDA are asynchronous — synchronise before stopping the clock
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        elif DEVICE.type == "cuda":
            torch.cuda.synchronize()

    model.eval()
    times      = []
    batch_size = None

    with torch.no_grad():
        it = iter(loader)

        # Warm-up on one batch (JIT/MPS graph compile) — don't count it.
        # Drawn outside the measurement loop so it doesn't eat into the
        # n_batches budget when the loader is very short (e.g. --quick mode
        # wraps it in a single-batch loader).
        warm_consumed = False
        try:
            warm_imgs, _ = next(it)
            batch_size   = warm_imgs.size(0)
            _ = model(warm_imgs.to(DEVICE, non_blocking=False))
            _sync()
            warm_consumed = True
        except StopIteration:
            pass  # empty loader — handled below

        for i, (imgs, _) in enumerate(it):
            if i >= n_batches:
                break
            batch_size = imgs.size(0)
            imgs       = imgs.to(DEVICE, non_blocking=False)

            t0 = time.perf_counter()
            _  = model(imgs)
            _sync()
            times.append(time.perf_counter() - t0)

        # Fallback: if the loader had only the warm-up batch (e.g. --quick
        # mode), re-time that single batch so we still return a meaningful
        # number instead of dividing by zero.
        if not times and warm_consumed:
            imgs = warm_imgs.to(DEVICE, non_blocking=False)
            t0   = time.perf_counter()
            _    = model(imgs)
            _sync()
            times.append(time.perf_counter() - t0)

    if not times:
        print("  [inference] Warning: loader was empty; returning zeros.")
        return {"ms_per_batch": 0.0, "ms_per_image": 0.0, "batch_size": batch_size}

    ms_per_batch = (sum(times) / len(times)) * 1000.0
    ms_per_image = ms_per_batch / (batch_size or 1)

    print(f"  [inference] Batch size   : {batch_size}")
    print(f"  [inference] ms / batch   : {ms_per_batch:.2f}")
    print(f"  [inference] ms / image   : {ms_per_image:.3f}")
    return {
        "ms_per_batch": ms_per_batch,
        "ms_per_image": ms_per_image,
        "batch_size"  : batch_size,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    histories  : dict[str, dict],
    save_name  : str  = "learning_curves.png",
    show       : bool = True,
) -> None:
    """
    Plot validation accuracy and loss curves for one or more training runs.

    Args:
        histories:  {label: history_dict} where each history_dict has keys
                    val_acc, train_acc, val_loss, train_loss from run_training().
        save_name:  Filename inside outputs/plots/.
        show:       Call plt.show() after saving.
    """
    n_models = len(histories)
    # Colour palette — one per model
    colours  = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5", "#854F0B"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Training history", fontsize=13, y=1.01)

    for i, (name, h) in enumerate(histories.items()):
        c      = colours[i % len(colours)]
        epochs = range(1, len(h["val_acc"]) + 1)

        # Left: accuracy
        axes[0].plot(epochs, h["val_acc"],   color=c, lw=2,   label=f"{name} val")
        axes[0].plot(epochs, h["train_acc"], color=c, lw=1,
                     linestyle="--", alpha=0.55, label=f"{name} train")

        # Right: loss
        axes[1].plot(epochs, h["val_loss"],   color=c, lw=2,   label=f"{name} val")
        axes[1].plot(epochs, h["train_loss"], color=c, lw=1,
                     linestyle="--", alpha=0.55, label=f"{name} train")

    for ax, title, ylabel in zip(
        axes,
        ["Accuracy (solid=val, dashed=train)", "Loss (solid=val, dashed=train)"],
        ["Top-1 accuracy", "Cross-entropy loss"],
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, ncol=max(1, n_models))
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Learning curves → {path}")
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    labels    : np.ndarray,
    preds     : np.ndarray,
    title     : str  = "Confusion matrix",
    save_name : str  = "confusion.png",
    show      : bool = True,
) -> None:
    """
    Plot a 100×100 confusion matrix with superclass boundary lines.

    The diagonal shows correct predictions.
    Off-diagonal blocks along the superclass boundaries reveal
    which categories the model confuses within the same superclass.

    Args:
        labels:    Ground-truth labels, shape [N].
        preds:     Predicted labels, shape [N].
        title:     Plot title.
        save_name: Filename inside outputs/plots/.
        show:      Call plt.show() after saving.
    """
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    # Normalise per true class → shows recall per class
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums,
                          out=np.zeros_like(cm, dtype=float),
                          where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        ax          = ax,
        cmap        = "Blues",
        vmin        = 0.0,
        vmax        = 1.0,
        annot       = False,
        xticklabels = False,
        yticklabels = False,
        cbar_kws    = {"shrink": 0.6, "label": "Recall (per true class)"},
        linewidths  = 0,
    )

    # Draw superclass boundary lines every 5 classes
    for boundary in range(5, NUM_CLASSES, 5):
        ax.axhline(boundary, color="white", lw=0.8, alpha=0.7)
        ax.axvline(boundary, color="white", lw=0.8, alpha=0.7)

    # Superclass labels on the right margin (one per 5-class block)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([i * 5 + 2.5 for i in range(20)])
    ax2.set_yticklabels(
        [s[0] for s in SUPERCLASSES],
        fontsize = 6.5,
        rotation = 0,
    )
    ax2.tick_params(left=False, right=False)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted class", fontsize=9)
    ax.set_ylabel("True class",      fontsize=9)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Confusion matrix → {path}")
    if show:
        plt.show()
    plt.close()


def plot_per_class_accuracy(
    per_class  : dict[str, float],
    n_show     : int  = 20,
    save_name  : str  = "per_class_accuracy.png",
    title      : str  = "Per-class accuracy",
    show       : bool = True,
) -> None:
    """
    Horizontal bar chart showing the n_show worst and n_show best classes.

    Args:
        per_class: Output of evaluate_per_class() — {class_name: accuracy}.
        n_show:    How many classes to show at each end.
        save_name: Filename inside outputs/plots/.
        title:     Plot title.
        show:      Call plt.show() after saving.
    """
    names = list(per_class.keys())
    accs  = list(per_class.values())

    # per_class is already sorted worst→best
    worst_n = list(zip(names[:n_show],          accs[:n_show]))
    best_n  = list(zip(names[-n_show:][::-1],   accs[-n_show:][::-1]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    fig.suptitle(title, fontsize=12)

    for ax, data, subtitle, colour in zip(
        axes,
        [worst_n, best_n],
        [f"Worst {n_show} classes", f"Best {n_show} classes"],
        ["#D85A30", "#1D9E75"],
    ):
        cls_names = [d[0] for d in data]
        cls_accs  = [d[1] for d in data]

        bars = ax.barh(cls_names, cls_accs, color=colour, alpha=0.82, height=0.65)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Top-1 accuracy", fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.5)

        # Value labels at end of each bar
        for bar, acc in zip(bars, cls_accs):
            ax.text(
                min(acc + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{acc:.0%}", va="center", ha="left",
                fontsize=7.5, color="#2C2C2A",
            )

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Per-class accuracy → {path}")
    if show:
        plt.show()
    plt.close()


def plot_robustness(
    robustness_dict : dict[str, dict[float, float]],
    save_name       : str  = "robustness.png",
    show            : bool = True,
) -> None:
    """
    Line chart of accuracy vs Gaussian noise level for each model.

    Args:
        robustness_dict: {model_label: {sigma: accuracy}}
                         — output of evaluate_robustness() for each model.
        save_name:       Filename inside outputs/plots/.
        show:            Call plt.show() after saving.
    """
    colours = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for i, (name, results) in enumerate(robustness_dict.items()):
        sigmas = sorted(results.keys())
        accs   = [results[s] for s in sigmas]
        c      = colours[i % len(colours)]
        m      = markers[i % len(markers)]
        ax.plot(sigmas, accs, color=c, marker=m, lw=2,
                markersize=6, label=name)

        # Shade area under curve lightly
        ax.fill_between(sigmas, accs, alpha=0.08, color=c)

    ax.set_title("Robustness to Gaussian noise", fontsize=12)
    ax.set_xlabel("Noise level (σ)", fontsize=10)
    ax.set_ylabel("Top-1 accuracy",  fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Robustness → {path}")
    if show:
        plt.show()
    plt.close()


def plot_top_failures(
    eval_result  : dict,
    loader       : torch.utils.data.DataLoader,
    n_images     : int  = 25,
    save_name    : str  = "top_failures.png",
    title        : str  = "Most confident wrong predictions",
    show         : bool = True,
) -> None:
    """
    Show the images the model got most confidently wrong.
    These are the highest-confidence mistakes — useful for understanding
    what kinds of patterns fool the model.

    Args:
        eval_result: Output of evaluate().
        loader:      The same DataLoader used in evaluate()
                     (images returned in the same order, no shuffle).
        n_images:    Number of failure images to display.
        save_name:   Filename inside outputs/plots/.
        title:       Plot title.
        show:        Call plt.show() after saving.
    """
    class_names = _get_class_names()
    preds       = eval_result["preds"]
    labels      = eval_result["labels"]
    probs       = eval_result["probs"]

    # Find wrong predictions sorted by descending confidence
    wrong_mask    = preds != labels
    wrong_indices = np.where(wrong_mask)[0]
    sorted_wrong  = wrong_indices[np.argsort(-probs[wrong_mask])]
    top_n_indices = sorted_wrong[:n_images]

    # Collect raw images from the loader (un-normalise for display)
    all_imgs = []
    for imgs, _ in loader:
        all_imgs.append(imgs)
    all_imgs = torch.cat(all_imgs, dim=0)   # [N, 3, H, W]

    # Un-normalise: we don't know which normalisation was used, so
    # just clamp to [0,1] after shifting by min — good enough for display
    def to_display(t: torch.Tensor) -> np.ndarray:
        t = t - t.min()
        t = t / (t.max() + 1e-8)
        return t.permute(1, 2, 0).numpy()

    n_cols = min(5, n_images)
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.4, n_rows * 2.6))
    fig.suptitle(title, fontsize=11, y=1.01)
    axes = np.array(axes).reshape(-1)

    for ax_idx, sample_idx in enumerate(top_n_indices):
        img       = to_display(all_imgs[sample_idx])
        true_name = class_names[labels[sample_idx]]
        pred_name = class_names[preds[sample_idx]]
        conf      = probs[sample_idx]

        axes[ax_idx].imshow(img, interpolation="nearest")
        axes[ax_idx].set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.0%})",
            fontsize=6.5, color="#A32D2D", pad=2,
        )
        axes[ax_idx].axis("off")

    # Hide unused subplots
    for ax in axes[len(top_n_indices):]:
        ax.axis("off")

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Top failures → {path}")
    if show:
        plt.show()
    plt.close()


# ── Comparison table ───────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict]) -> None:
    """
    Print a formatted side-by-side comparison table and save it as CSV.

    Args:
        results: {model_label: metrics_dict}
                 Each metrics_dict should contain keys returned by evaluate()
                 plus optional keys: params, avg_epoch_time, ms_per_image.

    Example:
        print_comparison_table({
            "Scratch CNN"  : {**scratch_eval, "params": 9_200_000,
                              "avg_epoch_time": 57.3, "ms_per_image": 0.21},
            "EfficientNet" : {**tl_eval,      "params": 5_288_548,
                              "avg_epoch_time": 95.1, "ms_per_image": 0.38},
        })
    """
    import csv

    metrics = [
        ("Top-1 accuracy",     "top1",           "{:.2%}"),
        ("Top-5 accuracy",     "top5",            "{:.2%}"),
        ("Test loss",          "loss",            "{:.4f}"),
        ("Parameters",         "params",          "{:,.0f}"),
        ("Avg epoch time (s)", "avg_epoch_time",  "{:.1f}"),
        ("Inference ms/image", "ms_per_image",    "{:.3f}"),
    ]

    model_names = list(results.keys())
    col_w       = max(22, max(len(n) for n in model_names) + 2)
    header_w    = 28

    # Header row
    header = f"{'Metric':<{header_w}}" + "".join(
        f"{n:>{col_w}}" for n in model_names)
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))

    rows_csv = [["Metric"] + model_names]

    for display_name, key, fmt in metrics:
        row_str  = f"{display_name:<{header_w}}"
        csv_row  = [display_name]
        for name in model_names:
            val = results[name].get(key)
            if val is None:
                cell = "—"
            else:
                try:
                    cell = fmt.format(val)
                except (ValueError, TypeError):
                    cell = str(val)
            row_str += f"{cell:>{col_w}}"
            csv_row.append(cell)
        print(row_str)
        rows_csv.append(csv_row)

    print("═" * len(header) + "\n")

    # Save CSV
    path = RESULT_DIR / "comparison_table.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows_csv)
    print(f"[evaluate] Comparison table saved → {path}")


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test: runs every function with random data to verify
    nothing crashes before you run compare.py on real models.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    print("[evaluate] Smoke test — random logits, no real model needed\n")

    N          = 320
    fake_imgs  = torch.randn(N, 3, 32, 32)
    fake_labels_t = torch.randint(0, NUM_CLASSES, (N,))
    fake_ds    = TensorDataset(fake_imgs, fake_labels_t)
    fake_loader = DataLoader(fake_ds, batch_size=64, shuffle=False)

    # Tiny random model
    tiny = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, NUM_CLASSES),
    ).to(DEVICE)

    print("── evaluate() ───────────────────────────────────────────────")
    result = evaluate(tiny, fake_loader)
    print(f"  top1 : {result['top1']:.4f}")
    print(f"  top5 : {result['top5']:.4f}")
    print(f"  loss : {result['loss']:.4f}")

    print("\n── evaluate_per_class() ─────────────────────────────────────")
    pc = evaluate_per_class(result, save=False)
    best_5  = list(pc.items())[-5:]
    worst_5 = list(pc.items())[:5]
    print(f"  Best 5  : {best_5}")
    print(f"  Worst 5 : {worst_5}")

    print("\n── evaluate_robustness() ────────────────────────────────────")
    rob = evaluate_robustness(tiny, fake_loader, noise_levels=[0.0, 0.1, 0.3])

    print("\n── measure_inference_time() ─────────────────────────────────")
    t = measure_inference_time(tiny, fake_loader, n_batches=5)

    print("\n── plot_learning_curves() ───────────────────────────────────")
    fake_hist = {
        "train_loss": [4.5, 4.0, 3.5], "val_loss"  : [4.6, 4.1, 3.7],
        "train_acc" : [0.05, 0.10, 0.18], "val_acc" : [0.04, 0.09, 0.16],
    }
    plot_learning_curves(
        {"Random model": fake_hist}, save_name="smoke_curves.png", show=False)

    print("\n── plot_confusion_matrix() ──────────────────────────────────")
    plot_confusion_matrix(
        result["labels"], result["preds"],
        title="Smoke test — confusion matrix",
        save_name="smoke_confusion.png", show=False)

    print("\n── plot_per_class_accuracy() ────────────────────────────────")
    plot_per_class_accuracy(pc, n_show=10,
                             save_name="smoke_per_class.png", show=False)

    print("\n── plot_robustness() ────────────────────────────────────────")
    plot_robustness({"Random model": rob},
                    save_name="smoke_robustness.png", show=False)

    print("\n── plot_top_failures() ──────────────────────────────────────")
    plot_top_failures(result, fake_loader, n_images=10,
                       save_name="smoke_failures.png", show=False)

    print("\n── print_comparison_table() ─────────────────────────────────")
    print_comparison_table({
        "Random model": {
            **result,
            "params"         : 3_145_700,
            "avg_epoch_time" : 12.3,
            "ms_per_image"   : t["ms_per_image"],
        }
    })

    print("[evaluate] All smoke tests passed.")