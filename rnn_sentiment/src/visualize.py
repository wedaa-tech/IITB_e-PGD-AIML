"""
All plots for the RNN sentiment assignment.
Saves every figure to outputs/logs/ as high-res PNG.
"""

import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc
)

from src.config import LOG_DIR, CHECKPOINT_DIR

log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

MODEL_COLORS = {
    "rnn":       "#534AB7",   # purple
    "lstm":      "#1D9E75",   # teal
    "attention": "#D85A30",   # coral
}
MODEL_LABELS = {
    "rnn":       "Vanilla RNN",
    "lstm":      "LSTM",
    "attention": "Attention LSTM",
}

# Friendly labels for the pretrained-embedding runs produced by experiment.py
# so plots and classification reports aren't keyed on raw strings.
EMBEDDING_LABELS = {
    "word2vec": "Word2Vec",
    "glove":    "GloVe",
    "onehot":   "One-Hot",
    "learned":  "learned",
}


def _label(name: str) -> str:
    """
    Human-readable label for a run key.

    Accepts:
      - bare model names like "rnn", "lstm", "attention"
      - composite keys from experiment.py like "rnn_word2vec",
        "lstm_glove", "attention_word2vec"
      - sweep keys like "rnn_L2_H256" (returned as-is, no special mapping)

    Falls back to the raw key for anything unrecognised, which is always
    safe for printing.
    """
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]

    base, _, tail = name.partition("_")
    if base in MODEL_LABELS and tail in EMBEDDING_LABELS:
        return f"{MODEL_LABELS[base]} ({EMBEDDING_LABELS[tail]})"

    return name


def _color(name: str) -> str:
    """Colour for a run key — shared by composite and bare names."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    base = name.split("_", 1)[0]
    return MODEL_COLORS.get(base, "#888780")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves  (loss + accuracy side by side)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict[str, dict]):
    """
    histories = { "rnn": {train_loss, val_loss, train_acc, val_acc}, ... }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — All Models", fontsize=14, fontweight="bold", y=1.02)

    for name, hist in histories.items():
        color  = _color(name)
        label  = _label(name)
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, hist["train_loss"], color=color,
                     linestyle="--", linewidth=1.4, alpha=0.7, label=f"{label} train")
        axes[0].plot(epochs, hist["val_loss"],   color=color,
                     linestyle="-",  linewidth=2.0,             label=f"{label} val")

        # Accuracy
        axes[1].plot(epochs, [a * 100 for a in hist["train_acc"]], color=color,
                     linestyle="--", linewidth=1.4, alpha=0.7, label=f"{label} train")
        axes[1].plot(epochs, [a * 100 for a in hist["val_acc"]],   color=color,
                     linestyle="-",  linewidth=2.0,             label=f"{label} val")

    axes[0].set_title("Loss",     fontsize=12)
    axes[1].set_title("Accuracy", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Shared legend below both axes
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout()
    out = LOG_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-model training curve (detailed, with val-loss star marker)
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_curve(name: str, hist: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{_label(name)} — Training Detail",
                 fontsize=13, fontweight="bold")
    color  = _color(name)
    epochs = range(1, len(hist["train_loss"]) + 1)

    best_ep = int(np.argmin(hist["val_loss"])) + 1

    for ax, tr_key, vl_key, ylabel, fmt in [
        (axes[0], "train_loss", "val_loss",  "BCE Loss",     lambda x: x),
        (axes[1], "train_acc",  "val_acc",   "Accuracy (%)", lambda x: x * 100),
    ]:
        tr = [fmt(v) for v in hist[tr_key]]
        vl = [fmt(v) for v in hist[vl_key]]
        ax.plot(epochs, tr, "--", color=color, alpha=0.6, linewidth=1.4, label="Train")
        ax.plot(epochs, vl, "-",  color=color, linewidth=2.2,             label="Val")
        ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Best epoch ({best_ep})")
        ax.scatter([best_ep], [vl[best_ep - 1]],
                   color=color, s=80, zorder=5)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        if ylabel == "Accuracy (%)":
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = LOG_DIR / f"{name}_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion matrices (side by side)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(test_results: dict[str, dict]):
    n   = len(test_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip(axes, test_results.items()):
        cm  = confusion_matrix(res["labels"], res["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        acc  = res["accuracy"]
        ax.set_title(f"{_label(name)}\nTest acc: {acc:.2%}", fontsize=11)
        ax.grid(False)

    plt.tight_layout()
    out = LOG_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bar chart — final test comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(test_results: dict[str, dict]):
    names    = list(test_results.keys())
    accs     = [test_results[n]["accuracy"] * 100 for n in names]
    losses   = [test_results[n]["loss"]           for n in names]
    colors   = [_color(n)                         for n in names]
    x        = np.arange(len(names))
    width    = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Test Set Comparison", fontsize=13, fontweight="bold")

    # Accuracy bars
    bars = axes[0].bar(x, accs, color=colors, width=width, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_label(n) for n in names])
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(60, 100)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3, f"{acc:.2f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Loss bars
    bars2 = axes[1].bar(x, losses, color=colors, width=width, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_label(n) for n in names])
    axes[1].set_ylabel("Test BCE Loss")
    axes[1].set_ylim(0, max(losses) * 1.25)
    for bar, loss in zip(bars2, losses):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005, f"{loss:.4f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = LOG_DIR / "comparison_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Classification report (printed + saved as text)
# ─────────────────────────────────────────────────────────────────────────────

def print_classification_reports(test_results: dict[str, dict]):
    out_lines = []
    for name, res in test_results.items():
        header = f"\n{'='*50}\n  {_label(name)}  —  Classification Report\n{'='*50}"
        report = classification_report(
            res["labels"], res["preds"],
            target_names=["Negative", "Positive"],
            digits=4,
        )
        print(header)
        print(report)
        out_lines.append(header + "\n" + report)

    report_path = LOG_DIR / "classification_reports.txt"
    report_path.write_text("\n".join(out_lines))
    log.info(f"Reports saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load cached history from disk
# ─────────────────────────────────────────────────────────────────────────────

def load_history(name: str) -> dict:
    path = CHECKPOINT_DIR / f"{name}_history.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)



def plot_embedding_comparison(test_results: dict):
    """
    Grouped bar chart: x-axis = model, groups = embedding strategy.
    Specifically designed for B4 answer.
    Expects keys like 'rnn_word2vec', 'lstm_glove' etc.
    """
    import matplotlib.patches as mpatches

    models     = ["rnn", "lstm", "attention"]
    embeddings = ["word2vec", "glove"]
    emb_colors = {"word2vec": "#1D9E75", "glove": "#534AB7"}
    emb_labels = {"word2vec": "Word2Vec", "glove": "GloVe"}

    x      = np.arange(len(models))
    width  = 0.30
    offset = {"word2vec": -0.15, "glove": 0.15}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Embedding Strategy Comparison Across Models",
                 fontsize=13, fontweight="bold")

    for emb in embeddings:
        accs   = [test_results[f"{m}_{emb}"]["accuracy"] * 100
                  for m in models]
        losses = [test_results[f"{m}_{emb}"]["loss"]
                  for m in models]
        color  = emb_colors[emb]
        label  = emb_labels[emb]
        off    = offset[emb]

        bars_a = axes[0].bar(x + off, accs, width,
                             color=color, label=label,
                             edgecolor="white", linewidth=0.8)
        bars_l = axes[1].bar(x + off, losses, width,
                             color=color, label=label,
                             edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars_a, accs):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.2,
                         f"{val:.1f}%", ha="center",
                         va="bottom", fontsize=9)
        for bar, val in zip(bars_l, losses):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center",
                         va="bottom", fontsize=9)

    model_labels = ["Vanilla RNN", "LSTM", "Attention LSTM"]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_ylim(60, 100)
    axes[0].yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.0f%%")
    )
    axes[0].set_title("Test Accuracy")

    axes[1].set_ylabel("Test BCE Loss")
    axes[1].set_ylim(0, max(
        test_results[f"{m}_{e}"]["loss"]
        for m in models for e in embeddings
    ) * 1.3)
    axes[1].set_title("Test Loss")

    plt.tight_layout()
    out = LOG_DIR / "embedding_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")