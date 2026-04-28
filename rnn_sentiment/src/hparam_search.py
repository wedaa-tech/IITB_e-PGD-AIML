"""
Hyperparameter sweep over L (num_layers) and h (hidden_dim) for:
  - VanillaRNN
  - LSTMClassifier

Produces:
  - Heatmap of val accuracy  (L × h grid)
  - Gradient norm plot       (per layer, per epoch)
  - Best config summary

All results cached to outputs/logs/hparam_results.pkl
"""

import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from itertools import product
from pathlib import Path

import torch

from src.config      import LOG_DIR, CHECKPOINT_DIR, VOCAB_SIZE, NUM_EPOCHS as CFG_NUM_EPOCHS
from src.utils       import set_seed
from src.models      import MODEL_REGISTRY
from src.trainer     import train_with_grad_tracking, evaluate_test
from src.embeddings  import get_embedding, REAL_VOCAB_SIZE

log = logging.getLogger(__name__)

# ── Search grid ───────────────────────────────────────────────────────────────
L_VALUES = [1, 2, 3]          # number of hidden layers
H_VALUES = [128, 256, 512]    # hidden dimension

# ── Fixed across all runs ────────────────────────────────────────────────────
EMBED_STRATEGY = "glove"       # best embedding from previous experiment
EMBED_DIM      = 100
DROPOUT        = 0.5
# Share the epoch budget with main.py / experiment.py so that a single
# RNN_EPOCHS env var (set by run_all.sh --epochs N) controls every training
# script consistently. Subtract 2 by default so the sweep stays cheaper than
# the headline runs, but respect any explicit override from the environment.
NUM_EPOCHS     = max(1, CFG_NUM_EPOCHS - 2) if CFG_NUM_EPOCHS >= 3 else CFG_NUM_EPOCHS
PATIENCE       = 3
BASE_KWARGS    = dict(
    vocab_size = REAL_VOCAB_SIZE,
    embed_dim  = EMBED_DIM,
    dropout    = DROPOUT,
    pad_idx    = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_hparam_sweep(
    train_loader, val_loader, test_loader,
    vocab,
    models_to_sweep: list[str] = ["rnn", "lstm"],
    force: bool = False,
) -> dict:
    """
    For each (model, L, h) combination:
      1. Build model with GloVe embedding
      2. Train with gradient norm tracking
      3. Evaluate on test set
      4. Store history, grad norms, test results

    Returns nested dict:
      results[model_name][L][h] = {
          history, grad_history, test_loss, test_acc
      }
    """
    cache = LOG_DIR / "hparam_results.pkl"
    if cache.exists() and not force:
        log.info(f"Loading cached hparam results from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Build GloVe embedding once — deep-copied per run
    import copy
    log.info("Building GloVe embedding for hparam sweep …")
    base_emb, _ = get_embedding(
        strategy  = EMBED_STRATEGY,
        vocab     = vocab,
        trainable = True,
    )

    results = {m: {} for m in models_to_sweep}
    total   = len(models_to_sweep) * len(L_VALUES) * len(H_VALUES)
    run_idx = 0

    for model_name in models_to_sweep:
        results[model_name] = {L: {} for L in L_VALUES}

        for L, h in product(L_VALUES, H_VALUES):
            run_idx += 1
            run_key = f"{model_name}_L{L}_H{h}"
            log.info(f"\n[{run_idx}/{total}]  {run_key}")

            emb_copy = copy.deepcopy(base_emb)

            model = MODEL_REGISTRY[model_name](
                **BASE_KWARGS,
                hidden_dim = h,
                num_layers = L,
                embedding  = emb_copy,
            )

            history, grad_history = train_with_grad_tracking(
                model_name   = run_key,
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                num_epochs   = NUM_EPOCHS,
                patience     = PATIENCE,
            )

            test_res = evaluate_test(run_key, model, test_loader)

            results[model_name][L][h] = {
                "history":      history,
                "grad_history": grad_history,
                "test_loss":    test_res["loss"],
                "test_acc":     test_res["accuracy"],
                "params":       sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }

    with open(cache, "wb") as f:
        pickle.dump(results, f)
    log.info(f"\nHparam results cached → {cache}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_hparam_heatmaps(results: dict):
    """
    Two side-by-side heatmaps (RNN vs LSTM) showing test accuracy
    across the L × h grid. Best cell highlighted.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    fig.suptitle("Hyperparameter Sweep — Test Accuracy (L layers × h hidden dim)",
                 fontsize=13, fontweight="bold")

    for ax, (model_name, res) in zip(axes, results.items()):
        # Build matrix: rows = L, cols = h
        matrix = np.array([
            [res[L][h]["test_acc"] * 100 for h in H_VALUES]
            for L in L_VALUES
        ])

        sns.heatmap(
            matrix,
            ax           = ax,
            annot        = True,
            fmt          = ".2f",
            cmap         = "YlGnBu",
            xticklabels  = [f"h={h}" for h in H_VALUES],
            yticklabels  = [f"L={L}" for L in L_VALUES],
            vmin         = matrix.min() - 1,
            vmax         = 100,
            linewidths   = 0.5,
            cbar_kws     = {"label": "Test Accuracy (%)"},
        )

        # Star the best cell
        best_idx = np.unravel_index(matrix.argmax(), matrix.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1], best_idx[0]), 1, 1,
            fill=False, edgecolor="crimson", lw=2.5, label="Best"
        ))
        ax.set_title(f"{model_name.upper()}", fontsize=12)
        ax.set_xlabel("Hidden dimension (h)")
        ax.set_ylabel("Number of layers (L)")

    plt.tight_layout()
    out = LOG_DIR / "hparam_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


def plot_gradient_norms(results: dict, best_configs: dict):
    """
    For the best (L, h) config of each model, plot gradient norms
    across layers and epochs to demonstrate vanishing gradients.

    best_configs = { "rnn": (L, h), "lstm": (L, h) }
    """
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    MODEL_COLORS = {
        "rnn":  "#534AB7",
        "lstm": "#1D9E75",
    }
    LAYER_STYLES = ["-", "--", ":"]

    fig.suptitle("Gradient Norms per Layer — RNN vs LSTM\n"
                 "(lower = more vanishing; best L×h config shown)",
                 fontsize=12, fontweight="bold")

    for ax, (model_name, res) in zip(axes, results.items()):
        L, h      = best_configs[model_name]
        grad_hist = res[L][h]["grad_history"]
        color     = MODEL_COLORS.get(model_name, "#888")

        # Group parameters by RNN layer index
        layer_norms = {}   # layer_idx → list of norms per epoch
        for param_name, norms in grad_hist.items():
            # RNN weight names contain 'l0', 'l1', 'l2' etc.
            for li, layer_idx in enumerate(range(L)):
                tag = f"l{layer_idx}_"
                if tag in param_name and "weight_hh" in param_name:
                    key = f"Layer {layer_idx + 1}"
                    if key not in layer_norms:
                        layer_norms[key] = norms
                    break

        if not layer_norms:
            # Fallback: plot embedding and fc grad norms
            for i, (pname, norms) in enumerate(
                list(grad_hist.items())[:3]
            ):
                short = pname.split(".")[-1]
                ax.plot(norms, label=short,
                        color=color, linestyle=LAYER_STYLES[i % 3],
                        linewidth=1.8)
        else:
            for li, (layer_key, norms) in enumerate(
                sorted(layer_norms.items())
            ):
                ax.plot(norms, label=layer_key, color=color,
                        linestyle=LAYER_STYLES[li % 3], linewidth=1.8,
                        alpha=1.0 - li * 0.2)

        ax.set_title(f"{model_name.upper()}  (L={L}, h={h})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm (L2)")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter("%.4f")
        )

    plt.tight_layout()
    out = LOG_DIR / "gradient_norms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


def plot_best_training_curves(results: dict, best_configs: dict):
    """
    Training + validation loss curves for best config of each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Training Curves — Best Hyperparameter Config per Model",
                 fontsize=12, fontweight="bold")

    COLORS = {"rnn": "#534AB7", "lstm": "#1D9E75"}

    for model_name, res in results.items():
        L, h    = best_configs[model_name]
        hist    = res[L][h]["history"]
        color   = COLORS.get(model_name, "#888")
        label   = f"{model_name.upper()} (L={L}, h={h})"
        epochs  = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], "--",
                     color=color, linewidth=1.4, alpha=0.6,
                     label=f"{label} train")
        axes[0].plot(epochs, hist["val_loss"], "-",
                     color=color, linewidth=2.2,
                     label=f"{label} val")

        axes[1].plot(epochs,
                     [a * 100 for a in hist["train_acc"]], "--",
                     color=color, linewidth=1.4, alpha=0.6,
                     label=f"{label} train")
        axes[1].plot(epochs,
                     [a * 100 for a in hist["val_acc"]], "-",
                     color=color, linewidth=2.2,
                     label=f"{label} val")

    for ax, title, ylabel in [
        (axes[0], "Loss",     "BCE Loss"),
        (axes[1], "Accuracy", "Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    axes[1].yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.0f%%")
    )

    plt.tight_layout()
    out = LOG_DIR / "best_config_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info(f"Saved → {out}")


def print_hparam_summary(results: dict) -> dict:
    """Print full grid and return best configs."""
    best_configs = {}

    for model_name, res in results.items():
        print(f"\n{'='*58}")
        print(f"  {model_name.upper()} — Test Accuracy Grid")
        print(f"  {'':6} " +
              "  ".join(f"h={h:>4}" for h in H_VALUES))
        print(f"  {'─'*50}")

        best_acc = 0
        best_cfg = (1, 128)

        for L in L_VALUES:
            row = f"  L={L}  "
            for h in H_VALUES:
                r   = res[L][h]
                acc = r["test_acc"] * 100
                row += f"  {acc:>6.2f}%"
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (L, h)
            print(row)

        best_configs[model_name] = best_cfg
        L, h = best_cfg
        r    = res[L][h]
        print(f"\n  Best config : L={L}, h={h}")
        print(f"  Test acc    : {r['test_acc']:.4f}")
        print(f"  Test loss   : {r['test_loss']:.4f}")
        print(f"  Params      : {r['params']:,}")
        print(f"{'='*58}")

    return best_configs