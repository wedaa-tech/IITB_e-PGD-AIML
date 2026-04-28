"""
B5 analysis — RNN vs LSTM direct comparison.
Uses results already saved from main.py and experiment.py runs.
No retraining needed.
"""
import copy
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import classification_report

from src.utils      import set_seed, get_device_info
from src.preprocess import run_pipeline
from src.dataset    import get_dataloaders
from src.models     import MODEL_REGISTRY
from src.trainer    import evaluate_test
from src.config     import LOG_DIR, CHECKPOINT_DIR
from src.embeddings import REAL_VOCAB_SIZE, get_embedding
from src.visualize  import load_history

log = logging.getLogger(__name__)


def _load_tokenized_train() -> list[list[str]] | None:
    """
    Returns the cached tokenised training corpus that experiment.py uses
    to fit Word2Vec. None if the cache isn't on disk — GloVe still works.
    """
    cache = LOG_DIR.parent.parent / "data/processed/tokenized_train.pkl"
    if not cache.exists():
        log.warning(f"Tokenised training corpus not found at {cache}. "
                    f"Re-run experiment.py first if Word2Vec is needed.")
        return None
    with open(cache, "rb") as f:
        return pickle.load(f)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

COLORS = {
    "rnn_learned":    "#B4B2A9",   # gray   — baseline RNN
    "lstm_learned":   "#1D9E75",   # teal   — LSTM
    "rnn_word2vec":   "#AFA9EC",   # light purple
    "lstm_word2vec":  "#534AB7",   # purple
    "rnn_glove":      "#F0997B",   # light coral
    "lstm_glove":     "#D85A30",   # coral
}

MODEL_KWARGS = dict(
    vocab_size = REAL_VOCAB_SIZE,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Side-by-side training curves RNN vs LSTM (learned embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rnn_vs_lstm_curves(rnn_hist: dict, lstm_hist: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "RNN vs LSTM — Training Curves (L=2, h=256, learned embeddings)",
        fontsize=13, fontweight="bold"
    )

    pairs = [
        ("rnn",  rnn_hist,  "Vanilla RNN"),
        ("lstm", lstm_hist, "LSTM"),
    ]
    styles = {"rnn": ("#888780", "--"), "lstm": ("#1D9E75", "-")}

    for key, hist, label in pairs:
        color, ls = styles[key]
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], linestyle=":",
                     color=color, linewidth=1.4, alpha=0.7,
                     label=f"{label} train")
        axes[0].plot(epochs, hist["val_loss"], linestyle=ls,
                     color=color, linewidth=2.2,
                     label=f"{label} val")

        axes[1].plot(epochs, [a*100 for a in hist["train_acc"]],
                     linestyle=":", color=color, linewidth=1.4, alpha=0.7,
                     label=f"{label} train")
        axes[1].plot(epochs, [a*100 for a in hist["val_acc"]],
                     linestyle=ls, color=color, linewidth=2.2,
                     label=f"{label} val")

    for ax, title, ylabel in [
        (axes[0], "Loss (BCE)",     "Loss"),
        (axes[1], "Accuracy",       "Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    out = LOG_DIR / "b5_rnn_vs_lstm_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Convergence speed: val accuracy per epoch overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_speed(rnn_hist: dict, lstm_hist: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Convergence Speed — Validation Accuracy per Epoch",
        fontsize=13, fontweight="bold"
    )

    for hist, label, color, ls in [
        (rnn_hist,  "Vanilla RNN", "#888780", "--"),
        (lstm_hist, "LSTM",        "#1D9E75", "-"),
    ]:
        epochs = range(1, len(hist["val_acc"]) + 1)
        accs   = [a * 100 for a in hist["val_acc"]]
        ax.plot(epochs, accs, linestyle=ls, color=color,
                linewidth=2.4, marker="o", markersize=5, label=label)

        # Mark best epoch
        best_ep  = int(np.argmax(hist["val_acc"])) + 1
        best_acc = max(accs)
        ax.annotate(
            f"Best: {best_acc:.1f}%\n(ep {best_ep})",
            xy=(best_ep, best_acc),
            xytext=(best_ep + 0.3, best_acc - 3),
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.axhline(75, color="gray", linewidth=0.8, linestyle=":",
               alpha=0.6, label="75% reference")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = LOG_DIR / "b5_convergence_speed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Full comparison bar: all embedding × model combos
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_comparison(test_results: dict):
    """
    Groups: learned / word2vec / glove
    Bars: RNN (gray shades) vs LSTM (teal shades)
    """
    groups  = ["learned", "word2vec", "glove"]
    g_labels= ["Learned\n(baseline)", "Word2Vec", "GloVe"]
    models  = ["rnn", "lstm"]
    m_colors= {"rnn": "#888780", "lstm": "#1D9E75"}
    m_labels= {"rnn": "Vanilla RNN", "lstm": "LSTM"}

    x     = np.arange(len(groups))
    width = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "RNN vs LSTM — All Embedding Strategies (B5 Summary)",
        fontsize=13, fontweight="bold"
    )

    offsets = {"rnn": -0.15, "lstm": 0.15}

    for m in models:
        accs   = [test_results[f"{m}_{g}"]["accuracy"] * 100 for g in groups]
        losses = [test_results[f"{m}_{g}"]["loss"]           for g in groups]
        color  = m_colors[m]
        off    = offsets[m]

        bars_a = axes[0].bar(x + off, accs,   width, color=color,
                             label=m_labels[m], edgecolor="white")
        bars_l = axes[1].bar(x + off, losses, width, color=color,
                             label=m_labels[m], edgecolor="white")

        for bar, v in zip(bars_a, accs):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.2,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        for bar, v in zip(bars_l, losses):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.003,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(g_labels)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_ylim(60, 100)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[0].set_title("Test Accuracy by Embedding")

    axes[1].set_ylabel("Test BCE Loss")
    axes[1].set_title("Test Loss by Embedding")

    plt.tight_layout()
    out = LOG_DIR / "b5_full_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Training stability: val loss smoothness (std dev rolling window)
# ─────────────────────────────────────────────────────────────────────────────

def plot_stability(rnn_hist: dict, lstm_hist: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Training Stability — Validation Loss Trajectory",
        fontsize=13, fontweight="bold"
    )

    for hist, label, color, ls in [
        (rnn_hist,  "Vanilla RNN", "#888780", "--"),
        (lstm_hist, "LSTM",        "#1D9E75", "-"),
    ]:
        epochs   = list(range(1, len(hist["val_loss"]) + 1))
        val_loss = hist["val_loss"]
        ax.plot(epochs, val_loss, linestyle=ls, color=color,
                linewidth=2.2, marker="o", markersize=5, label=label)

        # Shade ±std of val loss to show volatility
        mean_vl = np.mean(val_loss)
        std_vl  = np.std(val_loss)
        ax.axhspan(mean_vl - std_vl, mean_vl + std_vl,
                   alpha=0.08, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (BCE)")
    ax.legend(fontsize=10)

    # Annotate RNN spike
    rnn_vl = rnn_hist["val_loss"]
    spike_ep  = int(np.argmax(rnn_vl)) + 1
    spike_val = max(rnn_vl)
    ax.annotate(
        f"RNN spike\n(ep {spike_ep}: {spike_val:.3f})",
        xy=(spike_ep, spike_val),
        xytext=(spike_ep - 2, spike_val + 0.02),
        fontsize=9, color="#888780",
        arrowprops=dict(arrowstyle="->", color="#888780", lw=1.2),
    )

    plt.tight_layout()
    out = LOG_DIR / "b5_stability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    get_device_info()
    set_seed()

    data = run_pipeline(force=False)
    _, _, test_loader = get_dataloaders(data)
    vocab             = data["vocab"]

    # ── Load histories ────────────────────────────────────────────────────
    rnn_hist  = load_history("rnn")
    lstm_hist = load_history("lstm")

    # ── Rebuild pretrained embeddings to match what experiment.py saved ──
    # The *_word2vec / *_glove checkpoints were trained with 100-dim pretrained
    # embeddings injected via `embedding=`. Rebuilding the model without that
    # argument falls back to `embed_dim=128`, which breaks load_state_dict.
    tokenized_train = _load_tokenized_train()
    emb_layers = {}
    for emb in ("word2vec", "glove"):
        log.info(f"Rebuilding embedding layer: {emb}")
        emb_layer, _ = get_embedding(
            strategy        = emb,
            vocab           = vocab,
            tokenized_train = tokenized_train if emb == "word2vec" else None,
            trainable       = True,
        )
        emb_layers[emb] = emb_layer

    def _build(model_name: str, emb_name: str | None):
        """
        Construct a model that matches the checkpoint architecture.
          emb_name=None → learned embedding runs (main.py, 128-dim default)
          emb_name="word2vec" / "glove" → pretrained runs (experiment.py, 100-dim)
        """
        if emb_name is None:
            return MODEL_REGISTRY[model_name](**MODEL_KWARGS)
        return MODEL_REGISTRY[model_name](
            **MODEL_KWARGS,
            embedding = copy.deepcopy(emb_layers[emb_name]),
        )

    # ── Re-evaluate from saved checkpoints ────────────────────────────────
    test_results = {
        # Learned-embedding runs (from main.py — 128-dim learned embeddings)
        "rnn_learned":   evaluate_test("rnn",  _build("rnn",  None), test_loader),
        "lstm_learned":  evaluate_test("lstm", _build("lstm", None), test_loader),
        # Pretrained-embedding runs (from experiment.py — 100-dim W2V / GloVe)
        "rnn_word2vec":  evaluate_test("rnn_word2vec",
                            _build("rnn",  "word2vec"), test_loader),
        "rnn_glove":     evaluate_test("rnn_glove",
                            _build("rnn",  "glove"),    test_loader),
        "lstm_word2vec": evaluate_test("lstm_word2vec",
                            _build("lstm", "word2vec"), test_loader),
        "lstm_glove":    evaluate_test("lstm_glove",
                            _build("lstm", "glove"),    test_loader),
    }

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Config':<22} {'Test Acc':>10} {'Test Loss':>11}")
    print("  " + "─" * 48)
    labels = {
        "rnn_learned":   "RNN  (learned emb)",
        "lstm_learned":  "LSTM (learned emb)",
        "rnn_word2vec":  "RNN  (Word2Vec)",
        "lstm_word2vec": "LSTM (Word2Vec)",
        "rnn_glove":     "RNN  (GloVe)",
        "lstm_glove":    "LSTM (GloVe)",
    }
    for key, label in labels.items():
        r = test_results[key]
        print(f"  {label:<22} {r['accuracy']:>9.2%}  {r['loss']:>10.4f}")
    print("=" * 60)

    # ── Classification reports ────────────────────────────────────────────
    for key in ["rnn_learned", "lstm_learned"]:
        r = test_results[key]
        print(f"\n{labels[key]} — Classification Report")
        print(classification_report(
            r["labels"], r["preds"],
            target_names=["Negative", "Positive"], digits=4
        ))

    # ── All plots ─────────────────────────────────────────────────────────
    plot_rnn_vs_lstm_curves(rnn_hist, lstm_hist)
    plot_convergence_speed(rnn_hist, lstm_hist)
    plot_stability(rnn_hist, lstm_hist)
    plot_full_comparison(test_results)

    print("\nDone. B5 outputs saved to outputs/logs/")

if __name__ == "__main__":
    main()