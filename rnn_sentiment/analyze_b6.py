"""
B6 analysis — Attention mechanism study.
Covers:
  1. Full 3-model comparison (RNN / LSTM / Attention LSTM)
  2. Attention weight heatmap over real IMDB reviews
  3. Computational overhead table (params, epoch time, memory)
  4. Attention distribution analysis
No retraining needed — loads from saved checkpoints.
"""

import logging
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.utils      import set_seed, get_device_info
from src.preprocess import run_pipeline, preprocess_text, clean_text, tokenize
from src.dataset    import get_dataloaders
from src.models     import MODEL_REGISTRY
from src.models.attention_lstm import AttentionLSTM
from src.trainer    import evaluate_test
from src.config     import LOG_DIR, CHECKPOINT_DIR, DEVICE, MAX_SEQ_LEN
from src.embeddings import REAL_VOCAB_SIZE
from src.visualize  import load_history, plot_training_curves
from src.preprocess import Vocabulary, pad_sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True,  "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

MODEL_KWARGS = dict(
    vocab_size = REAL_VOCAB_SIZE,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)

MODELS = ["rnn", "lstm", "attention"]
MODEL_LABELS = {
    "rnn":       "Vanilla RNN",
    "lstm":      "LSTM",
    "attention": "Attention LSTM",
}
COLORS = {
    "rnn":       "#888780",
    "lstm":      "#1D9E75",
    "attention": "#D85A30",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Three-model comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_model_comparison(test_results: dict):
    """
    Bar chart: RNN vs LSTM vs Attention LSTM on accuracy and loss.
    """
    names  = list(test_results.keys())
    accs   = [test_results[n]["accuracy"] * 100 for n in names]
    losses = [test_results[n]["loss"]           for n in names]
    colors = [COLORS[n]                         for n in names]
    labels = [MODEL_LABELS[n]                   for n in names]
    x      = np.arange(len(names))
    width  = 0.45

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Three-Model Comparison — RNN / LSTM / Attention LSTM\n"
        "(L=2, h=256, learned embeddings)",
        fontsize=13, fontweight="bold"
    )

    for ax, vals, ylabel, ylim, fmt in [
        (axes[0], accs,   "Test Accuracy (%)",  (60, 100), "{:.2f}%"),
        (axes[1], losses, "Test BCE Loss",       (0, 0.7),  "{:.4f}"),
    ]:
        bars = ax.bar(x, vals, width, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if ylabel.startswith("Test Acc") else 0.005),
                fmt.format(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    out = LOG_DIR / "b6_three_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training curves — all three models overlaid
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_three_curves(histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Training Curves — RNN vs LSTM vs Attention LSTM",
        fontsize=13, fontweight="bold"
    )

    for name, hist in histories.items():
        color  = COLORS[name]
        label  = MODEL_LABELS[name]
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["val_loss"], "-",
                     color=color, linewidth=2.2, label=f"{label} val",
                     marker="o", markersize=4)
        axes[0].plot(epochs, hist["train_loss"], ":",
                     color=color, linewidth=1.2, alpha=0.5,
                     label=f"{label} train")

        axes[1].plot(epochs, [a*100 for a in hist["val_acc"]], "-",
                     color=color, linewidth=2.2, label=f"{label} val",
                     marker="o", markersize=4)
        axes[1].plot(epochs, [a*100 for a in hist["train_acc"]], ":",
                     color=color, linewidth=1.2, alpha=0.5,
                     label=f"{label} train")

    axes[0].set_title("Loss (solid=val, dotted=train)")
    axes[1].set_title("Accuracy (solid=val, dotted=train)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("BCE Loss")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    out = LOG_DIR / "b6_all_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention weight extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_weights(
    model:  AttentionLSTM,
    tokens: list[str],
    vocab:  Vocabulary,
) -> tuple[np.ndarray, list[str]]:
    """
    Run one review through the model and return:
      - attention weights  (np array, length = num non-pad tokens)
      - token list         (strings, aligned to weights)
    """
    model.eval()
    encoded = vocab.encode(tokens)
    padded  = pad_sequence(encoded, max_len=MAX_SEQ_LEN)
    x       = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, attn_weights = model(x, return_attn=True)

    weights = attn_weights.squeeze(0).cpu().numpy()  # (seq_len,)

    # Trim to non-pad region (pre-padding: first pad_len items are 0)
    pad_len   = MAX_SEQ_LEN - len(encoded)
    weights   = weights[pad_len:]       # keep only real token weights
    real_toks = tokens[:MAX_SEQ_LEN]    # truncate if needed

    return weights, real_toks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attention heatmap for sample reviews
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmaps(
    model:   AttentionLSTM,
    vocab:   Vocabulary,
    samples: list[dict],          # [{"text": str, "label": str}, ...]
    top_k:   int = 30,            # show top-k tokens by weight
):
    """
    For each sample review, plot a horizontal bar chart of attention
    weights for the top-k tokens. Colour-coded by weight magnitude.
    """
    n   = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Attention Weights — What the Model Focuses On\n"
        "(top 30 tokens by attention weight)",
        fontsize=13, fontweight="bold"
    )

    cmap = plt.cm.YlOrRd

    for ax, sample in zip(axes, samples):
        tokens  = tokenize(clean_text(sample["text"]))
        weights, toks = get_attention_weights(model, tokens, vocab)

        # Top-k by weight
        if len(toks) > top_k:
            top_idx = np.argsort(weights)[-top_k:][::-1]
        else:
            top_idx = np.argsort(weights)[::-1]

        top_weights = weights[top_idx]
        top_tokens  = [toks[i] for i in top_idx]

        # Normalise for colour mapping
        norm_w  = top_weights / (top_weights.max() + 1e-9)
        bar_col = [cmap(w) for w in norm_w]

        bars = ax.barh(
            range(len(top_tokens)), top_weights,
            color=bar_col, edgecolor="white", height=0.7
        )
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Attention Weight")

        # Predicted label
        encoded = vocab.encode(tokens)
        padded  = pad_sequence(encoded)
        x       = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(x)
        pred  = "Positive ✓" if logit.item() > 0 else "Negative ✓"
        color = "#1D9E75"    if logit.item() > 0 else "#D85A30"

        ax.set_title(
            f"True label: {sample['label']}  |  "
            f"Predicted: {pred}  |  "
            f"Confidence: {torch.sigmoid(logit).item():.2%}",
            fontsize=10, color=color
        )

        # Annotate bar values
        for bar, val in zip(bars, top_weights):
            ax.text(
                val + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8
            )

    plt.tight_layout()
    out = LOG_DIR / "b6_attention_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attention distribution — entropy analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_entropy(
    model:  AttentionLSTM,
    vocab:  Vocabulary,
    loader,
    n_batches: int = 20,
):
    """
    Compute entropy of attention weight distribution over n_batches.
    Low entropy  → focused attention (model is selective)
    High entropy → diffuse attention (model looks at everything equally)
    """
    model.eval()
    entropies = []

    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            if i >= n_batches:
                break
            X = X.to(DEVICE)
            _, attn = model(X, return_attn=True)  # (B, S)

            # Shannon entropy per sample: H = -Σ p log p
            eps = 1e-9
            H   = -(attn * (attn + eps).log()).sum(dim=1)  # (B,)
            entropies.extend(H.cpu().numpy().tolist())

    entropies = np.array(entropies)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Attention Weight Distribution — Entropy Analysis\n"
        "(measures how focused vs diffuse the model's attention is)",
        fontsize=12, fontweight="bold"
    )

    axes[0].hist(entropies, bins=40, color="#D85A30",
                 edgecolor="white", linewidth=0.6)
    axes[0].axvline(entropies.mean(), color="#534AB7",
                    linewidth=2, linestyle="--",
                    label=f"Mean entropy: {entropies.mean():.3f}")
    axes[0].set_xlabel("Attention Entropy (nats)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Attention Entropy")
    axes[0].legend()

    # Cumulative distribution
    sorted_e = np.sort(entropies)
    cdf      = np.arange(1, len(sorted_e)+1) / len(sorted_e)
    axes[1].plot(sorted_e, cdf, color="#D85A30", linewidth=2.2)
    axes[1].set_xlabel("Attention Entropy (nats)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("CDF of Attention Entropy")
    axes[1].axvline(np.log(MAX_SEQ_LEN), color="gray",
                    linestyle=":", linewidth=1.2,
                    label=f"Max entropy (uniform): {np.log(MAX_SEQ_LEN):.2f}")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out = LOG_DIR / "b6_attention_entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")
    print(f"\n  Attention entropy stats:")
    print(f"  Mean  : {entropies.mean():.4f}")
    print(f"  Std   : {entropies.std():.4f}")
    print(f"  Min   : {entropies.min():.4f}")
    print(f"  Max   : {entropies.max():.4f}")
    print(f"  Max possible (uniform over 500): {np.log(500):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Computational overhead measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_overhead(test_loader, vocab):
    """
    Measure per-batch inference time and parameter count
    for all three models. Reports overhead of attention vs LSTM.
    """
    results = {}

    for name in MODELS:
        model = MODEL_REGISTRY[name](**MODEL_KWARGS)
        ckpt  = CHECKPOINT_DIR / f"{name}_best.pt"
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE)
        )
        model = model.to(DEVICE)
        model.eval()

        # Parameter count
        total_params  = sum(p.numel() for p in model.parameters())
        train_params  = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Measure inference time over 50 batches
        times = []
        with torch.no_grad():
            for i, (X, _) in enumerate(test_loader):
                if i >= 50:
                    break
                X = X.to(DEVICE)
                t0 = time.perf_counter()
                _  = model(X)
                # Sync MPS/CUDA before timing
                if str(DEVICE) == "mps":
                    torch.mps.synchronize()
                elif str(DEVICE) == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        results[name] = {
            "total_params": total_params,
            "train_params": train_params,
            "mean_ms":      np.mean(times)  * 1000,
            "std_ms":       np.std(times)   * 1000,
            "min_ms":       np.min(times)   * 1000,
        }

    # Print overhead table
    print("\n" + "=" * 68)
    print(f"  Computational Overhead — Inference Time (batch_size=64)")
    print(f"  {'Model':<18} {'Params':>10} {'Mean ms':>10}"
          f" {'Std ms':>9} {'Overhead vs LSTM':>18}")
    print("  " + "─" * 60)

    lstm_ms = results["lstm"]["mean_ms"]
    for name in MODELS:
        r        = results[name]
        overhead = (r["mean_ms"] - lstm_ms) / lstm_ms * 100
        ovr_str  = f"{overhead:+.1f}%" if name != "lstm" else "baseline"
        print(f"  {MODEL_LABELS[name]:<18} "
              f"{r['total_params']:>10,} "
              f"{r['mean_ms']:>9.2f}ms "
              f"{r['std_ms']:>8.2f}ms "
              f"{ovr_str:>18}")
    print("=" * 68)

    # Extra params in attention vs LSTM
    attn_p = results["attention"]["total_params"]
    lstm_p = results["lstm"]["total_params"]
    print(f"\n  Attention extra params vs LSTM : "
          f"{attn_p - lstm_p:,} "
          f"({(attn_p-lstm_p)/lstm_p*100:.2f}%)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    get_device_info()
    set_seed()

    data = run_pipeline(force=False)
    _, _, test_loader = get_dataloaders(data)
    vocab = data["vocab"]

    # ── Load all three models from checkpoints ────────────────────────────
    models_loaded = {}
    test_results  = {}
    for name in MODELS:
        m = MODEL_REGISTRY[name](**MODEL_KWARGS)
        test_results[name] = evaluate_test(name, m, test_loader)
        # reload cleanly for attention extraction
        m2 = MODEL_REGISTRY[name](**MODEL_KWARGS)
        ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
        m2.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m2 = m2.to(DEVICE)
        models_loaded[name] = m2

    # ── Load histories ─────────────────────────────────────────────────────
    histories = {n: load_history(n) for n in MODELS}

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"  {'Model':<18} {'Test Acc':>10} {'Test Loss':>11}")
    print("  " + "─" * 42)
    for name in MODELS:
        r = test_results[name]
        print(f"  {MODEL_LABELS[name]:<18} "
              f"{r['accuracy']:>9.2%}  {r['loss']:>10.4f}")
    print("=" * 52)

    # ── Pick 4 real IMDB reviews for attention viz ────────────────────────
    raw = load_dataset("imdb")
    # 2 positive, 2 negative — short-ish reviews for readability
    samples = []
    pos_count = neg_count = 0
    for item in raw["test"]:
        word_count = len(item["text"].split())
        if item["label"] == 1 and pos_count < 2 and 80 < word_count < 200:
            samples.append({
                "text":  item["text"],
                "label": "Positive"
            })
            pos_count += 1
        elif item["label"] == 0 and neg_count < 2 and 80 < word_count < 200:
            samples.append({
                "text":  item["text"],
                "label": "Negative"
            })
            neg_count += 1
        if pos_count == 2 and neg_count == 2:
            break

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_three_model_comparison(test_results)
    plot_all_three_curves(histories)
    plot_attention_heatmaps(models_loaded["attention"], vocab, samples)
    plot_attention_entropy(
        models_loaded["attention"], vocab, test_loader
    )

    # ── Computational overhead ────────────────────────────────────────────
    overhead = measure_overhead(test_loader, vocab)

    print("\nDone. All B6 outputs saved to outputs/logs/")


if __name__ == "__main__":
    main()