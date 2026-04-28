"""
Full experiment grid:
  3 models  ×  3 embeddings  =  9 training runs

Results saved to outputs/logs/experiment_results.pkl
"""

import logging
import pickle
import numpy as np
from pathlib import Path

from src.utils       import set_seed, get_device_info
from src.preprocess  import run_pipeline, preprocess_text
from src.dataset     import get_dataloaders
from src.models      import MODEL_REGISTRY
from src.trainer     import train, evaluate_test
from src.embeddings  import get_embedding, REAL_VOCAB_SIZE
from src.config      import VOCAB_SIZE, LOG_DIR, NUM_EPOCHS
from src.visualize   import (
    plot_training_curves, plot_confusion_matrices,
    plot_comparison_bar, print_classification_reports,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Grid ─────────────────────────────────────────────────────────────────────
MODELS     = ["rnn", "lstm", "attention"]
EMBEDDINGS = ["word2vec", "glove"]

# Shared model kwargs (embedding layer injected separately)
MODEL_KWARGS = dict(
    vocab_size = REAL_VOCAB_SIZE,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)

# One-hot uses vocab_size-dim inputs → needs smaller hidden to stay fair
#ONEHOT_KWARGS = {**MODEL_KWARGS, "hidden_dim": 128}


def main():
    get_device_info()
    set_seed()

    # ── Load data ─────────────────────────────────────────────────────────
    data = run_pipeline(force=False)
    train_loader, val_loader, test_loader = get_dataloaders(data)
    vocab = data["vocab"]

    # ── Pre-tokenized training docs for Word2Vec ───────────────────────────
    log.info("Preparing tokenized corpus for Word2Vec …")
    tokenized_train_path = LOG_DIR.parent.parent / "data/processed/tokenized_train.pkl"

    # Load raw train texts and re-tokenize (or cache)
    from datasets import load_dataset
    if not tokenized_train_path.exists():
        raw   = load_dataset("imdb")
        texts = raw["train"]["text"][:20000]   # training portion only
        from src.preprocess import preprocess_text, clean_text, tokenize
        tokenized = [tokenize(clean_text(t)) for t in texts]
        with open(tokenized_train_path, "wb") as f:
            pickle.dump(tokenized, f)
    else:
        with open(tokenized_train_path, "rb") as f:
            tokenized = pickle.load(f)

    # ── Run grid ──────────────────────────────────────────────────────────
    all_histories    = {}
    all_test_results = {}

    for emb_name in EMBEDDINGS:
        log.info(f"\n{'─'*55}")
        log.info(f"  Embedding strategy : {emb_name.upper()}")
        log.info(f"{'─'*55}")

        emb_layer, emb_dim = get_embedding(
            strategy        = emb_name,
            vocab           = vocab,
            tokenized_train = tokenized if emb_name == "word2vec" else None,
            trainable       = True,
        )

        for model_name in MODELS:
            run_key = f"{model_name}_{emb_name}"
            log.info(f"\n  ── Run: {run_key} ──")

            import copy
            emb_copy = copy.deepcopy(emb_layer)

            model = MODEL_REGISTRY[model_name](
                **MODEL_KWARGS, embedding=emb_copy   # same kwargs for all
            )

            history = train(
                model_name   = run_key,
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                num_epochs   = NUM_EPOCHS,
                patience     = 3,
            )

            test_res = evaluate_test(run_key, model, test_loader)
            all_histories[run_key]    = history
            all_test_results[run_key] = test_res

    # ── Save all results ──────────────────────────────────────────────────
    results_path = LOG_DIR / "experiment_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "histories":    all_histories,
            "test_results": all_test_results,
        }, f)
    log.info(f"\nAll results saved → {results_path}")

    # ── Print summary grid ────────────────────────────────────────────────
    _print_grid(all_test_results)

    # ── Plots per embedding strategy ──────────────────────────────────────
    for emb_name in EMBEDDINGS:
        subset_h = {m: all_histories[f"{m}_{emb_name}"]    for m in MODELS}
        subset_r = {m: all_test_results[f"{m}_{emb_name}"] for m in MODELS}
        plot_training_curves(subset_h)
        plot_confusion_matrices(subset_r)

    print_classification_reports(all_test_results)


def _print_grid(results: dict):
    EMBEDDINGS = ["word2vec", "glove"]      # updated here too
    print("\n" + "=" * 52)
    print(f"  Test Accuracy Grid")
    print(f"  {'Model':<16} {'Word2Vec':>10} {'GloVe':>10}")
    print("  " + "─" * 40)
    for m in MODELS:
        row = f"  {m:<16}"
        for e in EMBEDDINGS:
            acc = results[f"{m}_{e}"]["accuracy"] * 100
            row += f" {acc:>9.2f}%"
        print(row)
    print("=" * 52)

    print(f"\n  Test Loss Grid")
    print(f"  {'Model':<16} {'Word2Vec':>10} {'GloVe':>10}")
    print("  " + "─" * 40)
    for m in MODELS:
        row = f"  {m:<16}"
        for e in EMBEDDINGS:
            loss = results[f"{m}_{e}"]["loss"]
            row += f" {loss:>10.4f}"
        print(row)
    print("=" * 52)


if __name__ == "__main__":
    main()