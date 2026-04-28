"""
B4 analysis — loads saved experiment results and generates all B4 plots.
Run this AFTER experiment.py completes.
"""
import copy
import logging
import pickle
from src.utils     import set_seed, get_device_info
from src.preprocess import run_pipeline
from src.dataset   import get_dataloaders
from src.models    import MODEL_REGISTRY
from src.trainer   import evaluate_test
from src.config    import VOCAB_SIZE, LOG_DIR
from src.embeddings import REAL_VOCAB_SIZE, get_embedding
from src.visualize  import (
    plot_embedding_comparison,
    plot_training_curves,
    print_classification_reports,
    load_history,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODELS     = ["rnn", "lstm", "attention"]
EMBEDDINGS = ["word2vec", "glove"]

MODEL_KWARGS = dict(
    vocab_size = REAL_VOCAB_SIZE,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)


def _load_tokenized_train() -> list[list[str]] | None:
    """
    Load the cached tokenised training corpus that experiment.py creates
    for Word2Vec. Returns None if it's not on disk — in that case GloVe
    still works but Word2Vec can't be rebuilt.
    """
    cache = LOG_DIR.parent.parent / "data/processed/tokenized_train.pkl"
    if not cache.exists():
        log.warning(f"Tokenised training corpus not found at {cache}. "
                    f"Re-run experiment.py first if Word2Vec is needed.")
        return None
    with open(cache, "rb") as f:
        return pickle.load(f)


def main():
    get_device_info()
    set_seed()

    data = run_pipeline(force=False)
    _, _, test_loader = get_dataloaders(data)
    vocab             = data["vocab"]

    # ── Load histories ────────────────────────────────────────────────────
    histories = {
        f"{m}_{e}": load_history(f"{m}_{e}")
        for m in MODELS for e in EMBEDDINGS
    }

    # ── Rebuild pretrained embeddings (must match what experiment.py used,
    #    otherwise the checkpoint shapes won't line up) ─────────────────────
    tokenized_train = _load_tokenized_train()
    emb_layers: dict[str, "object"] = {}
    for emb in EMBEDDINGS:
        log.info(f"Rebuilding embedding layer: {emb}")
        emb_layer, _ = get_embedding(
            strategy        = emb,
            vocab           = vocab,
            tokenized_train = tokenized_train if emb == "word2vec" else None,
            trainable       = True,
        )
        emb_layers[emb] = emb_layer

    # ── Re-evaluate from checkpoints ──────────────────────────────────────
    test_results = {}
    for emb in EMBEDDINGS:
        for m in MODELS:
            key   = f"{m}_{emb}"
            # Deep-copy the embedding so each model owns its own parameters
            # (same pattern as experiment.py), matching the checkpoint shape.
            model = MODEL_REGISTRY[m](
                **MODEL_KWARGS,
                embedding = copy.deepcopy(emb_layers[emb]),
            )
            test_results[key] = evaluate_test(key, model, test_loader)

    # ── Summary grid ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  {'Model':<16} {'Word2Vec Acc':>13} {'GloVe Acc':>11}")
    print("  " + "─" * 43)
    for m in MODELS:
        w2v = test_results[f"{m}_word2vec"]["accuracy"] * 100
        glv = test_results[f"{m}_glove"]["accuracy"]   * 100
        print(f"  {m:<16} {w2v:>12.2f}%  {glv:>10.2f}%")
    print("=" * 55)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_embedding_comparison(test_results)

    # Training curves per embedding
    for emb in EMBEDDINGS:
        subset = {m: histories[f"{m}_{emb}"] for m in MODELS}
        plot_training_curves(subset)

    print_classification_reports(test_results)
    print("\nDone. All B4 outputs in outputs/logs/")

if __name__ == "__main__":
    main()