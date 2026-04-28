"""
Load saved checkpoints + histories and generate all plots/reports.
No retraining needed.
"""
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils      import set_seed, get_device_info
from src.preprocess import run_pipeline
from src.dataset    import get_dataloaders, IMDBDataset
from src.models     import MODEL_REGISTRY
from src.trainer    import evaluate_test
from src.config     import VOCAB_SIZE, DEVICE, CHECKPOINT_DIR
from src.visualize  import (
    plot_training_curves, plot_single_curve,
    plot_confusion_matrices, plot_comparison_bar,
    print_classification_reports, load_history,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

MODELS_TO_ANALYZE = ["rnn", "lstm", "attention"]

MODEL_KWARGS = dict(
    vocab_size = VOCAB_SIZE + 2,
    embed_dim  = 128,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)

def main():
    get_device_info()
    set_seed()

    # Load preprocessed data (from cache — fast)
    data = run_pipeline(force=False)
    _, _, test_loader = get_dataloaders(data)

    # ── Load histories from disk ───────────────────────────────────────────
    histories = {name: load_history(name) for name in MODELS_TO_ANALYZE}

    # ── Evaluate each model from saved checkpoint ─────────────────────────
    test_results = {}
    for name in MODELS_TO_ANALYZE:
        model = MODEL_REGISTRY[name](**MODEL_KWARGS)
        test_results[name] = evaluate_test(name, model, test_loader)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 47)
    print(f"  {'Model':<14} {'Test Loss':>10} {'Test Acc':>10}")
    print("  " + "─" * 37)
    for name, r in test_results.items():
        print(f"  {name:<14} {r['loss']:>10.4f} {r['accuracy']:>9.2%}")
    print("=" * 47)

    # ── All plots ─────────────────────────────────────────────────────────
    plot_training_curves(histories)
    for name, hist in histories.items():
        plot_single_curve(name, hist)
    plot_confusion_matrices(test_results)
    plot_comparison_bar(test_results)
    print_classification_reports(test_results)

    print("\nDone. All outputs in outputs/logs/")

if __name__ == "__main__":
    main()