import logging
from src.utils      import set_seed, get_device_info
from src.preprocess import run_pipeline
from src.dataset    import get_dataloaders
from src.models     import MODEL_REGISTRY
from src.trainer    import train, evaluate_test
from src.config     import VOCAB_SIZE
from src.visualize  import (
    plot_training_curves, plot_single_curve,
    plot_confusion_matrices, plot_comparison_bar,
    print_classification_reports, load_history,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

MODEL_KWARGS = dict(
    vocab_size = VOCAB_SIZE + 2,
    embed_dim  = 128,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)

MODELS_TO_RUN = ["rnn", "lstm", "attention"]

def main():
    print("=" * 55)
    print("  RNN Sentiment — IMDB")
    print("=" * 55)
    get_device_info()
    set_seed()

    data = run_pipeline(force=False)
    train_loader, val_loader, test_loader = get_dataloaders(data)

    # ── Train + evaluate ──────────────────────────────────────────────────
    histories    = {}
    test_results = {}

    for name in MODELS_TO_RUN:
        model            = MODEL_REGISTRY[name](**MODEL_KWARGS)
        histories[name]  = train(name, model, train_loader, val_loader)
        test_results[name] = evaluate_test(name, model, test_loader)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 47)
    print(f"  {'Model':<14} {'Test Loss':>10} {'Test Acc':>10}")
    print("  " + "─" * 37)
    for name, r in test_results.items():
        print(f"  {name:<14} {r['loss']:>10.4f} {r['accuracy']:>9.2%}")
    print("=" * 47)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_training_curves(histories)
    for name, hist in histories.items():
        plot_single_curve(name, hist)
    plot_confusion_matrices(test_results)
    plot_comparison_bar(test_results)
    print_classification_reports(test_results)
    print(f"\nAll plots saved to outputs/logs/")

if __name__ == "__main__":
    main()