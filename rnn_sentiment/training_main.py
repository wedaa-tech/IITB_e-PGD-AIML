import logging
from src.utils        import set_seed, get_device_info
from src.preprocess   import run_pipeline
from src.dataset      import get_dataloaders
from src.models       import MODEL_REGISTRY
from src.trainer      import train, evaluate_test
from src.config       import VOCAB_SIZE, DEVICE

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Shared hyperparams ────────────────────────────────────────────────────────
MODEL_KWARGS = dict(
    vocab_size = VOCAB_SIZE + 2,    # +2 for <PAD> and <UNK>
    embed_dim  = 128,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.5,
    pad_idx    = 0,
)

MODELS_TO_RUN = ["rnn", "lstm", "attention"]   # change to run a subset

def main():
    print("=" * 55)
    print("  RNN Sentiment — IMDB")
    print("=" * 55)
    get_device_info()
    set_seed()

    # ── Data ─────────────────────────────────────────────────────────────
    data = run_pipeline(force=False)
    train_loader, val_loader, test_loader = get_dataloaders(data)

    # ── Train + evaluate each model ───────────────────────────────────────
    results = {}
    for name in MODELS_TO_RUN:
        model   = MODEL_REGISTRY[name](**MODEL_KWARGS)
        history = train(name, model, train_loader, val_loader)
        results[name] = evaluate_test(name, model, test_loader)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"  {'Model':<12} {'Test Loss':>10} {'Test Acc':>10}")
    print("  " + "─" * 35)
    for name, r in results.items():
        print(f"  {name:<12} {r['loss']:>10.4f} {r['accuracy']:>9.2%}")
    print("=" * 45)

if __name__ == "__main__":
    main()