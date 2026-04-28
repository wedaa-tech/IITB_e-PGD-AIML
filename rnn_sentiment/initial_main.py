from src.utils import set_seed, get_device_info
from src.preprocess import run_pipeline
from src.dataset import get_dataloaders

def main():
    print("=" * 50)
    print("  RNN Sentiment Classification — IMDB")
    print("=" * 50)

    get_device_info()
    set_seed()

    # Run (or load cached) preprocessing
    data = run_pipeline(force=False)

    # Build data loaders
    train_loader, val_loader, test_loader = get_dataloaders(data)

    # Sanity check
    Xb, yb = next(iter(train_loader))
    print(f"\nSanity check — batch shape : {Xb.shape}")   # [64, 500]
    print(f"Label batch shape          : {yb.shape}")    # [64]
    print(f"Label distribution in batch: pos={yb.sum().int()}, "
          f"neg={int(len(yb)-yb.sum())}")

    print("\nPipeline complete. Ready for model training.")

if __name__ == "__main__":
    main()