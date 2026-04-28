import logging
from src.utils          import set_seed, get_device_info
from src.preprocess     import run_pipeline
from src.dataset        import get_dataloaders
from src.hparam_search  import (
    run_hparam_sweep, plot_hparam_heatmaps,
    plot_gradient_norms, plot_best_training_curves,
    print_hparam_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

def main():
    get_device_info()
    set_seed()

    data = run_pipeline(force=False)
    train_loader, val_loader, test_loader = get_dataloaders(data)
    vocab = data["vocab"]

    # Run sweep (or load from cache if already done)
    results = run_hparam_sweep(
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        vocab        = vocab,
        models_to_sweep = ["rnn", "lstm"],
        force        = False,
    )

    # Summary table + find best configs
    best_configs = print_hparam_summary(results)

    # All plots
    plot_hparam_heatmaps(results)
    plot_best_training_curves(results, best_configs)
    plot_gradient_norms(results, best_configs)

    print("\nDone. Outputs in outputs/logs/")

if __name__ == "__main__":
    main()