"""
run_scratch.py
──────────────
Entry point: train the custom CNN from scratch on CIFAR-100.

Usage:
    python run_scratch.py                        # standard run
    python run_scratch.py --variant small        # lightweight model (~1.5M params)
    python run_scratch.py --variant residual     # residual model (~18M params)
    python run_scratch.py --epochs 30            # override epoch count
    python run_scratch.py --lr 5e-4              # override learning rate
    python run_scratch.py --batch-size 64        # override batch size
    python run_scratch.py --schedule cosine      # LR schedule (cosine/step/warmup)
    python run_scratch.py --resume               # resume from last checkpoint
    python run_scratch.py --dry-run              # 2 epochs, smoke-test the pipeline

    PYTORCH_ENABLE_MPS_FALLBACK=1 python run_scratch.py   # if MPS errors occur
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# ── Project imports ───────────────────────────────────────────────────────────
from src.config import (
    DEVICE,
    SCRATCH,
    NUM_CLASSES,
    RANDOM_SEED,
    RESULT_DIR,
    CKPT_DIR,
)
from src.utils import (
    set_seed,
    check_outputs_dir,
    log_system_info,
    count_parameters,
    model_summary,
    save_history,
    format_time,
    EpochProgressLogger,
    plot_lr_schedule,
)
from src.dataset import get_dataloaders
from src.models.scratch_cnn import build_scratch_cnn, count_parameters as cnn_count
from src.train import (
    run_training,
    make_scheduler,
    EarlyStopping,
    load_checkpoint,
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    All arguments have defaults pulled from src/config.py so the script
    runs with zero arguments for the standard assignment configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train a custom CNN from scratch on CIFAR-100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--variant",
        type    = str,
        default = "standard",
        choices = ["standard", "small", "residual"],
        help    = "CNN variant: standard (~9.2M), small (~1.5M), residual (~18M)",
    )

    # Training duration
    parser.add_argument(
        "--epochs",
        type    = int,
        default = SCRATCH["epochs"],
        help    = "Number of training epochs",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Run 2 epochs only — verifies the pipeline without a full train",
    )

    # Optimiser
    parser.add_argument(
        "--lr",
        type    = float,
        default = SCRATCH["lr"],
        help    = "Initial learning rate for AdamW",
    )
    parser.add_argument(
        "--weight-decay",
        type    = float,
        default = SCRATCH["weight_decay"],
        help    = "AdamW weight decay (L2 regularisation)",
    )
    parser.add_argument(
        "--label-smoothing",
        type    = float,
        default = 0.1,
        help    = "Label smoothing factor for cross-entropy loss (0 = off)",
    )

    # Data
    parser.add_argument(
        "--batch-size",
        type    = int,
        default = SCRATCH["batch_size"],
        help    = "Mini-batch size",
    )
    parser.add_argument(
        "--num-workers",
        type    = int,
        default = SCRATCH["num_workers"],
        help    = "DataLoader workers (keep 0 on macOS to avoid multiprocessing errors)",
    )

    # Schedule
    parser.add_argument(
        "--schedule",
        type    = str,
        default = "cosine",
        choices = ["cosine", "step", "warmup", "plateau"],
        help    = "Learning rate schedule",
    )
    parser.add_argument(
        "--warmup-epochs",
        type    = int,
        default = 5,
        help    = "Linear warm-up epochs (only used when --schedule warmup)",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type    = int,
        default = 15,
        help    = "Early stopping patience (epochs). 0 = disabled",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        action  = "store_true",
        help    = "Resume training from the last saved checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type    = Path,
        default = SCRATCH["checkpoint"],
        help    = "Path to save / load the best checkpoint",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type    = int,
        default = RANDOM_SEED,
        help    = "Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-summary",
        action  = "store_true",
        help    = "Skip printing the model layer summary (faster startup)",
    )
    parser.add_argument(
        "--plot-lr",
        action  = "store_true",
        help    = "Plot the LR schedule before training and exit",
    )

    return parser.parse_args()


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight(args: argparse.Namespace) -> None:
    """
    Run all environment and configuration checks before any training starts.
    A failed check raises immediately with a clear message rather than
    crashing mid-epoch after wasting compute time.
    """
    print("\n" + "═" * 60)
    print("  Pre-flight checks")
    print("═" * 60)

    # Output directories
    check_outputs_dir()

    # Device
    print(f"\n  Device         : {DEVICE}")
    if DEVICE.type == "cpu":
        print("  WARNING: Running on CPU — training will be slow.")
        print("  Expected time : ~3–5 hours for 50 epochs on Intel Mac.")
        print("  Tip: Use --variant small or --epochs 20 for a faster run.")
    elif DEVICE.type == "mps":
        print("  Apple Silicon MPS detected — GPU acceleration enabled.")
        print("  Expected time : ~45–70 minutes for 50 epochs.")

    # Data directory
    cifar_path = Path("data") / "cifar-100-python"
    if not cifar_path.exists():
        raise FileNotFoundError(
            f"\n  CIFAR-100 data not found at: {cifar_path}\n"
            f"  Expected structure:\n"
            f"    data/cifar-100-python/train\n"
            f"    data/cifar-100-python/test\n"
            f"    data/cifar-100-python/meta\n\n"
            f"  If you have cifar-100-python.tar.gz:\n"
            f"    cd data && tar -xzf cifar-100-python.tar.gz"
        )
    print(f"\n  CIFAR-100 data : found at {cifar_path}")

    # Resume check
    if args.resume and not args.checkpoint.exists():
        raise FileNotFoundError(
            f"\n  --resume requested but checkpoint not found: {args.checkpoint}\n"
            f"  Run without --resume to start a fresh training run."
        )

    # Dry-run override
    if args.dry_run:
        print("\n  DRY RUN mode — overriding epochs to 2")
        args.epochs = 2

    print("\n  All pre-flight checks passed.\n")


# ── Configuration summary ─────────────────────────────────────────────────────

def print_config(args: argparse.Namespace) -> None:
    """Print a full configuration table so every run is self-documenting."""
    print("═" * 60)
    print("  Training configuration — Scratch CNN")
    print("═" * 60)
    rows = [
        ("Model variant",    args.variant),
        ("Epochs",           args.epochs),
        ("Batch size",       args.batch_size),
        ("Learning rate",    f"{args.lr:.2e}"),
        ("Weight decay",     f"{args.weight_decay:.2e}"),
        ("Label smoothing",  args.label_smoothing),
        ("LR schedule",      args.schedule),
        ("Warmup epochs",    args.warmup_epochs if args.schedule == "warmup" else "n/a"),
        ("Early stop patience", args.patience if args.patience > 0 else "disabled"),
        ("Device",           str(DEVICE)),
        ("Seed",             args.seed),
        ("Checkpoint",       str(args.checkpoint)),
        ("Resume",           args.resume),
        ("Dry run",          args.dry_run),
    ]
    for label, value in rows:
        print(f"  {label:<24} {value}")
    print("═" * 60 + "\n")


# ── Main training routine ─────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Pre-flight ─────────────────────────────────────────────────────────
    preflight(args)

    # ── Reproducibility ────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── System info ────────────────────────────────────────────────────────
    log_system_info()

    # ── Config summary ─────────────────────────────────────────────────────
    print_config(args)

    # ── Data ───────────────────────────────────────────────────────────────
    print("── Loading data ─────────────────────────────────────────────")
    train_loader, val_loader, test_loader = get_dataloaders(
        mode        = "scratch",
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        input_size  = SCRATCH["input_size"],
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n── Building model ───────────────────────────────────────────")
    model = build_scratch_cnn(
        variant     = args.variant,
        num_classes = NUM_CLASSES,
    ).to(DEVICE)

    if not args.no_summary:
        model_summary(model, input_size=(1, 3, 32, 32), max_rows=40)

    param_counts = cnn_count(model)

    # ── Optimiser ──────────────────────────────────────────────────────────
    print("\n── Configuring optimiser ────────────────────────────────────")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )
    print(f"  Optimiser : AdamW  lr={args.lr:.2e}  "
          f"wd={args.weight_decay:.2e}  betas=(0.9, 0.999)")

    # ── LR Scheduler ───────────────────────────────────────────────────────
    print(f"\n── Configuring LR schedule ({args.schedule}) ────────────────")
    scheduler = make_scheduler(
        optimizer = optimizer,
        schedule  = args.schedule,
        epochs    = args.epochs,
        warmup    = args.warmup_epochs,
    )
    print(f"  Schedule  : {args.schedule}  "
          f"({'T_max=' + str(args.epochs) if args.schedule == 'cosine' else ''})")

    # Optionally plot the LR curve and exit
    if args.plot_lr:
        print("\n-- Plotting LR schedule (--plot-lr flag set) ─────────────")
        # Use fresh copies so the simulation does not advance the real scheduler
        opt_sim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sch_sim = make_scheduler(opt_sim, args.schedule, args.epochs, args.warmup_epochs)
        plot_lr_schedule(opt_sim, sch_sim, epochs=args.epochs,
                          save_name="scratch_lr_schedule.png", show=True)
        print("LR schedule saved. Exiting (remove --plot-lr to start training).")
        sys.exit(0)

    # ── Early stopping ─────────────────────────────────────────────────────
    early_stopping = None
    if args.patience > 0:
        early_stopping = EarlyStopping(
            patience  = args.patience,
            min_delta = 1e-4,
            verbose   = True,
        )
        print(f"\n  Early stopping : patience={args.patience} epochs")

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_path = args.checkpoint if args.resume else None

    # ── Training ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Starting training")
    print("═" * 60)

    wall_start = time.perf_counter()

    history = run_training(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        epochs          = args.epochs,
        optimizer       = optimizer,
        scheduler       = scheduler,
        label           = f"Scratch-{args.variant}",
        checkpoint_path = args.checkpoint,
        early_stopping  = early_stopping,
        label_smoothing = args.label_smoothing,
        resume_from     = resume_path,
    )

    wall_total = time.perf_counter() - wall_start

    # ── Save history ───────────────────────────────────────────────────────
    print("\n── Saving history ───────────────────────────────────────────")
    history_file = f"scratch_{args.variant}_history.json"
    save_history(history, history_file)

    # ── Quick test-set evaluation ──────────────────────────────────────────
    print("\n── Quick test-set evaluation ────────────────────────────────")
    print("  Loading best checkpoint for final test evaluation …")

    best_ckpt = load_checkpoint(model, args.checkpoint)
    model.eval()

    correct_1 = 0
    correct_5 = 0
    total     = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)

            # Top-1
            correct_1 += (logits.argmax(dim=1) == labels).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=1).indices
            correct_5 += sum(
                labels[i].item() in top5[i].tolist()
                for i in range(len(labels))
            )
            total += len(labels)

    top1 = correct_1 / total
    top5 = correct_5 / total

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training complete — final results")
    print("═" * 60)
    print(f"  Model variant      : {args.variant}")
    print(f"  Total parameters   : {param_counts['total']:,}")
    print(f"  Epochs trained     : {len(history['val_acc'])}")
    print(f"  Best val accuracy  : {max(history['val_acc']):.4f}  "
          f"(epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
    print(f"  Test top-1 accuracy: {top1:.4f}  ({top1*100:.2f}%)")
    print(f"  Test top-5 accuracy: {top5:.4f}  ({top5*100:.2f}%)")
    print(f"  Total wall time    : {format_time(wall_total)}")
    print(f"  Avg epoch time     : {format_time(sum(history['epoch_time']) / len(history['epoch_time']))}")
    print(f"  Checkpoint saved   : {args.checkpoint}")
    print(f"  History saved      : {RESULT_DIR / history_file}")
    print("═" * 60)

    # ── Next step hint ─────────────────────────────────────────────────────
    print("\n  Next steps:")
    print("    1. Train the transfer model :  python run_transfer.py")
    print("    2. Compare both models      :  python compare.py")
    print("    3. Explore plots in         :  outputs/plots/")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()