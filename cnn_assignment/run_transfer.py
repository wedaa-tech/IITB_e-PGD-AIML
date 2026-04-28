"""
run_transfer.py
───────────────
Entry point: fine-tune a pretrained backbone on CIFAR-100 using
a two-phase transfer learning strategy.

Phase A — Feature extraction  (head only, backbone frozen)
    The backbone's ImageNet weights are frozen.
    Only the randomly-initialised classification head is trained.
    Higher learning rate (1e-3) because the head starts from scratch.

Phase B — Full fine-tuning  (all layers, discriminative LRs)
    Every layer is unfrozen and trained end-to-end.
    Backbone LR (1e-5) is kept very small to prevent catastrophic
    forgetting of ImageNet features. Head LR (1e-3) stays higher.

Usage:
    python run_transfer.py                            # standard run
    python run_transfer.py --backbone resnet34        # swap backbone
    python run_transfer.py --backbone efficientnet_b1 --input-size 240
    python run_transfer.py --epochs-frozen 10 --epochs-finetune 20
    python run_transfer.py --phase-a-only             # run Phase A only
    python run_transfer.py --phase-b-only             # Phase B only (needs checkpoint)
    python run_transfer.py --dry-run                  # 2+2 epochs smoke test
    python run_transfer.py --list-backbones           # show available backbones

    PYTORCH_ENABLE_MPS_FALLBACK=1 python run_transfer.py  # if MPS errors occur
"""

import argparse
import sys
import time
import json
from pathlib import Path

import torch

# ── Project imports ───────────────────────────────────────────────────────────
from src.config import (
    DEVICE,
    TRANSFER,
    NUM_CLASSES,
    RANDOM_SEED,
    RESULT_DIR,
    CKPT_DIR,
    PLOT_DIR,
)
from src.utils import (
    set_seed,
    check_outputs_dir,
    log_system_info,
    save_history,
    format_time,
    plot_lr_schedule,
    model_summary,
)
from src.dataset import get_dataloaders
from src.models.transfer_model import (
    build_transfer_model,
    freeze_backbone,
    unfreeze_all,
    get_param_groups,
    count_parameters,
    print_backbone_table,
    SUPPORTED_BACKBONES,
)
from src.train import (
    run_training,
    make_scheduler,
    EarlyStopping,
    load_checkpoint,
)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    All arguments default to values from src/config.py TRANSFER dict
    so the script runs correctly with zero arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained backbone on CIFAR-100 (two-phase TL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Backbone ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--backbone",
        type    = str,
        default = TRANSFER["backbone"],
        help    = "timm backbone name (e.g. efficientnet_b0, resnet50).",
    )
    parser.add_argument(
        "--input-size",
        type    = int,
        default = TRANSFER["input_size"],
        help    = "Spatial size images are resized to before the model.",
    )
    parser.add_argument(
        "--list-backbones",
        action  = "store_true",
        help    = "Print the curated backbone table and exit.",
    )
    parser.add_argument(
        "--no-pretrained",
        action  = "store_true",
        help    = "Skip ImageNet weights — train with random init (ablation only).",
    )

    # ── Phase A ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs-frozen",
        type    = int,
        default = TRANSFER["epochs_frozen"],
        help    = "Phase A: epochs to train with backbone frozen.",
    )
    parser.add_argument(
        "--lr-head",
        type    = float,
        default = TRANSFER["lr_head"],
        help    = "Phase A & B: learning rate for the classification head.",
    )

    # ── Phase B ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs-finetune",
        type    = int,
        default = TRANSFER["epochs_finetune"],
        help    = "Phase B: epochs to train with all layers unfrozen.",
    )
    parser.add_argument(
        "--lr-backbone",
        type    = float,
        default = TRANSFER["lr_backbone"],
        help    = "Phase B: learning rate for backbone parameters (keep small).",
    )

    # ── Shared training ───────────────────────────────────────────────────
    parser.add_argument(
        "--weight-decay",
        type    = float,
        default = TRANSFER["weight_decay"],
        help    = "AdamW weight decay applied to all param groups.",
    )
    parser.add_argument(
        "--label-smoothing",
        type    = float,
        default = 0.1,
        help    = "Label smoothing factor for cross-entropy loss (0 = off).",
    )
    parser.add_argument(
        "--schedule",
        type    = str,
        default = "cosine",
        choices = ["cosine", "step", "warmup", "plateau"],
        help    = "LR schedule used in both phases.",
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--batch-size",
        type    = int,
        default = TRANSFER["batch_size"],
        help    = "Mini-batch size (smaller than scratch — upscaled images use more RAM).",
    )
    parser.add_argument(
        "--num-workers",
        type    = int,
        default = TRANSFER["num_workers"],
        help    = "DataLoader workers (keep 0 on macOS).",
    )

    # ── Early stopping ────────────────────────────────────────────────────
    parser.add_argument(
        "--patience",
        type    = int,
        default = 10,
        help    = "Early stopping patience per phase (epochs). 0 = disabled.",
    )

    # ── Phase control ─────────────────────────────────────────────────────
    parser.add_argument(
        "--phase-a-only",
        action  = "store_true",
        help    = "Run Phase A only — skip Phase B fine-tuning.",
    )
    parser.add_argument(
        "--phase-b-only",
        action  = "store_true",
        help    = "Skip Phase A — start directly with Phase B full fine-tuning.\n"
                  "Requires an existing Phase A checkpoint.",
    )

    # ── Checkpoints ───────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-a",
        type    = Path,
        default = CKPT_DIR / "transfer_phase_a_best.pth",
        help    = "Path to save/load the Phase A best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type    = Path,
        default = TRANSFER["checkpoint"],
        help    = "Path to save the final best checkpoint (Phase B, or Phase A if --phase-a-only).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Run 2 epochs per phase — smoke-tests the full pipeline.",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = RANDOM_SEED,
        help    = "Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-summary",
        action  = "store_true",
        help    = "Skip printing the model layer summary.",
    )
    parser.add_argument(
        "--plot-lr",
        action  = "store_true",
        help    = "Plot both phase LR schedules and exit (no training).",
    )

    return parser.parse_args()


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight(args: argparse.Namespace) -> None:
    """
    Validate the environment, paths, and argument combinations
    before any model loading or training begins.
    """
    print("\n" + "═" * 62)
    print("  Pre-flight checks")
    print("═" * 62)

    # Output directories
    check_outputs_dir()

    # Device
    print(f"\n  Device : {DEVICE}")
    if DEVICE.type == "cpu":
        print("  WARNING: CPU-only — transfer learning will be slow.")
        print(f"  Input size {args.input_size}×{args.input_size} images are large.")
        print("  Tip: use --backbone resnet34 --batch-size 32 on Intel Mac.")
    elif DEVICE.type == "mps":
        print("  Apple Silicon MPS — GPU acceleration active.")
        expected_mins = (args.epochs_frozen + args.epochs_finetune) * 1.6
        print(f"  Expected time: ~{expected_mins:.0f}–{expected_mins*1.3:.0f} min "
              f"for {args.epochs_frozen}+{args.epochs_finetune} epochs.")

    # CIFAR-100 data
    cifar_path = Path("data") / "cifar-100-python"
    if not cifar_path.exists():
        raise FileNotFoundError(
            f"\n  CIFAR-100 data not found at: {cifar_path}\n"
            f"  Expected structure:\n"
            f"    data/cifar-100-python/train\n"
            f"    data/cifar-100-python/test\n"
            f"    data/cifar-100-python/meta\n\n"
            f"  Extract with:\n"
            f"    cd data && tar -xzf cifar-100-python.tar.gz"
        )
    print(f"\n  CIFAR-100 data : found at {cifar_path}")

    # Backbone validation
    if args.backbone not in SUPPORTED_BACKBONES:
        print(f"\n  WARNING: '{args.backbone}' is not in the curated list.")
        print("  Training will proceed but results may be unexpected.")
        print("  Run --list-backbones to see curated options.")
    else:
        meta = SUPPORTED_BACKBONES[args.backbone]
        if args.input_size != meta["input_size"]:
            print(f"\n  WARNING: {args.backbone} expects "
                  f"input_size={meta['input_size']} "
                  f"but --input-size={args.input_size} was given.")
            print(f"  Recommended: --input-size {meta['input_size']}")

    # Phase conflict
    if args.phase_a_only and args.phase_b_only:
        raise ValueError(
            "--phase-a-only and --phase-b-only cannot be used together."
        )

    # Phase B only requires Phase A checkpoint
    if args.phase_b_only and not args.checkpoint_a.exists():
        raise FileNotFoundError(
            f"\n  --phase-b-only requires a Phase A checkpoint at:\n"
            f"  {args.checkpoint_a}\n\n"
            f"  Run Phase A first:\n"
            f"    python run_transfer.py --phase-a-only"
        )

    # Dry run overrides
    if args.dry_run:
        print("\n  DRY RUN — overriding epochs to 2 per phase.")
        args.epochs_frozen   = 2
        args.epochs_finetune = 2

    print("\n  All pre-flight checks passed.\n")


# ── Configuration summary ──────────────────────────────────────────────────────

def print_config(args: argparse.Namespace) -> None:
    """Print a complete configuration table for reproducibility."""
    print("═" * 62)
    print("  Training configuration — Transfer Learning")
    print("═" * 62)
    rows = [
        ("Backbone",              args.backbone),
        ("Input size",            f"{args.input_size}×{args.input_size}"),
        ("Pretrained",            not args.no_pretrained),
        ("─── Phase A ───",       ""),
        ("  Epochs (frozen)",     args.epochs_frozen if not args.phase_b_only else "skipped"),
        ("  LR head",             f"{args.lr_head:.2e}"),
        ("─── Phase B ───",       ""),
        ("  Epochs (finetune)",   args.epochs_finetune if not args.phase_a_only else "skipped"),
        ("  LR backbone",         f"{args.lr_backbone:.2e}"),
        ("  LR head",             f"{args.lr_head:.2e}"),
        ("─── Shared ───",        ""),
        ("  Batch size",          args.batch_size),
        ("  Weight decay",        f"{args.weight_decay:.2e}"),
        ("  Label smoothing",     args.label_smoothing),
        ("  LR schedule",         args.schedule),
        ("  Early stop patience", args.patience if args.patience > 0 else "disabled"),
        ("  Device",              str(DEVICE)),
        ("  Seed",                args.seed),
        ("Checkpoint A",          str(args.checkpoint_a)),
        ("Checkpoint final",      str(args.checkpoint)),
        ("Dry run",               args.dry_run),
    ]
    for label, value in rows:
        if str(value) == "":
            print(f"  {label}")
        else:
            print(f"  {label:<28} {value}")
    print("═" * 62 + "\n")


# ── LR schedule preview ────────────────────────────────────────────────────────

def plot_both_schedules(args: argparse.Namespace, model: torch.nn.Module) -> None:
    """
    Simulate and plot LR schedules for both phases side by side.
    Uses fresh optimiser/scheduler copies so real training state
    is not advanced by the simulation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Transfer learning LR schedules", fontsize=12)

    for ax, phase, epochs, lr, title in [
        (axes[0], "A", args.epochs_frozen,   args.lr_head,
         f"Phase A — head only  (lr={args.lr_head:.1e}, {args.epochs_frozen} epochs)"),
        (axes[1], "B", args.epochs_finetune, args.lr_head,
         f"Phase B — full fine-tune  (lr_head={args.lr_head:.1e}, "
         f"lr_backbone={args.lr_backbone:.1e}, {args.epochs_finetune} epochs)"),
    ]:
        opt_sim = torch.optim.AdamW(model.parameters(), lr=lr)
        sch_sim = make_scheduler(opt_sim, args.schedule, epochs)
        lrs_sim = []
        for _ in range(epochs):
            lrs_sim.append(opt_sim.param_groups[0]["lr"])
            sch_sim.step()

        ax.plot(range(1, epochs + 1), lrs_sim, lw=2, color="#534AB7")
        ax.fill_between(range(1, epochs + 1), lrs_sim, alpha=0.12, color="#534AB7")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Learning rate", fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = PLOT_DIR / "transfer_lr_schedules.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[run_transfer] LR schedule saved → {path}")
    plt.show()


# ── History merge ─────────────────────────────────────────────────────────────

def merge_histories(history_a: dict, history_b: dict) -> dict:
    """
    Concatenate Phase A and Phase B history lists into a single dict.
    List-valued keys are concatenated; non-list keys take Phase B's value.

    Args:
        history_a: History returned by run_training() for Phase A.
        history_b: History returned by run_training() for Phase B.

    Returns:
        Combined history dict spanning all epochs.
    """
    merged = {}
    for key in history_a:
        val_a = history_a[key]
        val_b = history_b.get(key, [])
        if isinstance(val_a, list) and isinstance(val_b, list):
            merged[key] = val_a + val_b
        else:
            merged[key] = val_b if val_b else val_a

    # Mark the phase boundary so plotting scripts can draw a divider
    merged["phase_a_epochs"] = len(history_a.get("val_acc", []))
    merged["phase_b_epochs"] = len(history_b.get("val_acc", []))

    return merged


# ── Test evaluation helper ────────────────────────────────────────────────────

def quick_test_eval(
    model       : torch.nn.Module,
    test_loader : torch.utils.data.DataLoader,
    checkpoint  : Path,
    label       : str = "model",
) -> dict:
    """
    Load the best checkpoint and evaluate top-1 and top-5 on the test set.

    Args:
        model:       Architecture instance (state dict will be overwritten).
        test_loader: Test DataLoader with no augmentation.
        checkpoint:  Path to the best .pth checkpoint to load.
        label:       Short name printed in the log.

    Returns:
        dict with keys: top1, top5, correct_1, correct_5, total.
    """
    print(f"\n  Loading best checkpoint: {checkpoint.name}")
    load_checkpoint(model, checkpoint)
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
            top5_indices = logits.topk(5, dim=1).indices
            correct_5   += sum(
                labels[i].item() in top5_indices[i].tolist()
                for i in range(len(labels))
            )
            total += len(labels)

    top1 = correct_1 / total
    top5 = correct_5 / total

    print(f"  [{label}] Test top-1 : {top1:.4f}  ({top1*100:.2f}%)")
    print(f"  [{label}] Test top-5 : {top5:.4f}  ({top5*100:.2f}%)")

    return {
        "top1"      : top1,
        "top5"      : top5,
        "correct_1" : correct_1,
        "correct_5" : correct_5,
        "total"     : total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── List backbones and exit ────────────────────────────────────────────
    if args.list_backbones:
        print_backbone_table()
        sys.exit(0)

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
        mode        = "transfer",
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        input_size  = args.input_size,
    )

    # ── Build model ────────────────────────────────────────────────────────
    print("\n── Building model ───────────────────────────────────────────")
    model = build_transfer_model(
        backbone    = args.backbone,
        num_classes = NUM_CLASSES,
        pretrained  = not args.no_pretrained,
    ).to(DEVICE)

    if not args.no_summary:
        model_summary(
            model,
            input_size = (1, 3, args.input_size, args.input_size),
            max_rows   = 30,
        )

    # ── LR schedule preview and exit ──────────────────────────────────────
    if args.plot_lr:
        print("\n── Plotting LR schedules (--plot-lr) ────────────────────")
        plot_both_schedules(args, model)
        print("Schedules saved. Exiting (remove --plot-lr to start training).")
        sys.exit(0)

    # ── Track overall wall time ────────────────────────────────────────────
    wall_start = time.perf_counter()

    # These will be populated by whichever phases run
    history_a: dict = {}
    history_b: dict = {}

    # ─────────────────────────────────────────────────────────────────────
    # PHASE A — head only (backbone frozen)
    # ─────────────────────────────────────────────────────────────────────
    if not args.phase_b_only:
        print("\n" + "═" * 62)
        print("  Phase A — Feature extraction (backbone frozen)")
        print("═" * 62)

        # Freeze backbone, only head is trainable
        freeze_backbone(model)
        count_parameters(model)

        # Optimiser — only passes trainable (head) params
        opt_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = args.lr_head,
            weight_decay = args.weight_decay,
            betas        = (0.9, 0.999),
        )
        print(f"\n  Optimiser : AdamW  lr={args.lr_head:.2e}  "
              f"wd={args.weight_decay:.2e}")

        sch_a = make_scheduler(
            optimizer = opt_a,
            schedule  = args.schedule,
            epochs    = args.epochs_frozen,
        )
        print(f"  Schedule  : {args.schedule}  "
              f"T_max={args.epochs_frozen} epochs")

        es_a = None
        if args.patience > 0:
            es_a = EarlyStopping(
                patience  = args.patience,
                min_delta = 1e-4,
                verbose   = True,
            )
            print(f"  Early stop: patience={args.patience}")

        history_a = run_training(
            model           = model,
            train_loader    = train_loader,
            val_loader      = val_loader,
            epochs          = args.epochs_frozen,
            optimizer       = opt_a,
            scheduler       = sch_a,
            label           = f"TL-A-{args.backbone}",
            checkpoint_path = args.checkpoint_a,
            early_stopping  = es_a,
            label_smoothing = args.label_smoothing,
        )

        best_a = max(history_a["val_acc"]) if history_a["val_acc"] else 0.0
        print(f"\n  Phase A complete — best val accuracy: {best_a:.4f}")

        # Reload best Phase A weights before entering Phase B
        # (in case the last epoch was not the best)
        if args.checkpoint_a.exists():
            load_checkpoint(model, args.checkpoint_a)
            print(f"  Best Phase A weights reloaded from {args.checkpoint_a.name}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE B — full fine-tuning (all layers unfrozen)
    # ─────────────────────────────────────────────────────────────────────
    if not args.phase_a_only:
        print("\n" + "═" * 62)
        print("  Phase B — Full fine-tuning (all layers unfrozen)")
        print("═" * 62)

        # If --phase-b-only, load the Phase A checkpoint first
        if args.phase_b_only:
            print(f"\n  Loading Phase A checkpoint: {args.checkpoint_a.name}")
            load_checkpoint(model, args.checkpoint_a)

        # Unfreeze every layer
        unfreeze_all(model)
        count_parameters(model)

        # Discriminative LR param groups
        param_groups = get_param_groups(
            model        = model,
            lr_backbone  = args.lr_backbone,
            lr_head      = args.lr_head,
            weight_decay = args.weight_decay,
        )

        opt_b = torch.optim.AdamW(param_groups)
        print(f"\n  Optimiser : AdamW  (discriminative LRs)")
        print(f"    backbone group → lr={args.lr_backbone:.1e}  "
              f"wd={args.weight_decay:.2e}")
        print(f"    head group     → lr={args.lr_head:.1e}  "
              f"wd={args.weight_decay:.2e}")

        sch_b = make_scheduler(
            optimizer = opt_b,
            schedule  = args.schedule,
            epochs    = args.epochs_finetune,
        )
        print(f"  Schedule  : {args.schedule}  "
              f"T_max={args.epochs_finetune} epochs")

        es_b = None
        if args.patience > 0:
            es_b = EarlyStopping(
                patience  = args.patience,
                min_delta = 1e-4,
                verbose   = True,
            )
            print(f"  Early stop: patience={args.patience}")

        history_b = run_training(
            model           = model,
            train_loader    = train_loader,
            val_loader      = val_loader,
            epochs          = args.epochs_finetune,
            optimizer       = opt_b,
            scheduler       = sch_b,
            label           = f"TL-B-{args.backbone}",
            checkpoint_path = args.checkpoint,
            early_stopping  = es_b,
            label_smoothing = args.label_smoothing,
        )

        best_b = max(history_b["val_acc"]) if history_b["val_acc"] else 0.0
        print(f"\n  Phase B complete — best val accuracy: {best_b:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Save history
    # ─────────────────────────────────────────────────────────────────────
    print("\n── Saving history ───────────────────────────────────────────")

    backbone_tag  = args.backbone.replace("/", "_")
    history_file  = f"transfer_{backbone_tag}_history.json"

    if history_a and history_b:
        # Both phases ran — merge into one continuous history
        combined = merge_histories(history_a, history_b)
        save_history(combined, history_file)
        print(f"  Merged Phase A ({combined['phase_a_epochs']} ep) + "
              f"Phase B ({combined['phase_b_epochs']} ep) history saved.")

    elif history_a:
        # Phase A only
        history_a["phase_a_epochs"] = len(history_a.get("val_acc", []))
        save_history(history_a, history_file)

    elif history_b:
        # Phase B only
        history_b["phase_b_epochs"] = len(history_b.get("val_acc", []))
        save_history(history_b, history_file)

    # ─────────────────────────────────────────────────────────────────────
    # Quick test-set evaluation
    # ─────────────────────────────────────────────────────────────────────
    print("\n── Quick test-set evaluation ────────────────────────────────")

    # Determine which checkpoint holds the final best weights
    final_checkpoint = (
        args.checkpoint   if (not args.phase_a_only and args.checkpoint.exists())
        else args.checkpoint_a
    )

    test_results = quick_test_eval(
        model       = model,
        test_loader = test_loader,
        checkpoint  = final_checkpoint,
        label       = f"TL-{args.backbone}",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    wall_total     = time.perf_counter() - wall_start
    all_val_accs   = (
        history_a.get("val_acc", []) + history_b.get("val_acc", [])
    )
    all_epoch_times = (
        history_a.get("epoch_time", []) + history_b.get("epoch_time", [])
    )
    best_val_acc   = max(all_val_accs)   if all_val_accs    else 0.0
    best_val_epoch = (all_val_accs.index(best_val_acc) + 1) if all_val_accs else 0
    avg_epoch_time = (sum(all_epoch_times) / len(all_epoch_times)
                      if all_epoch_times else 0.0)
    total_params   = sum(p.numel() for p in model.parameters())

    print("\n" + "═" * 62)
    print("  Training complete — final results")
    print("═" * 62)
    print(f"  Backbone           : {args.backbone}")
    print(f"  Input size         : {args.input_size}×{args.input_size}")
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Phase A epochs     : {len(history_a.get('val_acc', []))}"
          f"{'  (skipped)' if args.phase_b_only else ''}")
    print(f"  Phase B epochs     : {len(history_b.get('val_acc', []))}"
          f"{'  (skipped)' if args.phase_a_only else ''}")
    print(f"  Best val accuracy  : {best_val_acc:.4f}  (epoch {best_val_epoch})")
    print(f"  Test top-1 accuracy: {test_results['top1']:.4f}  "
          f"({test_results['top1']*100:.2f}%)")
    print(f"  Test top-5 accuracy: {test_results['top5']:.4f}  "
          f"({test_results['top5']*100:.2f}%)")
    print(f"  Total wall time    : {format_time(wall_total)}")
    print(f"  Avg epoch time     : {format_time(avg_epoch_time)}")
    print(f"  Phase A checkpoint : {args.checkpoint_a}")
    print(f"  Final checkpoint   : {final_checkpoint}")
    print(f"  History saved      : {RESULT_DIR / history_file}")
    print("═" * 62)

    # ── Next step hint ─────────────────────────────────────────────────────
    print("\n  Next steps:")
    print("    1. Compare both models : python compare.py")
    print("    2. Explore plots in    : outputs/plots/")
    print("    3. Check results in    : outputs/results/")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()