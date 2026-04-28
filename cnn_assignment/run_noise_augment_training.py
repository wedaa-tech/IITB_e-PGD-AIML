"""
run_noise_augment_training.py
──────────────────────────────
Train an improved model using noise-augmented training strategy.

This script retrains the scratch CNN (or transfer model) using the
NoisyAugmentConfig strategy from src/noise_augment.py.

The trained model is saved to a separate checkpoint so it can be
compared against the baseline in compare_robustness_improvement.py.

Usage:
    python run_noise_augment_training.py                  # scratch CNN
    python run_noise_augment_training.py --model transfer # transfer model
    python run_noise_augment_training.py --noise-prob 0.3 # override fraction
    python run_noise_augment_training.py --no-mixup       # disable Mixup
    python run_noise_augment_training.py --dry-run        # 2 epochs only
    python run_noise_augment_training.py --verify-dist    # verify distribution
"""

import argparse
import time
from pathlib import Path

import torch

from src.config import (
    DEVICE, SCRATCH, TRANSFER, NUM_CLASSES,
    RANDOM_SEED, RESULT_DIR, CKPT_DIR, PLOT_DIR,
)
from src.utils import (
    set_seed, check_outputs_dir, log_system_info,
    format_time, save_history,
)
from src.models.scratch_cnn import build_scratch_cnn, count_parameters
from src.models.transfer_model import (
    build_transfer_model, freeze_backbone, unfreeze_all,
    get_param_groups, count_parameters as tl_count,
)
from src.train import make_scheduler, EarlyStopping, load_checkpoint
from src.noise_augment import (
    NoisyAugmentConfig,
    save_augment_config,
    load_augment_config,
    BatchNoiseAugmenter,
    MixupCollator,
    NoisyAugmentDataLoader,
    SoftCrossEntropyLoss,
    build_noisy_train_loader,
    verify_distribution_constraint,
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train with noise augmentation strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="scratch",
                        choices=["scratch", "transfer"],
                        help="Which model to train with noise augmentation")
    parser.add_argument("--scratch-variant", default="standard",
                        choices=["standard", "small", "residual"])
    parser.add_argument("--backbone",        default=TRANSFER["backbone"])
    parser.add_argument("--input-size",      type=int, default=TRANSFER["input_size"])

    # Augmentation hyperparameters
    parser.add_argument("--noise-prob",   type=float, default=0.25,
                        help="Fraction of each batch that receives noise")
    parser.add_argument("--noise-variance", type=float, default=0.05,
                        help="Noise variance σ² (should match test noise)")
    parser.add_argument("--no-mixup",     action="store_true",
                        help="Disable Mixup augmentation")
    parser.add_argument("--mixup-alpha",  type=float, default=0.2,
                        help="Beta distribution parameter for Mixup")

    # Training
    parser.add_argument("--epochs",          type=int, default=50)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--batch-size",      type=int, default=128)
    parser.add_argument("--patience",        type=int, default=15)
    parser.add_argument("--schedule",        default="cosine",
                        choices=["cosine","step","warmup","plateau"])

    # Transfer-specific
    parser.add_argument("--epochs-frozen",   type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=25)
    parser.add_argument("--lr-backbone",     type=float, default=1e-5)

    # Misc
    parser.add_argument("--dry-run",      action="store_true",
                        help="2 epochs only — pipeline check")
    parser.add_argument("--verify-dist",  action="store_true",
                        help="Verify distribution constraint before training")
    parser.add_argument("--seed",         type=int, default=RANDOM_SEED)
    parser.add_argument("--no-summary",   action="store_true")

    return parser.parse_args()


# ── Custom training loop (handles soft Mixup labels) ─────────────────────────

def train_one_epoch_augmented(
    model     : torch.nn.Module,
    loader    : NoisyAugmentDataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : SoftCrossEntropyLoss,
) -> tuple[float, float]:
    """
    One training epoch with noise-augmented loader.
    Handles both hard labels (no Mixup) and soft labels (Mixup).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        # Accuracy: use argmax of soft labels for Mixup case
        if labels.dim() == 2:
            hard_labels = labels.argmax(dim=1)
        else:
            hard_labels = labels

        correct += (logits.argmax(dim=1) == hard_labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


def validate_clean(
    model     : torch.nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : SoftCrossEntropyLoss,
) -> tuple[float, float]:
    """Validate on clean images (hard labels). Always clean."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)   # hard labels here
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Main training function ────────────────────────────────────────────────────

def run_augmented_training(
    model           : torch.nn.Module,
    aug_loader      : NoisyAugmentDataLoader,
    val_loader      : torch.utils.data.DataLoader,
    epochs          : int,
    optimizer       : torch.optim.Optimizer,
    scheduler,
    config          : NoisyAugmentConfig,
    label           : str,
    checkpoint_path : Path,
    early_stopping  : EarlyStopping | None = None,
) -> dict:
    """Full training loop for noise-augmented training."""
    criterion = SoftCrossEntropyLoss(smoothing=0.1)
    best_val_acc = 0.0
    history = {
        "train_loss":[], "val_loss":[],
        "train_acc":[], "val_acc":[],
        "epoch_time":[], "lr":[],
        "noise_prob": config.noise_prob,
        "noise_variance": config.noise_variance,
        "use_mixup": config.use_mixup,
    }

    print(f"\n{'─'*62}")
    print(f"  Training [{label}]")
    print(f"  Strategy: {config.noise_prob:.0%} noisy + "
          f"{1-config.noise_prob:.0%} clean per batch")
    print(f"  Mixup: {'enabled α=' + str(config.mixup_alpha) if config.use_mixup else 'disabled'}")
    print(f"  Epochs: {epochs}  |  Device: {DEVICE}")
    print(f"{'─'*62}")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch_augmented(
            model, aug_loader, optimizer, criterion)
        vl_loss, vl_acc = validate_clean(
            model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        elapsed = time.perf_counter() - t0

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"]  .append(round(vl_loss, 6))
        history["train_acc"] .append(round(tr_acc,  6))
        history["val_acc"]   .append(round(vl_acc,  6))
        history["epoch_time"].append(round(elapsed,  2))
        history["lr"]        .append(round(current_lr, 8))

        improved = "  *" if vl_acc > best_val_acc else ""
        print(f"  [{label}] Ep {epoch:3d}/{epochs}  |  "
              f"TrainAcc {tr_acc:.4f}  ValAcc {vl_acc:.4f}  |  "
              f"Loss {vl_loss:.4f}  |  LR {current_lr:.1e}  |  "
              f"{elapsed:.1f}s{improved}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch"          : epoch,
                "val_acc"        : vl_acc,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history"        : history,
                "augment_config" : {
                    "noise_prob"    : config.noise_prob,
                    "noise_variance": config.noise_variance,
                    "use_mixup"     : config.use_mixup,
                    "mixup_alpha"   : config.mixup_alpha,
                },
            }, checkpoint_path)

        if early_stopping and early_stopping(vl_acc):
            break

    print(f"\n  [{label}] Best val accuracy: {best_val_acc:.4f}")
    print(f"  [{label}] Checkpoint: {checkpoint_path}")
    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    check_outputs_dir()
    set_seed(args.seed)
    log_system_info()

    if args.dry_run:
        args.epochs = 2
        args.epochs_frozen   = 2
        args.epochs_finetune = 2
        print("  DRY RUN — 2 epochs")

    # ── 1. Define augmentation config ─────────────────────────────────────
    print("\n── Step 1: Define and save augmentation config ──────────────")
    augment_config = NoisyAugmentConfig(
        noise_prob       = args.noise_prob,
        noise_variance   = args.noise_variance,
        use_mixup        = not args.no_mixup,
        mixup_alpha      = args.mixup_alpha,
        noise_apply_mode = "batch",
        normalisation    = "cifar100" if args.model == "scratch" else "imagenet",
        seed             = args.seed,
    )
    save_augment_config(augment_config,
                        filename=f"noise_augment_config_{args.model}.json")

    # ── 2. Build data loaders ──────────────────────────────────────────────
    print("\n── Step 2: Build noise-augmented training loader ────────────")

    mode       = "scratch" if args.model == "scratch" else "transfer"
    input_size = 32 if args.model == "scratch" else args.input_size

    # Build base loaders
    from src.dataset import get_dataloaders
    base_train, val_loader, test_loader = get_dataloaders(
        mode        = mode,
        batch_size  = args.batch_size,
        num_workers = 0,
        input_size  = input_size,
    )

    # Wrap train loader with Mixup collator if enabled
    if augment_config.use_mixup:
        print(f"  Rebuilding train loader with Mixup collator (α={args.mixup_alpha})")
        from torch.utils.data import DataLoader
        collator  = MixupCollator(augment_config)
        base_train_mx = DataLoader(
            base_train.dataset,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = False,
            collate_fn  = collator,
        )
    else:
        base_train_mx = base_train

    # Wrap with BatchNoiseAugmenter
    augmenter  = BatchNoiseAugmenter(augment_config)
    aug_loader = NoisyAugmentDataLoader(base_train_mx, augmenter)

    # ── 3. Verify distribution constraint ─────────────────────────────────
    if args.verify_dist:
        print("\n── Step 3: Verify distribution constraint ───────────────")
        verify_distribution_constraint(base_train, augment_config, n_batches=20)

    # ── 4. Build model ─────────────────────────────────────────────────────
    print("\n── Step 4: Build model ──────────────────────────────────────")

    if args.model == "scratch":
        # ── Scratch CNN ───────────────────────────────────────────────────
        ckpt_path = CKPT_DIR / f"scratch_noisy_augmented_best.pth"
        model     = build_scratch_cnn(
            variant=args.scratch_variant, num_classes=NUM_CLASSES).to(DEVICE)
        count_parameters(model)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        scheduler = make_scheduler(
            optimizer, args.schedule, args.epochs)
        es = EarlyStopping(patience=args.patience, verbose=True) \
             if args.patience > 0 else None

        print(f"\n── Step 5: Train scratch CNN with noise augmentation ────")
        history = run_augmented_training(
            model           = model,
            aug_loader      = aug_loader,
            val_loader      = val_loader,
            epochs          = args.epochs,
            optimizer       = optimizer,
            scheduler       = scheduler,
            config          = augment_config,
            label           = f"Scratch-NoisyAug (p={args.noise_prob})",
            checkpoint_path = ckpt_path,
            early_stopping  = es,
        )
        save_history(history, "scratch_noisy_augmented_history.json")

    else:
        # ── Transfer model ────────────────────────────────────────────────
        ckpt_path_a = CKPT_DIR / "transfer_noisy_aug_phase_a_best.pth"
        ckpt_path   = CKPT_DIR / "transfer_noisy_augmented_best.pth"

        model = build_transfer_model(
            backbone=args.backbone, num_classes=NUM_CLASSES,
            pretrained=True).to(DEVICE)

        # Phase A (frozen backbone)
        print(f"\n── Step 5a: Phase A — frozen backbone ───────────────────")
        freeze_backbone(model)
        opt_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
        sch_a = make_scheduler(opt_a, args.schedule, args.epochs_frozen)
        es_a  = EarlyStopping(patience=args.patience, verbose=True)

        hist_a = run_augmented_training(
            model, aug_loader, val_loader,
            epochs=args.epochs_frozen, optimizer=opt_a, scheduler=sch_a,
            config=augment_config,
            label=f"TL-NoisyAug-PhaseA (p={args.noise_prob})",
            checkpoint_path=ckpt_path_a, early_stopping=es_a)

        load_checkpoint(model, ckpt_path_a)

        # Phase B (full fine-tune)
        print(f"\n── Step 5b: Phase B — full fine-tuning ──────────────────")
        unfreeze_all(model)
        param_groups = get_param_groups(
            model, lr_backbone=args.lr_backbone, lr_head=args.lr)
        opt_b = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        sch_b = make_scheduler(opt_b, args.schedule, args.epochs_finetune)
        es_b  = EarlyStopping(patience=args.patience, verbose=True)

        hist_b = run_augmented_training(
            model, aug_loader, val_loader,
            epochs=args.epochs_finetune, optimizer=opt_b, scheduler=sch_b,
            config=augment_config,
            label=f"TL-NoisyAug-PhaseB (p={args.noise_prob})",
            checkpoint_path=ckpt_path, early_stopping=es_b)

        # Merge histories
        merged = {k: hist_a.get(k, []) + hist_b.get(k, [])
                  for k in hist_a if isinstance(hist_a[k], list)}
        merged.update({k: v for k, v in hist_a.items()
                       if not isinstance(v, list)})
        merged["phase_a_epochs"] = len(hist_a.get("val_acc", []))
        save_history(merged, "transfer_noisy_augmented_history.json")
        history = merged

    # ── 6. Final summary ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Noise-Augmented Training Complete — {args.model.upper()}")
    print("═" * 60)
    print(f"  Strategy        : {augment_config.noise_prob:.0%} noisy + "
          f"{augment_config.clean_fraction:.0%} clean per batch")
    print(f"  Noise σ²        : {augment_config.noise_variance}  "
          f"(σ={augment_config.sigma:.4f})")
    print(f"  Mixup           : {'enabled α=' + str(augment_config.mixup_alpha) if augment_config.use_mixup else 'disabled'}")
    print(f"  Best val acc    : {max(history['val_acc']):.4f}")
    print(f"  Epochs trained  : {len(history['val_acc'])}")
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"\n  Next step: python compare_robustness_improvement.py "
          f"--model {args.model}")
    print("═" * 60)


if __name__ == "__main__":
    main()