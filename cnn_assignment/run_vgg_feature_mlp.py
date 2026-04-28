"""
run_vgg_feature_mlp.py
───────────────────────
Complete VGG feature extraction + MLP training pipeline.

Stages:
    1. Build frozen VGG16-BN extractor
    2. Extract features for train/val/test → cache to disk (.npy)
    3. Train small MLP on cached features
    4. Evaluate on clean test set (top-1, top-5, loss)
    5. Compare with scratch CNN and EfficientNet results
    6. Generate comparison plots and report

Usage:
    python run_vgg_feature_mlp.py                    # full pipeline
    python run_vgg_feature_mlp.py --skip-extraction  # use cached features
    python run_vgg_feature_mlp.py --dry-run          # 2 epochs smoke test
    python run_vgg_feature_mlp.py --vgg vgg11_bn     # lighter backbone
    python run_vgg_feature_mlp.py --no-plots         # metrics only
    python run_vgg_feature_mlp.py --hidden 1024 512  # wider MLP

    PYTORCH_ENABLE_MPS_FALLBACK=1 python run_vgg_feature_mlp.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import (
    DEVICE, NUM_CLASSES, SCRATCH, TRANSFER,
    RESULT_DIR, PLOT_DIR, CKPT_DIR, DATA_DIR,
)
from src.utils import (
    set_seed, check_outputs_dir, log_system_info,
    format_time, save_history, load_history,
)
from src.dataset import get_dataloaders
from src.models.vgg_extractor import (
    VGGFeatureExtractor,
    MLPClassifier,
    VGGWithMLP,
    extract_and_cache,
    load_cached_features,
    count_parameters,
)


# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_DIR   = DATA_DIR / "vgg_feature_cache"
MLP_CKPT    = CKPT_DIR / "vgg_mlp_best.pth"
MLP_HISTORY = "vgg_mlp_history.json"


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VGG feature extraction + MLP classifier for CIFAR-100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vgg", default="vgg16_bn",
                        choices=["vgg11_bn","vgg13_bn","vgg16_bn","vgg19_bn"],
                        help="VGG variant to use as feature extractor")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256],
                        help="MLP hidden layer sizes e.g. --hidden 512 256")
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.5, 0.3],
                        help="Dropout rates for each hidden layer")

    parser.add_argument("--epochs",       type=int,   default=60)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size",   type=int,   default=512,
                        help="Large batches are fine — features fit in RAM")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience",     type=int,   default=15)

    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction if .npy cache already exists")
    parser.add_argument("--dry-run",         action="store_true",
                        help="2 epochs only — pipeline smoke test")
    parser.add_argument("--no-plots",        action="store_true")
    parser.add_argument("--seed",            type=int, default=42)
    return parser.parse_args()


# ── MLP training loop ─────────────────────────────────────────────────────────

def train_mlp(
    mlp          : MLPClassifier,
    train_loader : torch.utils.data.DataLoader,
    val_loader   : torch.utils.data.DataLoader,
    epochs       : int,
    lr           : float,
    weight_decay : float,
    label_smooth : float,
    patience     : int,
) -> dict:
    """
    Train the MLP classifier on cached VGG features.

    The training loop is simpler than the CNN loop:
    - No image augmentation (features are fixed)
    - No Mixup (labels are always hard integers)
    - Larger batch sizes (512+) are safe because 512-dim features
      are tiny compared to 224×224 images
    - Many more epochs (60+) are possible because each epoch is
      very fast (no backbone forward pass)

    Returns:
        History dict with train_loss, val_loss, train_acc, val_acc,
        epoch_time, lr per epoch.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc  = 0.0
    no_improve    = 0
    history = {
        "train_loss":[], "val_loss":[],
        "train_acc":[], "val_acc":[],
        "epoch_time":[], "lr":[],
    }

    print(f"\n{'─'*60}")
    print(f"  Training MLP on VGG features")
    print(f"  Epochs: {epochs}  |  LR: {lr:.1e}  |  "
          f"Batch: {train_loader.batch_size}")
    print(f"  Device: {DEVICE}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # ── Train ────────────────────────────────────────────────────────
        mlp.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = mlp(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss    += loss.item() * feats.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += feats.size(0)

        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────
        mlp.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                logits = mlp(feats)
                loss   = criterion(logits, labels)
                vl_loss    += loss.item() * feats.size(0)
                vl_correct += (logits.argmax(1) == labels).sum().item()
                vl_total   += feats.size(0)

        tr_acc = tr_correct / tr_total
        vl_acc = vl_correct / vl_total
        tr_l   = tr_loss    / tr_total
        vl_l   = vl_loss    / vl_total
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(round(tr_l, 6))
        history["val_loss"]  .append(round(vl_l, 6))
        history["train_acc"] .append(round(tr_acc, 6))
        history["val_acc"]   .append(round(vl_acc, 6))
        history["epoch_time"].append(round(elapsed, 2))
        history["lr"]        .append(round(lr_now, 8))

        improved = "  *" if vl_acc > best_val_acc else ""
        print(f"  Ep {epoch:3d}/{epochs}  |  "
              f"TrainAcc {tr_acc:.4f}  ValAcc {vl_acc:.4f}  |  "
              f"Loss {vl_l:.4f}  |  LR {lr_now:.1e}  |  "
              f"{elapsed:.2f}s{improved}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({
                "epoch"       : epoch,
                "val_acc"     : vl_acc,
                "model_state" : mlp.state_dict(),
            }, MLP_CKPT)
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"\n  Early stopping — no improvement for {patience} epochs.")
            break

    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Checkpoint        : {MLP_CKPT}")
    return history


# ── Test evaluation ───────────────────────────────────────────────────────────

def evaluate_mlp(
    mlp          : MLPClassifier,
    test_loader  : torch.utils.data.DataLoader,
    top_k        : int = 5,
) -> dict:
    """Evaluate MLP on cached test features."""
    mlp.eval()
    criterion = nn.CrossEntropyLoss()

    t1, t5, loss_sum, total = 0, 0, 0.0, 0
    all_preds, all_labels   = [], []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = mlp(feats)
            preds  = logits.argmax(dim=1)

            t1       += (preds == labels).sum().item()
            topk_idx  = logits.topk(top_k, dim=1).indices
            t5       += sum(labels[i].item() in topk_idx[i].tolist()
                            for i in range(len(labels)))
            loss_sum += criterion(logits, labels).item() * feats.size(0)
            total    += feats.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "top1"  : t1 / total,
        "top5"  : t5 / total,
        "loss"  : loss_sum / total,
        "preds" : np.array(all_preds),
        "labels": np.array(all_labels),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_mlp_learning_curves(history: dict) -> None:
    """Plot MLP training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("VGG feature extractor + MLP — learning curves", fontsize=11)

    epochs = range(1, len(history["val_acc"]) + 1)

    axes[0].plot(epochs, history["train_acc"], label="Train", color="#534AB7")
    axes[0].plot(epochs, history["val_acc"],   label="Val",   color="#1D9E75", lw=2)
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0))
    axes[0].legend(); axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_loss"], label="Train", color="#534AB7")
    axes[1].plot(epochs, history["val_loss"],   label="Val",   color="#1D9E75", lw=2)
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    plt.tight_layout()
    path = PLOT_DIR / "vgg_mlp_learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plot] Learning curves → {path}")
    plt.show(); plt.close()


def plot_three_way_comparison(
    results: dict[str, dict],
) -> None:
    """
    Four-panel bar chart comparing all three approaches:
        Scratch CNN | VGG + MLP | EfficientNet fine-tuned

    Panels: top-1, top-5, parameters (M), avg epoch time (s).
    """
    labels   = list(results.keys())
    colours  = ["#534AB7", "#D85A30", "#1D9E75"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(
        "Three-way comparison: Scratch CNN  vs  VGG+MLP  vs  EfficientNet",
        fontsize=11, y=1.02,
    )

    metrics = [
        ("top1",           "Top-1 accuracy",    "{:.1%}",  True),
        ("top5",           "Top-5 accuracy",    "{:.1%}",  True),
        ("params_m",       "Parameters (M)",    "{:.1f}M", False),
        ("avg_epoch_s",    "Avg epoch time (s)","  {:.0f}s", False),
    ]

    for ax, (key, title, fmt, pct) in zip(axes, metrics):
        vals  = [results[lbl].get(key, 0) for lbl in labels]
        clrs  = [colours[i % len(colours)] for i in range(len(labels))]
        bars  = ax.bar(labels, vals, color=clrs, alpha=0.85, width=0.5)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                fmt.format(val), ha="center", va="bottom", fontsize=8.5,
            )
        if pct:
            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.3 + 0.02)
        ax.tick_params(axis="x", labelsize=8, rotation=12)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    plt.tight_layout()
    path = PLOT_DIR / "three_way_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plot] Three-way comparison → {path}")
    plt.show(); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    check_outputs_dir()
    set_seed(args.seed)
    log_system_info()

    if args.dry_run:
        args.epochs = 2
        print("  DRY RUN — 2 epochs\n")

    wall_start = time.perf_counter()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1 — Build VGG feature extractor
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print(f"  Stage 1: Build {args.vgg.upper()} feature extractor")
    print("═"*60)

    extractor = VGGFeatureExtractor(
        pretrained  = True,
        vgg_variant = args.vgg,
    ).to(DEVICE)
    extractor.eval()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2 — Load image data (224×224, ImageNet normalisation)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 2: Load CIFAR-100 data (upscaled to 224×224)")
    print("═"*60)
    print("  Note: ImageNet normalisation used — required by VGG pretrained weights")

    # Use transfer mode = 224×224 with ImageNet norm
    # These loaders are used ONLY for feature extraction
    train_img_loader, val_img_loader, test_img_loader = get_dataloaders(
        mode        = "transfer",
        batch_size  = 64,           # smaller batch — full images in RAM
        num_workers = 0,
        input_size  = 224,
    )

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3 — Extract features and cache to disk
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 3: Extract VGG features → cache to disk")
    print("═"*60)
    print(f"  Cache directory: {CACHE_DIR}")
    print("  This runs once. Subsequent runs load from cache.\n")

    if args.skip_extraction and (CACHE_DIR / "vgg_features_train.npy").exists():
        print("  --skip-extraction: using existing cache")
    else:
        img_loaders = {
            "train": train_img_loader,
            "val"  : val_img_loader,
            "test" : test_img_loader,
        }
        t_extract = time.perf_counter()
        extract_and_cache(extractor, img_loaders, CACHE_DIR)
        print(f"\n  Feature extraction complete in "
              f"{format_time(time.perf_counter() - t_extract)}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 4 — Load cached features into fast DataLoaders
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 4: Load cached features into DataLoaders")
    print("═"*60)
    print(f"  Batch size: {args.batch_size}  "
          f"(large batch fine — features are 512 floats, not images)")

    feat_loaders = load_cached_features(
        cache_dir  = CACHE_DIR,
        batch_size = args.batch_size,
    )
    train_feat_loader = feat_loaders["train"]
    val_feat_loader   = feat_loaders["val"]
    test_feat_loader  = feat_loaders["test"]

    # ══════════════════════════════════════════════════════════════════
    # STAGE 5 — Build MLP classifier
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 5: Build MLP classifier")
    print("═"*60)

    assert len(args.hidden) == len(args.dropout), \
        "Must provide same number of --hidden and --dropout values"

    mlp = MLPClassifier(
        input_dim    = extractor.feature_dim,    # 512
        hidden_dims  = args.hidden,              # [512, 256]
        num_classes  = NUM_CLASSES,              # 100
        dropout_rates= args.dropout,             # [0.5, 0.3]
    ).to(DEVICE)

    param_counts = count_parameters(extractor, mlp)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 6 — Train MLP on cached features
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 6: Train MLP on cached VGG features")
    print("═"*60)
    print("  VGG backbone: FROZEN — no gradients flow through it")
    print("  Only MLP weights are updated\n")

    history = train_mlp(
        mlp          = mlp,
        train_loader = train_feat_loader,
        val_loader   = val_feat_loader,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        label_smooth = args.label_smoothing,
        patience     = args.patience,
    )
    save_history(history, MLP_HISTORY)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 7 — Evaluate on test set
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 7: Evaluate on test set")
    print("═"*60)

    # Load best MLP weights
    ckpt     = torch.load(MLP_CKPT, map_location=DEVICE)
    mlp.load_state_dict(ckpt["model_state"])
    print(f"  Best checkpoint (val_acc={ckpt['val_acc']:.4f}) "
          f"from epoch {ckpt['epoch']} loaded")

    test_results = evaluate_mlp(mlp, test_feat_loader, top_k=5)

    print(f"\n  VGG-MLP test results:")
    print(f"    Top-1 accuracy : {test_results['top1']:.4f}  "
          f"({test_results['top1']*100:.2f}%)")
    print(f"    Top-5 accuracy : {test_results['top5']:.4f}  "
          f"({test_results['top5']*100:.2f}%)")
    print(f"    Test loss      : {test_results['loss']:.4f}")

    # Save results
    results_path = RESULT_DIR / "vgg_mlp_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "top1"            : float(test_results["top1"]),
            "top5"            : float(test_results["top5"]),
            "loss"            : float(test_results["loss"]),
            "mlp_params"      : param_counts["mlp_trainable"],
            "vgg_frozen_params": param_counts["vgg_frozen"],
            "total_params"    : param_counts["total"],
            "best_val_acc"    : float(ckpt["val_acc"]),
            "best_epoch"      : int(ckpt["epoch"]),
            "vgg_variant"     : args.vgg,
            "mlp_hidden"      : args.hidden,
        }, f, indent=2)
    print(f"  Results saved → {results_path}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 8 — Three-way comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Stage 8: Three-way comparison")
    print("═"*60)

    # Load baseline results from previous runs
    def _load_result(path: Path, top1_fallback: float = 0.0) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"  Note: {path.name} not found — using placeholder.")
            return {"top1": top1_fallback, "top5": 0.0}

    scratch_res  = _load_result(RESULT_DIR / "scratch_eval.json",  0.58)
    transfer_res = _load_result(RESULT_DIR / "transfer_eval.json", 0.76)

    # Load epoch times from histories
    def _avg_epoch_time(hist_file: str) -> float:
        try:
            h = load_history(hist_file)
            return float(np.mean(h.get("epoch_time", [0])))
        except Exception:
            return 0.0

    scratch_epoch_t  = _avg_epoch_time("scratch_standard_history.json")
    transfer_epoch_t = _avg_epoch_time("transfer_efficientnet_b0_history.json")
    mlp_epoch_t      = float(np.mean(history.get("epoch_time", [1])))

    comparison = {
        "Scratch CNN": {
            "top1"        : scratch_res.get("top1", 0.58),
            "top5"        : scratch_res.get("top5", 0.83),
            "params_m"    : 9.22,
            "avg_epoch_s" : scratch_epoch_t,
            "description" : "4-block VGG-style CNN, all weights trained from random init",
        },
        "VGG16-BN + MLP": {
            "top1"        : test_results["top1"],
            "top5"        : test_results["top5"],
            "params_m"    : param_counts["mlp_trainable"] / 1e6,
            "avg_epoch_s" : mlp_epoch_t,
            "description" : "Frozen VGG16-BN features + small MLP classifier",
        },
        "EfficientNet-B0": {
            "top1"        : transfer_res.get("top1", 0.76),
            "top5"        : transfer_res.get("top5", 0.93),
            "params_m"    : 5.29,
            "avg_epoch_s" : transfer_epoch_t,
            "description" : "Pretrained EfficientNet-B0 fine-tuned (Phase A+B)",
        },
    }

    # Print table
    print(f"\n  {'Method':<22} {'Top-1':>8} {'Top-5':>8} "
          f"{'Params':>10} {'Epoch(s)':>10}")
    print("  " + "─" * 62)
    for name, r in comparison.items():
        print(f"  {name:<22} {r['top1']:>7.2%} {r['top5']:>7.2%} "
              f"{r['params_m']:>8.2f}M {r['avg_epoch_s']:>9.1f}s")

    # Save comparison JSON
    comp_path = RESULT_DIR / "three_way_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Three-way comparison saved → {comp_path}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 9 — Plots
    # ══════════════════════════════════════════════════════════════════
    if not args.no_plots:
        print("\n" + "═"*60)
        print("  Stage 9: Generate plots")
        print("═"*60)
        plot_mlp_learning_curves(history)
        plot_three_way_comparison(comparison)

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    wall_total = time.perf_counter() - wall_start

    print("\n" + "═"*60)
    print("  VGG Feature Extraction + MLP — Final Summary")
    print("═"*60)
    print(f"  VGG backbone        : {args.vgg.upper()}  (ALL layers frozen)")
    print(f"  Feature dimension   : {extractor.feature_dim}")
    print(f"  MLP architecture    : {extractor.feature_dim} → "
          f"{' → '.join(str(h) for h in args.hidden)} → {NUM_CLASSES}")
    print(f"  VGG params (frozen) : {param_counts['vgg_frozen']:,}")
    print(f"  MLP params (trained): {param_counts['mlp_trainable']:,}")
    print(f"  Epochs trained      : {len(history['val_acc'])}")
    print(f"  Best val accuracy   : {max(history['val_acc']):.4f}")
    print(f"  Test top-1 accuracy : {test_results['top1']:.4f}  "
          f"({test_results['top1']*100:.2f}%)")
    print(f"  Test top-5 accuracy : {test_results['top5']:.4f}  "
          f"({test_results['top5']*100:.2f}%)")
    print(f"  Avg MLP epoch time  : {mlp_epoch_t:.2f}s  "
          f"(very fast — no backbone forward pass)")
    print(f"  Total wall time     : {format_time(wall_total)}")
    print()
    print("  Key finding:")
    gap_vs_scratch = test_results["top1"] - comparison["Scratch CNN"]["top1"]
    gap_vs_tl      = test_results["top1"] - comparison["EfficientNet-B0"]["top1"]
    print(f"    vs Scratch CNN     : {gap_vs_scratch:+.2%}")
    print(f"    vs EfficientNet-B0 : {gap_vs_tl:+.2%}")
    print()
    print("  Saved files:")
    print(f"    {CACHE_DIR}/vgg_features_*.npy  (feature cache)")
    print(f"    {MLP_CKPT}")
    print(f"    {RESULT_DIR / MLP_HISTORY}")
    print(f"    {results_path}")
    print(f"    {comp_path}")
    if not args.no_plots:
        print(f"    {PLOT_DIR}/vgg_mlp_learning_curves.png")
        print(f"    {PLOT_DIR}/three_way_comparison.png")
    print("═"*60)


if __name__ == "__main__":
    main()