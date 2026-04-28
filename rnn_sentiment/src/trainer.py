"""
Shared training + evaluation engine.
Works identically for VanillaRNN, LSTMClassifier, AttentionLSTM.
"""

import time
import pickle
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict


from src.config      import DEVICE, CHECKPOINT_DIR, NUM_EPOCHS, LEARNING_RATE
from src.metrics     import MetricTracker, binary_accuracy
from src.utils       import set_seed

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tracker = MetricTracker()

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds, _ = binary_accuracy(logits.detach(), y)
        tracker.update(loss.item(), preds, y.long())

    return tracker.avg_loss, tracker.accuracy


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()

    for X, y in loader:
        X, y  = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        preds, _ = binary_accuracy(logits, y)
        tracker.update(loss.item(), preds, y.long())

    return tracker.avg_loss, tracker.accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_name:   str,
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_epochs:   int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
    patience:     int   = 3,            # early stopping
) -> dict:
    """
    Full training loop with:
      - BCEWithLogitsLoss  (numerically stable, no sigmoid in forward needed)
      - Adam optimiser
      - ReduceLROnPlateau scheduler
      - Early stopping on val loss
      - Best model checkpoint saved automatically
    
    Returns history dict with train/val loss and accuracy per epoch.
    """
    set_seed()
    model     = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Note: the `verbose=` kwarg was removed from ReduceLROnPlateau in recent
    # PyTorch releases (2.4+ deprecated it, later versions deleted it). We log
    # LR reductions manually below after each scheduler.step() instead.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5,
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path      = CHECKPOINT_DIR / f"{model_name}_best.pt"

    log.info(f"\n{'='*55}")
    log.info(f"  Training  : {model_name.upper()}")
    log.info(f"  Device    : {DEVICE}")
    log.info(f"  Params    : {_count_params(model):,}")
    log.info(f"{'='*55}")
    _print_header()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = _eval_epoch(model, val_loader,   criterion, DEVICE)

        # Step the scheduler and log if it actually dropped the LR (this
        # replaces the old `verbose=True` behaviour that was removed upstream).
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vl_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            log.info(f"  [lr] ReduceLROnPlateau: {prev_lr:.2e} → {new_lr:.2e}")

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        improved = vl_loss < best_val_loss
        if improved:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = " ✓"
        else:
            patience_count += 1
            flag = f"  (no improvement {patience_count}/{patience})"

        _print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc, elapsed, flag)

        if patience_count >= patience:
            log.info(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    log.info(f"\n  Best val loss : {best_val_loss:.4f}")
    log.info(f"  Checkpoint    : {ckpt_path}")

    # Save history
    hist_path = CHECKPOINT_DIR / f"{model_name}_history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on test set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model_name: str, model: nn.Module, test_loader: DataLoader) -> dict:
    """
    Load the best checkpoint and evaluate on the test set.
    Returns dict with loss, accuracy, all predictions and labels.
    """
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model     = model.to(DEVICE)
    model.eval()

    criterion  = nn.BCEWithLogitsLoss()
    all_preds, all_labels = [], []
    tracker    = MetricTracker()

    for X, y in test_loader:
        X, y   = X.to(DEVICE), y.to(DEVICE)
        logits  = model(X)
        loss    = criterion(logits, y)
        preds, _ = binary_accuracy(logits, y)
        tracker.update(loss.item(), preds, y.long())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.long().cpu().numpy())

    log.info(f"\n  [{model_name.upper()}] Test  →  "
             f"loss={tracker.avg_loss:.4f}  acc={tracker.accuracy:.4f}")

    return {
        "loss":     tracker.avg_loss,
        "accuracy": tracker.accuracy,
        "preds":    all_preds,
        "labels":   all_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _print_header():
    log.info(f"  {'Epoch':>5} │ {'Tr Loss':>8} {'Tr Acc':>8} │ "
             f"{'Vl Loss':>8} {'Vl Acc':>8} │ {'Time':>6}")
    log.info("  " + "─" * 60)


def _print_row(epoch, tr_l, tr_a, vl_l, vl_a, t, flag):
    log.info(f"  {epoch:>5} │ {tr_l:>8.4f} {tr_a:>7.2%} │ "
             f"{vl_l:>8.4f} {vl_a:>7.2%} │ {t:>5.1f}s{flag}")



def train_with_grad_tracking(
    model_name:   str,
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_epochs:   int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
    patience:     int   = 3,
) -> tuple[dict, dict]:
    """
    Same as train() but additionally records per-layer gradient norms
    at each epoch. Used to visualise vanishing gradients in RNN vs LSTM.

    Returns:
        history      : same dict as train()
        grad_history : { layer_name : [norm_epoch1, norm_epoch2, ...] }
    """
    set_seed()
    model     = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    grad_history = defaultdict(list)   # layer_name → [norms per epoch]

    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path      = CHECKPOINT_DIR / f"{model_name}_best.pt"

    log.info(f"\n{'='*55}")
    log.info(f"  Training (grad tracking) : {model_name.upper()}")
    log.info(f"  Params : {_count_params(model):,}")
    log.info(f"{'='*55}")
    _print_header()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        tracker = MetricTracker()

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            preds, _ = binary_accuracy(logits.detach(), y)
            tracker.update(loss.item(), preds, y.long())

        # ── Record gradient norms for each named parameter ────────────────
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm(2).item()
                grad_history[name].append(norm)

        tr_loss, tr_acc = tracker.avg_loss, tracker.accuracy
        vl_loss, vl_acc = _eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(vl_loss)

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        improved = vl_loss < best_val_loss
        if improved:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = " ✓"
        else:
            patience_count += 1
            flag = f"  (no improvement {patience_count}/{patience})"

        _print_row(epoch, tr_loss, tr_acc, vl_loss, vl_acc, elapsed, flag)

        if patience_count >= patience:
            log.info(f"\n  Early stopping at epoch {epoch}.")
            break

    hist_path = CHECKPOINT_DIR / f"{model_name}_history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    return history, dict(grad_history)