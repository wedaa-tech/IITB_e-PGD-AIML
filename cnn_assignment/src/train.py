"""
src/train.py
────────────
Training loop, validation, checkpointing, and learning-rate scheduling
for both the scratch CNN and the transfer learning model.

Exports:
    train_one_epoch()   → runs one full pass over the training loader
    validate()          → evaluates loss and accuracy on any loader
    run_training()      → full training run with logging and checkpointing
    load_checkpoint()   → restore a saved model + metadata
    EarlyStopping       → stops training when val accuracy plateaus
"""

import time
import json
import copy
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, RESULT_DIR


# ── Early stopping ─────────────────────────────────────────────────────────────

@dataclass
class EarlyStopping:
    """
    Stops training early when validation accuracy stops improving.

    Args:
        patience:   Number of epochs to wait after last improvement.
        min_delta:  Minimum change in val_acc to count as improvement.
        verbose:    Print a message each time patience counter increments.

    Usage:
        es = EarlyStopping(patience=10)
        for epoch in ...:
            val_acc = validate(...)
            if es(val_acc):
                break
    """
    patience  : int   = 10
    min_delta : float = 1e-4
    verbose   : bool  = True

    _best     : float = field(default=0.0,  init=False, repr=False)
    _counter  : int   = field(default=0,    init=False, repr=False)
    triggered : bool  = field(default=False, init=False, repr=False)

    def __call__(self, val_acc: float) -> bool:
        """Return True when training should stop."""
        if val_acc > self._best + self.min_delta:
            self._best   = val_acc
            self._counter = 0
        else:
            self._counter += 1
            if self.verbose:
                print(f"  [early_stop] No improvement for {self._counter}"
                      f"/{self.patience} epochs  (best={self._best:.4f})")
            if self._counter >= self.patience:
                self.triggered = True
                print(f"  [early_stop] Triggered — stopping training.")
                return True
        return False


# ── Single epoch helpers ───────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    scaler    : torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Args:
        model:     The network being trained (already on DEVICE).
        loader:    Training DataLoader.
        optimizer: Optimiser (AdamW, SGD, etc.).
        criterion: Loss function (CrossEntropyLoss with label smoothing).
        scaler:    GradScaler for mixed-precision training.
                   Pass None on CPU or MPS (only supported on CUDA).

    Returns:
        (avg_loss, accuracy) over the full epoch.
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=False)
        labels = labels.to(DEVICE, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        if scaler is not None:
            # Mixed precision — CUDA only
            with torch.autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision — MPS and CPU
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size    = imgs.size(0)
        running_loss += loss.item() * batch_size
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += batch_size

    avg_loss = running_loss / total
    accuracy = correct     / total
    return avg_loss, accuracy


def validate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
) -> tuple[float, float]:
    """
    Evaluate loss and top-1 accuracy on a loader without updating weights.

    Args:
        model:     Network to evaluate (already on DEVICE).
        loader:    Validation or test DataLoader.
        criterion: Same loss function used during training.

    Returns:
        (avg_loss, accuracy) over the full loader.
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=False)
            labels = labels.to(DEVICE, non_blocking=False)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            batch_size    = imgs.size(0)
            running_loss += loss.item() * batch_size
            correct      += (logits.argmax(dim=1) == labels).sum().item()
            total        += batch_size

    avg_loss = running_loss / total
    accuracy = correct     / total
    return avg_loss, accuracy


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    model      : nn.Module,
    optimizer  : torch.optim.Optimizer,
    epoch      : int,
    val_acc    : float,
    history    : dict,
    path       : Path,
) -> None:
    """
    Save model weights, optimiser state, epoch number, best val_acc,
    and training history to a single .pth file.

    Saving the optimiser state means you can resume training exactly
    where it left off if the run is interrupted.
    """
    torch.save({
        "epoch"          : epoch,
        "val_acc"        : val_acc,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history"        : history,
    }, path)
    print(f"  [ckpt] Saved best model  val_acc={val_acc:.4f}  →  {path.name}")


def load_checkpoint(
    model : nn.Module,
    path  : Path,
    optimizer : torch.optim.Optimizer | None = None,
) -> dict:
    """
    Load a checkpoint saved by save_checkpoint().

    Args:
        model:     Instantiated model (architecture must match the checkpoint).
        path:      Path to the .pth file.
        optimizer: If provided, the optimiser state is also restored
                   (useful for resuming training). Pass None for inference only.

    Returns:
        The full checkpoint dict so callers can read epoch / val_acc / history.
    """
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[train] Checkpoint loaded  epoch={ckpt['epoch']}"
          f"  val_acc={ckpt['val_acc']:.4f}  ←  {path.name}")
    return ckpt


# ── Main training loop ────────────────────────────────────────────────────────

def run_training(
    model           : nn.Module,
    train_loader    : DataLoader,
    val_loader      : DataLoader,
    epochs          : int,
    optimizer       : torch.optim.Optimizer,
    scheduler       : object | None             = None,
    label           : str                       = "model",
    checkpoint_path : Path | None               = None,
    early_stopping  : EarlyStopping | None      = None,
    label_smoothing : float                     = 0.1,
    resume_from     : Path | None               = None,
) -> dict:
    """
    Full training run with per-epoch logging, best-model checkpointing,
    optional LR scheduling, and optional early stopping.

    Args:
        model:           Network to train (already moved to DEVICE).
        train_loader:    Augmented training DataLoader.
        val_loader:      Clean validation DataLoader (no augmentation).
        epochs:          Maximum number of epochs to run.
        optimizer:       Configured optimiser (AdamW etc.).
        scheduler:       LR scheduler called once per epoch (e.g. CosineAnnealingLR).
                         Pass None to keep a fixed LR.
        label:           Short string printed in log lines for identification.
        checkpoint_path: Where to save the best model. None = no saving.
        early_stopping:  EarlyStopping instance. None = train for full epochs.
        label_smoothing: Label smoothing factor for CrossEntropyLoss (0 = off).
        resume_from:     Path to a checkpoint to resume training from.
                         Restores model weights, optimiser state, and history.

    Returns:
        history dict with keys:
            train_loss, val_loss, train_acc, val_acc  → list[float], one per epoch
            epoch_time                                → list[float], seconds
            lr                                        → list[float], LR each epoch
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Mixed precision scaler — only meaningful on CUDA
    scaler = (torch.cuda.amp.GradScaler()
              if DEVICE.type == "cuda" else None)

    # Initialise history
    history: dict[str, list] = {
        "train_loss" : [],
        "val_loss"   : [],
        "train_acc"  : [],
        "val_acc"    : [],
        "epoch_time" : [],
        "lr"         : [],
    }

    best_val_acc  = 0.0
    start_epoch   = 1

    # ── Resume from checkpoint if requested ───────────────────────────────
    if resume_from is not None and resume_from.exists():
        ckpt        = load_checkpoint(model, resume_from, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["val_acc"]
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"[train] Resuming {label} from epoch {start_epoch}")

    # ── Epoch loop ────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Training : {label}")
    print(f"  Epochs   : {start_epoch} → {epochs}")
    print(f"  Device   : {DEVICE}")
    print(f"  LR       : {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'─'*62}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()

        # ── Train ─────────────────────────────────────────────────────────
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler)

        # ── Validate ──────────────────────────────────────────────────────
        vl_loss, vl_acc = validate(model, val_loader, criterion)

        # ── LR scheduler step ─────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - t0

        # ── Record history ─────────────────────────────────────────────────
        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"]  .append(round(vl_loss, 6))
        history["train_acc"] .append(round(tr_acc,  6))
        history["val_acc"]   .append(round(vl_acc,  6))
        history["epoch_time"].append(round(elapsed,  2))
        history["lr"]        .append(round(current_lr, 8))

        # ── Console log ───────────────────────────────────────────────────
        improved = "  *" if vl_acc > best_val_acc else ""
        print(
            f"  [{label}] "
            f"Ep {epoch:3d}/{epochs}  |  "
            f"TrainAcc {tr_acc:.4f}  ValAcc {vl_acc:.4f}  |  "
            f"Loss {vl_loss:.4f}  |  "
            f"LR {current_lr:.1e}  |  "
            f"{elapsed:5.1f}s"
            f"{improved}"
        )

        # ── Save best checkpoint ──────────────────────────────────────────
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            if checkpoint_path is not None:
                save_checkpoint(
                    model, optimizer, epoch, vl_acc, history, checkpoint_path)

        # ── Early stopping check ──────────────────────────────────────────
        if early_stopping is not None and early_stopping(vl_acc):
            break

    # ── End-of-run summary ─────────────────────────────────────────────────
    print(f"{'─'*62}")
    print(f"  [{label}] Training complete")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Total time        : {sum(history['epoch_time'])/60:.1f} min")
    if checkpoint_path is not None:
        print(f"  Checkpoint saved  : {checkpoint_path}")
    print(f"{'─'*62}\n")

    # ── Auto-save history JSON ─────────────────────────────────────────────
    history_path = RESULT_DIR / f"{label.lower().replace(' ', '_')}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [train] History saved → {history_path}")

    return history


# ── LR scheduler factory ──────────────────────────────────────────────────────

def make_scheduler(
    optimizer  : torch.optim.Optimizer,
    schedule   : str  = "cosine",
    epochs     : int  = 50,
    warmup     : int  = 5,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a learning rate scheduler.

    Args:
        optimizer: The optimiser the scheduler wraps.
        schedule:  One of:
                     "cosine"   — CosineAnnealingLR, decays smoothly to near 0
                     "step"     — StepLR, drops by ×0.1 every 15 epochs
                     "warmup"   — linear warmup for `warmup` epochs then cosine
                     "plateau"  — ReduceLROnPlateau (watch val_loss)
        epochs:    Total training epochs (used by cosine and warmup).
        warmup:    Warmup epochs (only used when schedule="warmup").

    Returns:
        A configured scheduler object ready to call .step() each epoch.

    Note:
        For "plateau", you must call scheduler.step(val_loss) instead of
        scheduler.step() — handle this in your training loop.
    """
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    elif schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.1)

    elif schedule == "warmup":
        # Linear warmup → cosine decay
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)        # ramp up
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif schedule == "plateau":
        # `verbose=` was removed from ReduceLROnPlateau in PyTorch 2.6+. If
        # you want LR-drop logging, watch optimizer.param_groups[0]["lr"]
        # before and after scheduler.step(val_loss) in your training loop.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
            min_lr=1e-7)

    else:
        raise ValueError(
            f"Unknown schedule '{schedule}'. "
            f"Choose from: 'cosine', 'step', 'warmup', 'plateau'")


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test: one epoch of training on a tiny random dataset.
    Verifies the training loop runs without errors on your Mac.
    Does NOT use real CIFAR-100 data.
    """
    import torch
    from torch.utils.data import TensorDataset

    print("[train] Smoke test — random data, 2 mini-batches, 2 epochs")

    # Fake 32×32 CIFAR-like data
    fake_imgs   = torch.randn(256, 3, 32, 32)
    fake_labels = torch.randint(0, 100, (256,))
    fake_ds     = TensorDataset(fake_imgs, fake_labels)
    fake_loader = DataLoader(fake_ds, batch_size=64, shuffle=True)

    # Minimal model
    tiny_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 512),
        nn.ReLU(),
        nn.Linear(512, 100),
    ).to(DEVICE)

    opt  = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    sch  = make_scheduler(opt, schedule="cosine", epochs=2)
    es   = EarlyStopping(patience=5, verbose=True)

    hist = run_training(
        model           = tiny_model,
        train_loader    = fake_loader,
        val_loader      = fake_loader,   # same data, just for the test
        epochs          = 2,
        optimizer       = opt,
        scheduler       = sch,
        label           = "smoke_test",
        checkpoint_path = None,
        early_stopping  = es,
    )

    print(f"\nFinal train_acc : {hist['train_acc'][-1]:.4f}")
    print(f"Final val_acc   : {hist['val_acc'][-1]:.4f}")
    print("[train] Smoke test passed.")