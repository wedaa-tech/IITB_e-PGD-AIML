"""
src/utils.py
────────────
Shared helper utilities used across the project.

Exports:
    save_history()          → write training history dict to JSON
    load_history()          → read training history dict from JSON
    save_results()          → write evaluation results dict to JSON
    load_results()          → read evaluation results dict from JSON
    count_parameters()      → total and trainable parameter counts
    model_summary()         → layer-by-layer size table
    set_seed()              → global reproducibility seed
    format_time()           → seconds → human-readable string
    get_lr()                → current learning rate from optimiser
    log_system_info()       → print Mac / PyTorch / MPS environment info
    check_outputs_dir()     → verify all output directories exist
    export_results_csv()    → write final metrics dict to CSV
    plot_lr_schedule()      → visualise a learning rate schedule curve
    EpochProgressLogger     → compact per-epoch console table logger
"""

import os
import sys
import json
import csv
import time
import math
import random
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import (
    DEVICE,
    ROOT,
    RESULT_DIR,
    CKPT_DIR,
    PLOT_DIR,
    DATA_DIR,
    SCRATCH,
    TRANSFER,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds globally for reproducible training.

    Covers Python's random module, NumPy, PyTorch CPU, and
    PyTorch MPS/CUDA. Note that full determinism on MPS is not
    guaranteed by Apple's Metal backend for all operations.

    Args:
        seed: Integer seed value. Default matches RANDOM_SEED in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Disable non-deterministic CUDA ops — slight speed cost
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # MPS has no explicit seed API beyond torch.manual_seed above
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[utils] Seed set to {seed}")


# ── Directory checks ──────────────────────────────────────────────────────────

def check_outputs_dir() -> None:
    """
    Verify that all expected output directories exist and are writable.
    Creates any that are missing. Call this at the top of each entry-point
    script as a fast pre-flight check before any training begins.
    """
    dirs = {
        "Data"        : DATA_DIR,
        "Checkpoints" : CKPT_DIR,
        "Plots"       : PLOT_DIR,
        "Results"     : RESULT_DIR,
    }
    all_ok = True
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        writable = os.access(path, os.W_OK)
        status   = "OK" if writable else "NOT WRITABLE"
        if not writable:
            all_ok = False
        print(f"  [utils] {name:<14} {str(path):<55} [{status}]")

    if not all_ok:
        raise PermissionError(
            "[utils] One or more output directories are not writable. "
            "Check folder permissions."
        )


# ── System info ───────────────────────────────────────────────────────────────

def log_system_info() -> dict:
    """
    Print and return a summary of the current hardware and software
    environment. Useful to include at the top of a training log so you
    can reproduce or compare runs later.

    Returns:
        dict with keys: os, python, torch, device, mps, cpu_count,
                        ram_gb, mac_chip (Apple Silicon only)
    """
    info: dict[str, Any] = {}

    # OS
    info["os"]        = platform.platform()
    info["python"]    = sys.version.split()[0]
    info["torch"]     = torch.__version__

    # Device
    info["device"]    = str(DEVICE)
    info["mps"]       = torch.backends.mps.is_available()
    info["cuda"]      = torch.cuda.is_available()
    info["cpu_count"] = os.cpu_count()

    # RAM
    try:
        import resource
        # macOS: sysctl for total RAM
        ram_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            .decode().strip()
        )
        info["ram_gb"] = round(ram_bytes / (1024 ** 3), 1)
    except Exception:
        info["ram_gb"] = "unknown"

    # Mac chip name (Apple Silicon only)
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
        info["mac_chip"] = chip
    except Exception:
        info["mac_chip"] = platform.processor()

    # Print
    print("\n" + "─" * 55)
    print("  System information")
    print("─" * 55)
    for k, v in info.items():
        print(f"  {k:<14} {v}")
    print("─" * 55 + "\n")

    return info


# ── History — save / load ─────────────────────────────────────────────────────

def save_history(history: dict, filename: str) -> Path:
    """
    Serialise a training history dict to JSON.

    Args:
        history:  Dict produced by run_training() — lists of floats
                  keyed by train_loss, val_loss, train_acc, val_acc,
                  epoch_time, lr.
        filename: Name of the file (e.g. "scratch_history.json").
                  Saved inside RESULT_DIR automatically.

    Returns:
        Path to the saved file.
    """
    # Ensure all values are JSON-serialisable (convert numpy types)
    def _serialise(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    path = RESULT_DIR / filename
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=_serialise)
    print(f"[utils] History saved  → {path}")
    return path


def load_history(filename: str) -> dict:
    """
    Load a training history dict previously saved by save_history().

    Args:
        filename: Name of the JSON file inside RESULT_DIR.

    Returns:
        History dict with lists of per-epoch metrics.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    path = RESULT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"[utils] History file not found: {path}\n"
            f"  Run the corresponding training script first."
        )
    with open(path) as f:
        history = json.load(f)
    epochs = len(history.get("val_acc", []))
    print(f"[utils] History loaded ← {path}  ({epochs} epochs)")
    return history


# ── Results — save / load ─────────────────────────────────────────────────────

def save_results(results: dict, filename: str) -> Path:
    """
    Save an evaluation results dict to JSON.
    NumPy arrays (preds, labels, probs) are excluded automatically
    since they are large and not human-readable as JSON.

    Args:
        results:  Output of evaluate() from evaluate.py.
        filename: Filename inside RESULT_DIR.

    Returns:
        Path to the saved file.
    """
    # Strip non-serialisable numpy arrays, keep scalar metrics only
    clean = {
        k: float(v) if isinstance(v, (float, np.floating)) else v
        for k, v in results.items()
        if not isinstance(v, np.ndarray)
    }
    path = RESULT_DIR / filename
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[utils] Results saved  → {path}")
    return path


def load_results(filename: str) -> dict:
    """
    Load evaluation results saved by save_results().

    Args:
        filename: Filename inside RESULT_DIR.

    Returns:
        Results dict (scalar metrics only — numpy arrays not restored).
    """
    path = RESULT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"[utils] Results file not found: {path}\n"
            f"  Run compare.py first."
        )
    with open(path) as f:
        results = json.load(f)
    print(f"[utils] Results loaded ← {path}")
    return results


# ── CSV export ────────────────────────────────────────────────────────────────

def export_results_csv(results: dict[str, dict], filename: str = "comparison_table.csv") -> Path:
    """
    Write a model-comparison results dict to a tidy CSV file.
    One row per metric, one column per model.

    Args:
        results:  {model_label: metrics_dict}
                  Metrics dict should contain top1, top5, loss,
                  params, avg_epoch_time, ms_per_image.
        filename: Output filename inside RESULT_DIR.

    Returns:
        Path to the saved CSV.

    Example output (comparison_table.csv):
        metric,Scratch CNN,EfficientNet-B0
        top1,0.5841,0.7612
        top5,0.8330,0.9310
        ...
    """
    metric_keys = [
        ("top1",           "Top-1 accuracy"),
        ("top5",           "Top-5 accuracy"),
        ("loss",           "Test loss"),
        ("params",         "Parameters"),
        ("avg_epoch_time", "Avg epoch time (s)"),
        ("ms_per_image",   "Inference ms/image"),
    ]

    model_names = list(results.keys())
    path = RESULT_DIR / filename

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + model_names)
        for key, label in metric_keys:
            row = [label]
            for name in model_names:
                val = results[name].get(key, "")
                row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
            writer.writerow(row)

    print(f"[utils] CSV exported   → {path}")
    return path


# ── Model inspection ──────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Args:
        model: Any nn.Module.

    Returns:
        dict with keys:
            total      int — all parameters including frozen ones
            trainable  int — only parameters with requires_grad=True
            frozen     int — parameters that will not be updated
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"  [utils] Total params     : {total:>12,}")
    print(f"  [utils] Trainable params : {trainable:>12,}")
    print(f"  [utils] Frozen params    : {frozen:>12,}")

    return {"total": total, "trainable": trainable, "frozen": frozen}


def model_summary(
    model      : nn.Module,
    input_size : tuple = (1, 3, 32, 32),
    max_rows   : int   = 40,
) -> None:
    """
    Print a compact layer-by-layer summary: name, type, output shape, params.
    Does a single forward pass with a dummy input to capture output shapes.

    Args:
        model:      Any nn.Module already on DEVICE.
        input_size: Tuple (batch, C, H, W). Default matches CIFAR-100 scratch mode.
                    Use (1, 3, 224, 224) for the transfer model.
        max_rows:   Truncate the table after this many layers.
    """
    hooks   = []
    rows    = []

    def _hook(name, module):
        def fn(mod, inp, out):
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            if isinstance(out, torch.Tensor):
                shape = tuple(out.shape)
            else:
                shape = "—"
            rows.append((name, type(mod).__name__, shape, n_params))
        return fn

    for name, module in model.named_modules():
        if name == "":
            continue   # skip root module
        h = module.register_forward_hook(_hook(name, module))
        hooks.append(h)

    dummy = torch.zeros(*input_size).to(DEVICE)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception as e:
            print(f"[utils] model_summary forward pass failed: {e}")
            for h in hooks:
                h.remove()
            return

    for h in hooks:
        h.remove()

    col_name   = 38
    col_type   = 22
    col_shape  = 22
    col_params = 12
    sep        = "─" * (col_name + col_type + col_shape + col_params + 6)

    print("\n" + sep)
    print(f"  {'Layer':<{col_name}} {'Type':<{col_type}} "
          f"{'Output shape':<{col_shape}} {'Params':>{col_params}}")
    print(sep)

    displayed = 0
    for name, type_name, shape, n_params in rows:
        if displayed >= max_rows:
            remaining = len(rows) - max_rows
            print(f"  ... ({remaining} more layers not shown)")
            break
        shape_str  = str(shape)
        params_str = f"{n_params:,}" if n_params > 0 else "—"
        print(f"  {name:<{col_name}} {type_name:<{col_type}} "
              f"{shape_str:<{col_shape}} {params_str:>{col_params}}")
        displayed += 1

    print(sep)
    count_parameters(model)
    print()


# ── Learning rate utilities ───────────────────────────────────────────────────

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Return the current learning rate from the first parameter group.
    Works for both single-LR and discriminative-LR (multiple param groups)
    optimisers — returns the first group's LR in both cases.

    Args:
        optimizer: Any PyTorch optimiser.

    Returns:
        Current LR as a float.
    """
    return optimizer.param_groups[0]["lr"]


def get_all_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Return learning rates for ALL parameter groups.
    Useful for transfer learning where backbone and head use different LRs.

    Args:
        optimizer: Any PyTorch optimiser.

    Returns:
        List of LR floats, one per parameter group.
    """
    return [pg["lr"] for pg in optimizer.param_groups]


def plot_lr_schedule(
    optimizer  : torch.optim.Optimizer,
    scheduler  : object,
    epochs     : int,
    save_name  : str  = "lr_schedule.png",
    show       : bool = True,
) -> list[float]:
    """
    Simulate the full LR schedule and plot it — without touching model weights.
    Call this before training to verify the schedule looks correct.

    Args:
        optimizer: Freshly created optimiser (will be stepped in a simulation).
        scheduler: Freshly created scheduler wrapping the optimiser.
        epochs:    Number of epochs to simulate.
        save_name: Filename inside outputs/plots/.
        show:      Call plt.show() after saving.

    Returns:
        List of LR values, one per epoch.

    Note:
        This advances the scheduler's internal state, so create fresh
        optimiser + scheduler instances just for this call — do not reuse
        them for actual training.
    """
    lrs = []
    for _ in range(epochs):
        lrs.append(get_lr(optimizer))
        scheduler.step()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, epochs + 1), lrs, lw=2, color="#534AB7")
    ax.fill_between(range(1, epochs + 1), lrs, alpha=0.12, color="#534AB7")
    ax.set_title("Learning rate schedule", fontsize=11)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Learning rate", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = PLOT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[utils] LR schedule plot → {path}")
    if show:
        plt.show()
    plt.close()

    return lrs


# ── Time formatting ───────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable string.

    Examples:
        format_time(45.3)    → "45s"
        format_time(125.0)   → "2m 05s"
        format_time(3725.0)  → "1h 02m 05s"
    """
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def estimate_remaining(
    elapsed_times : list[float],
    epochs_done   : int,
    total_epochs  : int,
) -> str:
    """
    Estimate remaining training time using the mean of recent epoch times.

    Args:
        elapsed_times: List of per-epoch durations (seconds) so far.
        epochs_done:   Number of epochs completed.
        total_epochs:  Total epochs planned.

    Returns:
        Human-readable ETA string.
    """
    if not elapsed_times:
        return "unknown"
    recent_mean  = np.mean(elapsed_times[-5:])       # rolling 5-epoch average
    remaining    = (total_epochs - epochs_done) * recent_mean
    return format_time(remaining)


# ── Console logging ───────────────────────────────────────────────────────────

@dataclass
class EpochProgressLogger:
    """
    Prints a compact, aligned per-epoch table to the console.
    Tracks best val_acc and ETA internally.

    Usage:
        logger = EpochProgressLogger(total_epochs=50, label="Scratch")
        for epoch in range(1, 51):
            train_loss, train_acc = train_one_epoch(...)
            val_loss,   val_acc   = validate(...)
            logger.log(epoch, train_loss, train_acc, val_loss, val_acc, lr)
    """
    total_epochs : int
    label        : str  = "model"

    _best_val_acc : float       = field(default=0.0,  init=False, repr=False)
    _times        : list[float] = field(default_factory=list, init=False, repr=False)
    _start        : float       = field(default_factory=time.perf_counter,
                                        init=False, repr=False)

    def __post_init__(self):
        header = (
            f"\n  {'Ep':>4}  {'TrainAcc':>9}  {'ValAcc':>9}  "
            f"{'TrainLoss':>10}  {'ValLoss':>9}  {'LR':>9}  "
            f"{'Time':>7}  {'ETA':>9}"
        )
        print("─" * len(header.rstrip()))
        print(header)
        print("─" * len(header.rstrip()))

    def log(
        self,
        epoch      : int,
        train_loss : float,
        train_acc  : float,
        val_loss   : float,
        val_acc    : float,
        lr         : float,
        epoch_time : float | None = None,
    ) -> None:
        """
        Log one epoch row. Call once per epoch after validation.

        Args:
            epoch:      Current epoch number (1-based).
            train_loss: Average training loss this epoch.
            train_acc:  Training top-1 accuracy this epoch.
            val_loss:   Average validation loss this epoch.
            val_acc:    Validation top-1 accuracy this epoch.
            lr:         Current learning rate (first param group).
            epoch_time: Duration of this epoch in seconds.
                        If None, wall-clock time since last log() is used.
        """
        now = time.perf_counter()
        if epoch_time is None:
            epoch_time = now - (self._times[-1] if self._times else self._start)

        self._times.append(now)

        is_best = val_acc > self._best_val_acc
        if is_best:
            self._best_val_acc = val_acc
        marker  = " *" if is_best else "  "

        eta = estimate_remaining(
            [t2 - t1 for t1, t2 in zip(self._times[:-1], self._times[1:])],
            epoch,
            self.total_epochs,
        )

        print(
            f"  [{self.label}] "
            f"{epoch:>3}/{self.total_epochs}  "
            f"{train_acc:>8.4f}  "
            f"{val_acc:>8.4f}  "
            f"{train_loss:>10.5f}  "
            f"{val_loss:>9.5f}  "
            f"{lr:>9.2e}  "
            f"{format_time(epoch_time):>7}  "
            f"{eta:>9}"
            f"{marker}"
        )

    def summary(self) -> None:
        """Print end-of-training summary."""
        total = time.perf_counter() - self._start
        print("─" * 80)
        print(f"  [{self.label}] Training complete")
        print(f"  Best val accuracy : {self._best_val_acc:.4f}")
        print(f"  Total time        : {format_time(total)}")
        print("─" * 80 + "\n")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def list_checkpoints() -> list[Path]:
    """
    List all .pth checkpoint files in CKPT_DIR, sorted newest first.

    Returns:
        List of Path objects.
    """
    ckpts = sorted(CKPT_DIR.glob("*.pth"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        print("[utils] No checkpoints found in", CKPT_DIR)
    else:
        print(f"[utils] Checkpoints in {CKPT_DIR}:")
        for p in ckpts:
            size_mb = p.stat().st_size / (1024 ** 2)
            mtime   = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {p.name:<35} {size_mb:5.1f} MB   {mtime}")
    return ckpts


def checkpoint_info(path: Path) -> dict:
    """
    Load only the metadata from a checkpoint (epoch, val_acc, keys present)
    without loading model weights into memory.

    Args:
        path: Path to a .pth checkpoint.

    Returns:
        dict with epoch, val_acc, keys.
    """
    ckpt = torch.load(path, map_location="cpu")
    info = {
        "epoch"  : ckpt.get("epoch",   "unknown"),
        "val_acc": ckpt.get("val_acc", "unknown"),
        "keys"   : list(ckpt.keys()),
    }
    print(f"[utils] Checkpoint: {path.name}")
    print(f"  epoch   : {info['epoch']}")
    print(f"  val_acc : {info['val_acc']}")
    print(f"  keys    : {info['keys']}")
    return info


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  src/utils.py — smoke test")
    print("=" * 55)

    # ── set_seed ───────────────────────────────────────────────────────────
    print("\n── set_seed() ───────────────────────────────────────────────")
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b), "Seed not reproducible!"
    print(f"  Reproducible tensors match: {a.tolist()}")

    # ── check_outputs_dir ─────────────────────────────────────────────────
    print("\n── check_outputs_dir() ──────────────────────────────────────")
    check_outputs_dir()

    # ── log_system_info ───────────────────────────────────────────────────
    print("\n── log_system_info() ────────────────────────────────────────")
    info = log_system_info()

    # ── save / load history ───────────────────────────────────────────────
    print("\n── save_history() / load_history() ──────────────────────────")
    fake_history = {
        "train_loss" : [4.5, 4.0, 3.6],
        "val_loss"   : [4.6, 4.1, 3.8],
        "train_acc"  : [0.05, 0.10, 0.18],
        "val_acc"    : [0.04, 0.09, 0.16],
        "epoch_time" : [60.1, 58.3, 59.7],
        "lr"         : [1e-3, 9e-4, 8e-4],
    }
    save_history(fake_history, "_smoke_history.json")
    loaded = load_history("_smoke_history.json")
    assert loaded["val_acc"] == fake_history["val_acc"]
    print("  Round-trip OK")
    (RESULT_DIR / "_smoke_history.json").unlink()   # clean up

    # ── save / load results ───────────────────────────────────────────────
    print("\n── save_results() / load_results() ──────────────────────────")
    fake_results = {"top1": 0.5841, "top5": 0.8330, "loss": 2.214}
    save_results(fake_results, "_smoke_results.json")
    lr2 = load_results("_smoke_results.json")
    assert abs(lr2["top1"] - 0.5841) < 1e-6
    print("  Round-trip OK")
    (RESULT_DIR / "_smoke_results.json").unlink()   # clean up

    # ── count_parameters + model_summary ──────────────────────────────────
    print("\n── count_parameters() + model_summary() ─────────────────────")
    tiny = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 100),
    ).to(DEVICE)
    count_parameters(tiny)
    model_summary(tiny, input_size=(1, 3, 32, 32), max_rows=10)

    # ── get_lr / get_all_lrs ──────────────────────────────────────────────
    print("\n── get_lr() / get_all_lrs() ─────────────────────────────────")
    opt = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
    print(f"  Single LR   : {get_lr(opt):.2e}")
    opt2 = torch.optim.AdamW([
        {"params": list(tiny.parameters())[:2], "lr": 1e-5},
        {"params": list(tiny.parameters())[2:], "lr": 1e-3},
    ])
    print(f"  All LRs     : {get_all_lrs(opt2)}")

    # ── plot_lr_schedule ──────────────────────────────────────────────────
    print("\n── plot_lr_schedule() ───────────────────────────────────────")
    opt_sim = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
    sch_sim = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_sim, T_max=50, eta_min=1e-6)
    lrs = plot_lr_schedule(opt_sim, sch_sim, epochs=50,
                            save_name="smoke_lr_schedule.png", show=False)
    assert len(lrs) == 50
    print(f"  LR at ep 1  : {lrs[0]:.2e}")
    print(f"  LR at ep 50 : {lrs[-1]:.2e}")

    # ── format_time / estimate_remaining ──────────────────────────────────
    print("\n── format_time() / estimate_remaining() ─────────────────────")
    for secs in [45, 125, 3725]:
        print(f"  {secs}s → {format_time(secs)}")
    eta = estimate_remaining([60.1, 58.3, 59.7, 57.4, 61.0],
                               epochs_done=5, total_epochs=50)
    print(f"  ETA after 5/50 epochs: {eta}")

    # ── EpochProgressLogger ───────────────────────────────────────────────
    print("\n── EpochProgressLogger ──────────────────────────────────────")
    logger = EpochProgressLogger(total_epochs=3, label="smoke")
    for ep in range(1, 4):
        logger.log(ep,
                   train_loss=4.5 - ep * 0.3,
                   train_acc =0.05 + ep * 0.05,
                   val_loss  =4.6 - ep * 0.25,
                   val_acc   =0.04 + ep * 0.05,
                   lr        =1e-3,
                   epoch_time=58.0)
    logger.summary()

    # ── list_checkpoints ──────────────────────────────────────────────────
    print("\n── list_checkpoints() ───────────────────────────────────────")
    list_checkpoints()

    # ── export_results_csv ────────────────────────────────────────────────
    print("\n── export_results_csv() ─────────────────────────────────────")
    export_results_csv({
        "Scratch CNN"  : {"top1": 0.584, "top5": 0.833, "loss": 2.21,
                          "params": 9_222_500, "avg_epoch_time": 57.3,
                          "ms_per_image": 0.21},
        "EfficientNet" : {"top1": 0.761, "top5": 0.931, "loss": 1.48,
                          "params": 5_288_548, "avg_epoch_time": 95.1,
                          "ms_per_image": 0.38},
    }, filename="_smoke_comparison.csv")
    (RESULT_DIR / "_smoke_comparison.csv").unlink()  # clean up

    print("\n[utils] All smoke tests passed.")