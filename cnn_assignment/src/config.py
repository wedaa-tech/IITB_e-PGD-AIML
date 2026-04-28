import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# ROOT resolves to the project root (one level above src/)
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"

_CIFAR_PATH = DATA_DIR / "cifar-100-python"
if not _CIFAR_PATH.exists():
    raise FileNotFoundError(
        f"\n[config] CIFAR-100 data not found at: {_CIFAR_PATH}\n"
        f"  Expected structure:\n"
        f"    {DATA_DIR}/\n"
        f"    └── cifar-100-python/\n"
        f"        ├── train\n"
        f"        ├── test\n"
        f"        └── meta\n\n"
        f"  If you have cifar-100-python.tar.gz, extract it:\n"
        f"    cd data && tar -xzf cifar-100-python.tar.gz"
    )
print(f"[config] CIFAR-100 data found at: {_CIFAR_PATH}")

CKPT_DIR   = ROOT / "outputs" / "checkpoints"
PLOT_DIR   = ROOT / "outputs" / "plots"
RESULT_DIR = ROOT / "outputs" / "results"

# Create all output directories automatically on import
for _dir in [DATA_DIR, CKPT_DIR, PLOT_DIR, RESULT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Device detection ───────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Priority order:
      1. MPS  — Apple Silicon (M1/M2/M3) GPU
      2. CUDA — NVIDIA GPU (not present on Mac, but kept for portability)
      3. CPU  — Intel Mac fallback
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[config] Device : {DEVICE}")
print(f"[config] PyTorch: {torch.__version__}")


# ── Dataset constants ──────────────────────────────────────────────────────────
NUM_CLASSES = 100
TRAIN_SIZE  = 45_000   # carved out of the 50k training set
VAL_SIZE    = 5_000    # remaining 5k used for validation
RANDOM_SEED = 42

# Per-channel mean and std for CIFAR-100 (precomputed over the training split)
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD  = [0.2675, 0.2565, 0.2761]

# ImageNet mean and std — used when feeding upscaled images into pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Scratch CNN hyperparameters ────────────────────────────────────────────────
SCRATCH = {
    # Data
    "input_size"  : 32,          # native CIFAR-100 resolution — no upscaling needed
    "batch_size"  : 128,
    "num_workers" : 0,           # 0 avoids macOS multiprocessing errors

    # Optimiser
    "lr"          : 1e-3,
    "weight_decay": 1e-4,

    # Schedule
    "epochs"      : 50,          # cosine annealing over the full run

    # I/O
    "checkpoint"  : CKPT_DIR / "scratch_best.pth",
    "history_file": "scratch_history.json",
}


# ── Transfer learning hyperparameters ──────────────────────────────────────────
TRANSFER = {
    # Backbone (any timm model name works here — swap freely)
    "backbone"         : "efficientnet_b0",

    # Data
    "input_size"       : 224,    # upscale 32×32 → 224×224 for ImageNet-pretrained models
    "batch_size"       : 64,     # smaller than scratch — upscaled images use more RAM
    "num_workers"      : 0,

    # Phase A — head-only training (backbone frozen)
    "epochs_frozen"    : 15,
    "lr_head"          : 1e-3,   # head weights are randomly initialised → higher LR

    # Phase B — full fine-tuning (all layers unfrozen)
    "epochs_finetune"  : 25,
    "lr_backbone"      : 1e-5,   # pretrained layers → very small LR to avoid forgetting
    # lr_head is reused from Phase A for the head in Phase B as well

    # Shared
    "weight_decay"     : 1e-4,

    # I/O
    "checkpoint"       : CKPT_DIR / "transfer_best.pth",
    "history_file"     : "transfer_history.json",
}


# ── Evaluation constants ───────────────────────────────────────────────────────
EVAL = {
    "top_k"            : 5,                          # report top-1 and top-5
    "noise_levels"     : [0.0, 0.05, 0.1, 0.2, 0.3],  # Gaussian σ for robustness test
    "n_inference_batch": 20,                         # batches used to time inference
}


# ── Quick sanity-check (runs when you do: python src/config.py) ────────────────
if __name__ == "__main__":
    print(f"\nProject root : {ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Checkpoints  : {CKPT_DIR}")
    print(f"Plots        : {PLOT_DIR}")
    print(f"Results      : {RESULT_DIR}")
    print(f"\nScratch epochs       : {SCRATCH['epochs']}")
    print(f"Transfer backbone    : {TRANSFER['backbone']}")
    print(f"Transfer input size  : {TRANSFER['input_size']}×{TRANSFER['input_size']}")
    print(f"Phase A epochs       : {TRANSFER['epochs_frozen']}")
    print(f"Phase B epochs       : {TRANSFER['epochs_finetune']}")