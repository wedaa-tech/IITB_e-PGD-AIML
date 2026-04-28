import os
from pathlib import Path
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
LOG_DIR        = BASE_DIR / "outputs" / "logs"

for d in [DATA_RAW, DATA_PROCESSED, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device (MPS → CUDA → CPU) ─────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ── Preprocessing ─────────────────────────────────────────────────────────
VOCAB_SIZE   = 20_000      # top-N tokens kept
MAX_SEQ_LEN  = 500         # truncate/pad to this length
PAD_TOKEN    = "<PAD>"     # index 0
UNK_TOKEN    = "<UNK>"     # index 1
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

# Negations to keep even after stop-word removal
NEGATIONS = {"not", "no", "never", "nor", "neither",
             "nobody", "nothing", "nowhere", "hardly",
             "scarcely", "barely", "doesn't", "isn't",
             "wasn't", "shouldn't", "wouldn't", "couldn't",
             "won't", "can't", "don't"}

# ── Dataset split ─────────────────────────────────────────────────────────
TRAIN_SIZE = 20_000   # out of 25k train reviews
VAL_SIZE   = 5_000    # remaining 5k used for validation
RANDOM_SEED = 42

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3

# NUM_EPOCHS is overridable from the shell via the RNN_EPOCHS env var so the
# same number can be shared by main.py, experiment.py, and hparam_search.py
# (used by run_all.sh --epochs N). Falls back to 10 when the variable is unset
# or not a positive integer.
def _epochs_from_env(default: int = 10) -> int:
    raw = os.environ.get("RNN_EPOCHS", "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
        return n if n > 0 else default
    except ValueError:
        return default

NUM_EPOCHS = _epochs_from_env()