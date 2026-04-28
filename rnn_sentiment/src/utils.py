import random
import numpy as np
import torch
from src.config import RANDOM_SEED, DEVICE

def set_seed(seed: int = RANDOM_SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device_info():
    print(f"  Device  : {DEVICE}")
    print(f"  PyTorch : {torch.__version__}")
    if str(DEVICE) == "mps":
        print(f"  Backend : Apple MPS (Metal GPU)")

def progress(iterable, desc="", total=None):
    """Thin tqdm wrapper."""
    from tqdm import tqdm
    return tqdm(iterable, desc=desc, total=total, ncols=90)