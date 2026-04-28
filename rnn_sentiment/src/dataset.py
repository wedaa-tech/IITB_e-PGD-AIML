import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import BATCH_SIZE, DEVICE


class IMDBDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data: dict, batch_size: int = BATCH_SIZE):
    """
    Returns (train_loader, val_loader, test_loader).
    Pins memory only on CUDA; MPS uses normal transfer.
    """
    pin = str(DEVICE) == "cuda"

    train_loader = DataLoader(
        IMDBDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True,
        pin_memory=pin, num_workers=0,
    )
    val_loader = DataLoader(
        IMDBDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size, shuffle=False,
        pin_memory=pin, num_workers=0,
    )
    test_loader = DataLoader(
        IMDBDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size, shuffle=False,
        pin_memory=pin, num_workers=0,
    )
    return train_loader, val_loader, test_loader