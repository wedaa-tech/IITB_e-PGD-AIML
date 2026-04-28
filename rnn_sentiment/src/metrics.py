"""
Tracking training metrics and computing evaluation stats.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch


@dataclass
class MetricTracker:
    """Accumulates loss + correct predictions over a full epoch."""
    losses:    List[float] = field(default_factory=list)
    correct:   int = 0
    total:     int = 0

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        self.losses.append(loss)
        self.correct += (preds == labels).sum().item()
        self.total   += labels.size(0)

    @property
    def avg_loss(self) -> float:
        return float(np.mean(self.losses))

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def reset(self):
        self.losses.clear()
        self.correct = 0
        self.total   = 0


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """Threshold logits at 0 (pre-sigmoid) → predicted class."""
    preds = (logits > 0).long()
    return preds, (preds == labels.long())