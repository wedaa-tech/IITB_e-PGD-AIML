"""
src/models/transfer_model.py
─────────────────────────────
Pretrained backbone models fine-tuned on CIFAR-100.
All backbones are loaded via the `timm` library with ImageNet weights.

Two-phase fine-tuning strategy
───────────────────────────────
Phase A — Feature extraction  (run_transfer.py, first N epochs)
    Backbone is frozen. Only the classification head is trained.
    Use a higher learning rate (1e-3) since the head starts random.
    Quickly adapts the head to CIFAR-100 class structure.

Phase B — Full fine-tuning  (run_transfer.py, next M epochs)
    All layers are unfrozen. Entire network is trained end-to-end.
    Discriminative learning rates: backbone 1e-5, head 1e-3.
    Low backbone LR prevents destroying ImageNet feature knowledge.

Exports:
    build_transfer_model()    → load backbone + fresh head
    freeze_backbone()         → freeze all non-head parameters (Phase A)
    unfreeze_all()            → unfreeze everything (Phase B)
    get_param_groups()        → discriminative LR param groups
    get_backbone_names()      → list available timm backbones
    count_parameters()        → total / trainable / frozen counts
    TransferModelWrapper      → thin wrapper with extra utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Backbone registry ─────────────────────────────────────────────────────────

# Curated set of timm backbones that work well on CIFAR-100 with 224×224 input.
# All are pretrained on ImageNet-1k unless noted.
# Sorted by approximate parameter count.
SUPPORTED_BACKBONES: dict[str, dict] = {
    "efficientnet_b0": {
        "params_m"   : 5.3,
        "input_size" : 224,
        "notes"      : "Best accuracy/speed tradeoff. Recommended default.",
    },
    "efficientnet_b1": {
        "params_m"   : 7.8,
        "input_size" : 240,
        "notes"      : "Slightly stronger than B0. Needs input_size=240.",
    },
    "resnet50": {
        "params_m"   : 25.6,
        "input_size" : 224,
        "notes"      : "Classic baseline. Strong but heavier than EfficientNet.",
    },
    "resnet34": {
        "params_m"   : 21.8,
        "input_size" : 224,
        "notes"      : "Lighter ResNet. Good for CPU/Intel Mac runs.",
    },
    "mobilenetv3_large_100": {
        "params_m"   : 5.5,
        "input_size" : 224,
        "notes"      : "Very fast inference. Good for latency comparison.",
    },
    "densenet121": {
        "params_m"   : 8.0,
        "input_size" : 224,
        "notes"      : "Dense connections. Strong feature reuse.",
    },
    "vgg16_bn": {
        "params_m"   : 138.4,
        "input_size" : 224,
        "notes"      : "Heavy. Use only if you have plenty of RAM.",
    },
}


def get_backbone_names() -> list[str]:
    """
    Return the list of curated backbone names supported by this project.
    Pass any of these strings to build_transfer_model(backbone=...).

    Example:
        for name in get_backbone_names():
            print(name)
    """
    return list(SUPPORTED_BACKBONES.keys())


def print_backbone_table() -> None:
    """Print a formatted table of all supported backbones and their properties."""
    print("\n" + "─" * 72)
    print(f"  {'Backbone':<30} {'Params (M)':>12} {'Input':>8}  Notes")
    print("─" * 72)
    for name, meta in SUPPORTED_BACKBONES.items():
        print(f"  {name:<30} {meta['params_m']:>10.1f}M "
              f"{meta['input_size']:>6}px  {meta['notes']}")
    print("─" * 72 + "\n")


# ── Head detection helpers ─────────────────────────────────────────────────────

# timm uses different attribute names for the classification head
# depending on the architecture family.
_HEAD_ATTR_NAMES = ["classifier", "head", "fc", "head.fc"]


def _find_head_attr(model: nn.Module) -> str | None:
    """
    Find the name of the classification head attribute in a timm model.
    Different architectures use different attribute names.

    Returns the attribute name string, or None if not found.
    """
    for attr in _HEAD_ATTR_NAMES:
        # Support dotted paths like "head.fc"
        parts = attr.split(".")
        obj   = model
        try:
            for part in parts:
                obj = getattr(obj, part)
            return attr
        except AttributeError:
            continue
    return None


def _is_head_param(name: str) -> bool:
    """
    Return True if a named parameter belongs to the classification head.
    Matches by checking if any known head attribute name appears in the
    parameter's fully-qualified name.
    """
    head_keywords = ["classifier", "head", ".fc."]
    return any(kw in name for kw in head_keywords)


# ── Core factory ──────────────────────────────────────────────────────────────

def build_transfer_model(
    backbone    : str  = "efficientnet_b0",
    num_classes : int  = 100,
    pretrained  : bool = True,
) -> nn.Module:
    """
    Load a pretrained timm model with a fresh classification head
    sized for num_classes.

    timm replaces the original head automatically when num_classes is
    passed to create_model() — no manual surgery needed.

    Args:
        backbone:    timm model name. See get_backbone_names() for curated list.
                     Any valid timm name works (e.g. "resnet18", "vit_small_patch16_224").
        num_classes: Number of output classes. 100 for CIFAR-100.
        pretrained:  Load ImageNet weights. Set False only for testing or
                     when loading a saved checkpoint.

    Returns:
        nn.Module with pretrained backbone + fresh head on CPU.
        Move to device with .to(DEVICE) in the calling script.

    Raises:
        ImportError      if timm is not installed.
        RuntimeError     if the model name is not found in timm's registry.
        ValueError       if backbone is not in SUPPORTED_BACKBONES (warning only).
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for transfer learning.\n"
            "Install it with:  pip install timm"
        )

    if backbone not in SUPPORTED_BACKBONES:
        print(f"  [transfer] WARNING: '{backbone}' is not in the curated list.")
        print(f"  [transfer] Proceeding anyway — results may vary.")
        print(f"  [transfer] Curated backbones: {get_backbone_names()}")

    source = "pretrained ImageNet weights" if pretrained else "random weights"
    print(f"  [transfer] Loading '{backbone}'  ({source})")

    try:
        model = timm.create_model(
            backbone,
            pretrained  = pretrained,
            num_classes = num_classes,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load timm model '{backbone}'.\n"
            f"Original error: {e}\n"
            f"Check the model name at https://huggingface.co/timm"
        ) from e

    total = sum(p.numel() for p in model.parameters())
    print(f"  [transfer] Total parameters: {total:,}")

    # Confirm head was replaced
    head_attr = _find_head_attr(model)
    if head_attr:
        print(f"  [transfer] Head attribute  : '{head_attr}'")

    return model


# ── Phase A: freeze backbone ──────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Phase A — freeze all parameters except the classification head.

    After calling this:
        - Backbone parameters: requires_grad = False  (not updated)
        - Head parameters:     requires_grad = True   (updated)

    Use a relatively high learning rate (1e-3) for the head since its
    weights are randomly initialised and need to learn quickly.

    Args:
        model: timm model returned by build_transfer_model().

    Returns:
        The same model with backbone frozen (mutates in place, also returns
        for convenient chaining).
    """
    frozen    = 0
    trainable = 0

    for name, param in model.named_parameters():
        if _is_head_param(name):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    total = frozen + trainable
    print(f"\n  [freeze_backbone] Backbone frozen")
    print(f"  Frozen params    : {frozen:>12,}  ({frozen/total*100:.1f}%)")
    print(f"  Trainable params : {trainable:>12,}  ({trainable/total*100:.1f}%)  ← head only")

    if trainable == 0:
        raise RuntimeError(
            "[freeze_backbone] No trainable parameters found.\n"
            f"  Head keywords searched: {_HEAD_ATTR_NAMES}\n"
            f"  Check that _is_head_param() matches your backbone's architecture."
        )

    return model


# ── Phase B: unfreeze all ─────────────────────────────────────────────────────

def unfreeze_all(model: nn.Module) -> nn.Module:
    """
    Phase B — unfreeze every parameter for full end-to-end fine-tuning.

    After calling this, use discriminative learning rates via
    get_param_groups() so the backbone is updated very slowly and
    the head can still adapt quickly.

    Args:
        model: timm model (typically after Phase A training).

    Returns:
        The same model with all parameters trainable.
    """
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"\n  [unfreeze_all] All layers unfrozen")
    print(f"  Trainable params : {total:>12,}  (100%)")
    return model


# ── Discriminative learning rates ─────────────────────────────────────────────

def get_param_groups(
    model        : nn.Module,
    lr_backbone  : float = 1e-5,
    lr_head      : float = 1e-3,
    weight_decay : float = 1e-4,
) -> list[dict]:
    """
    Split model parameters into two groups with different learning rates.

    Discriminative LRs are critical for Phase B:
        - Backbone LR (1e-5): Very small. Backbone has learned rich ImageNet
          features — large updates would destroy this knowledge (catastrophic
          forgetting). We nudge these weights towards CIFAR-100 gradually.
        - Head LR (1e-3):     Larger. The head adapts freely since it was
          either freshly initialised or only lightly trained in Phase A.

    Args:
        model:        timm model with all parameters unfrozen.
        lr_backbone:  Learning rate for backbone parameters.
        lr_head:      Learning rate for head parameters.
        weight_decay: Applied to both groups.

    Returns:
        List of two param-group dicts ready to pass to an optimiser:
            [{"params": backbone_params, "lr": lr_backbone, ...},
             {"params": head_params,     "lr": lr_head,     ...}]

    Usage:
        groups = get_param_groups(model, lr_backbone=1e-5, lr_head=1e-3)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    """
    backbone_params = []
    head_params     = []
    backbone_count  = 0
    head_count      = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head_param(name):
            head_params.append(param)
            head_count += param.numel()
        else:
            backbone_params.append(param)
            backbone_count += param.numel()

    print(f"\n  [get_param_groups] Discriminative LR groups:")
    print(f"  Backbone → {backbone_count:>12,} params  lr={lr_backbone:.1e}")
    print(f"  Head     → {head_count:>12,} params  lr={lr_head:.1e}")

    if not backbone_params:
        print("  [get_param_groups] WARNING: no backbone params found. "
              "Did you call unfreeze_all() first?")
    if not head_params:
        raise RuntimeError(
            "[get_param_groups] No head parameters found.\n"
            "Check _is_head_param() matches your backbone's architecture."
        )

    return [
        {
            "params"      : backbone_params,
            "lr"          : lr_backbone,
            "weight_decay": weight_decay,
            "name"        : "backbone",
        },
        {
            "params"      : head_params,
            "lr"          : lr_head,
            "weight_decay": weight_decay,
            "name"        : "head",
        },
    ]


# ── Parameter utilities ───────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Count total, trainable, and frozen parameters.

    Args:
        model: Any nn.Module.

    Returns:
        dict with keys: total, trainable, frozen.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"  Parameters — "
          f"total: {total:>12,}  "
          f"trainable: {trainable:>12,}  "
          f"frozen: {frozen:>10,}")

    return {"total": total, "trainable": trainable, "frozen": frozen}


# ── TransferModelWrapper ──────────────────────────────────────────────────────

class TransferModelWrapper(nn.Module):
    """
    Thin wrapper around a timm model that adds:
        - extract_features()  → backbone output before the head
        - predict_proba()     → softmax probabilities
        - freeze_backbone()   → convenience method (calls module-level function)
        - unfreeze_all()      → convenience method
        - __repr__()          → informative summary string

    This wrapper is optional — run_transfer.py works fine with the raw
    timm model. Use it when you want the extra utility methods available
    as model.extract_features() rather than standalone function calls.

    Args:
        backbone:    timm model name string.
        num_classes: Output classes.
        pretrained:  Load ImageNet weights.

    Usage:
        model = TransferModelWrapper("efficientnet_b0", num_classes=100)
        model.freeze_backbone()
        # ... Phase A training ...
        model.unfreeze_all()
        groups = model.param_groups(lr_backbone=1e-5, lr_head=1e-3)
        # ... Phase B training ...
    """

    def __init__(
        self,
        backbone    : str  = "efficientnet_b0",
        num_classes : int  = 100,
        pretrained  : bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes   = num_classes
        self.model         = build_transfer_model(backbone, num_classes, pretrained)
        self._head_attr    = _find_head_attr(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — returns raw logits [B, num_classes]."""
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the pooled feature vector before the classification head.
        Useful for t-SNE / UMAP visualisation and feature-based analysis.

        Uses timm's forward_features() which returns the backbone output
        after global average pooling, giving a 1D feature vector per image.

        Args:
            x: Input tensor [B, 3, H, W].

        Returns:
            Feature tensor [B, D] where D is the backbone's feature dimension
            (e.g. 1280 for EfficientNet-B0, 2048 for ResNet-50).
        """
        with torch.no_grad():
            # timm's forward_features returns spatial feature maps
            feats = self.model.forward_features(x)   # [B, D, H', W'] or [B, D]
            # Global average pool if spatial dims remain
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])        # [B, D]
        return feats

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax class probabilities.

        Args:
            x: Input tensor [B, 3, H, W].

        Returns:
            Probability tensor [B, num_classes], sums to 1 along dim=1.
        """
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def freeze_backbone(self) -> "TransferModelWrapper":
        """Freeze backbone, keep head trainable. Returns self for chaining."""
        freeze_backbone(self.model)
        return self

    def unfreeze_all(self) -> "TransferModelWrapper":
        """Unfreeze all parameters. Returns self for chaining."""
        unfreeze_all(self.model)
        return self

    def param_groups(
        self,
        lr_backbone  : float = 1e-5,
        lr_head      : float = 1e-3,
        weight_decay : float = 1e-4,
    ) -> list[dict]:
        """
        Return discriminative LR param groups for Phase B optimiser.
        Shortcut for get_param_groups(self.model, ...).
        """
        return get_param_groups(
            self.model, lr_backbone, lr_head, weight_decay)

    def __repr__(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        frozen    = total - trainable
        return (
            f"TransferModelWrapper(\n"
            f"  backbone        : {self.backbone_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  head attribute  : {self._head_attr}\n"
            f"  total params    : {total:,}\n"
            f"  trainable params: {trainable:,}\n"
            f"  frozen params   : {frozen:,}\n"
            f")"
        )


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import DEVICE, TRANSFER

    print("=" * 62)
    print("  src/models/transfer_model.py — smoke test")
    print("=" * 62)

    # ── Backbone table ─────────────────────────────────────────────────────
    print_backbone_table()

    # ── Build model ────────────────────────────────────────────────────────
    backbone = TRANSFER["backbone"]    # "efficientnet_b0" from config
    print(f"\n── Building '{backbone}' (pretrained=True) ──────────────────")
    model = build_transfer_model(
        backbone    = backbone,
        num_classes = 100,
        pretrained  = True,
    ).to(DEVICE)

    # ── Forward pass check ─────────────────────────────────────────────────
    print(f"\n── Forward pass check ───────────────────────────────────────")
    dummy_224 = torch.randn(4, 3, 224, 224).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_224)
    assert logits.shape == (4, 100), \
        f"Expected (4, 100), got {logits.shape}"
    print(f"  Input  : {tuple(dummy_224.shape)}")
    print(f"  Output : {tuple(logits.shape)}  ✓")

    # ── Phase A: freeze backbone ───────────────────────────────────────────
    print(f"\n── Phase A: freeze_backbone() ───────────────────────────────")
    freeze_backbone(model)
    trainable_A = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    frozen_A    = sum(p.numel() for p in model.parameters()
                      if not p.requires_grad)
    assert frozen_A > 0,    "Backbone should be frozen"
    assert trainable_A > 0, "Head should still be trainable"
    print(f"  Frozen    : {frozen_A:,}")
    print(f"  Trainable : {trainable_A:,}")

    # Verify backbone truly frozen: gradients should only flow to head
    model.train()
    loss = F.cross_entropy(model(dummy_224), torch.randint(0, 100, (4,)).to(DEVICE))
    loss.backward()
    backbone_grad_max = max(
        p.grad.abs().max().item()
        for name, p in model.named_parameters()
        if not _is_head_param(name) and p.grad is not None
    ) if any(p.grad is not None for name, p in model.named_parameters()
             if not _is_head_param(name)) else 0.0
    print(f"  Backbone max gradient : {backbone_grad_max:.6f}  "
          f"({'OK — no gradient flow' if backbone_grad_max == 0.0 else 'WARNING — gradients leaking'})")

    # ── Phase B: unfreeze all ──────────────────────────────────────────────
    print(f"\n── Phase B: unfreeze_all() ──────────────────────────────────")
    model.zero_grad()
    unfreeze_all(model)
    total_B     = sum(p.numel() for p in model.parameters())
    trainable_B = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    assert trainable_B == total_B, "All params should be trainable after unfreeze"
    print(f"  All {trainable_B:,} params trainable  ✓")

    # ── Discriminative LR param groups ────────────────────────────────────
    print(f"\n── get_param_groups() ───────────────────────────────────────")
    groups = get_param_groups(
        model,
        lr_backbone  = TRANSFER["lr_backbone"],
        lr_head      = TRANSFER["lr_head"],
        weight_decay = TRANSFER["weight_decay"],
    )
    assert len(groups) == 2, "Should produce exactly 2 param groups"
    opt = torch.optim.AdamW(groups)
    print(f"  Optimiser created with {len(opt.param_groups)} param groups  ✓")
    for g in opt.param_groups:
        print(f"    name='{g['name']}'  lr={g['lr']:.1e}  "
              f"params={sum(p.numel() for p in g['params']):,}")

    # ── Full fine-tune gradient check ──────────────────────────────────────
    print(f"\n── Full fine-tune gradient check ────────────────────────────")
    model.train()
    opt.zero_grad()
    loss2 = F.cross_entropy(
        model(dummy_224), torch.randint(0, 100, (4,)).to(DEVICE))
    loss2.backward()

    zero_layers = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
        and p.grad.abs().max().item() == 0.0
    ]
    if zero_layers:
        print(f"  WARNING — zero gradients in {len(zero_layers)} layers:")
        for z in zero_layers[:5]:
            print(f"    {z}")
    else:
        print(f"  All trainable layers have non-zero gradients  ✓")
    print(f"  Loss: {loss2.item():.4f}")

    # ── TransferModelWrapper ───────────────────────────────────────────────
    print(f"\n── TransferModelWrapper ─────────────────────────────────────")
    wrapper = TransferModelWrapper(
        backbone    = backbone,
        num_classes = 100,
        pretrained  = True,
    ).to(DEVICE)

    wrapper.freeze_backbone()

    wrapper.eval()
    with torch.no_grad():
        feats = wrapper.extract_features(dummy_224)
        proba = wrapper.predict_proba(dummy_224)

    print(f"  Feature shape : {tuple(feats.shape)}")   # [4, 1280] for EfficientNet-B0
    print(f"  Proba shape   : {tuple(proba.shape)}")   # [4, 100]
    assert abs(proba.sum(dim=1).mean().item() - 1.0) < 1e-5, \
        "Probabilities must sum to 1"
    print(f"  Proba sum     : {proba.sum(dim=1).mean():.6f}  ✓")

    wrapper.unfreeze_all()
    groups_w = wrapper.param_groups(lr_backbone=1e-5, lr_head=1e-3)
    opt_w    = torch.optim.AdamW(groups_w)
    print(f"  Wrapper param groups : {len(opt_w.param_groups)}  ✓")
    print(wrapper)

    # ── count_parameters ──────────────────────────────────────────────────
    print(f"\n── count_parameters() ───────────────────────────────────────")
    counts = count_parameters(wrapper)

    print("\n[transfer_model] All smoke tests passed.")