"""
Two-phase training for the Set card multi-task classifier.

Phase 1 (warm-up): freeze the backbone, train only the four classification
    heads. This lets the heads converge quickly without corrupting the
    pretrained backbone weights.

Phase 2 (fine-tune): unfreeze the backbone and train everything with
    differential learning rates (backbone at 10× lower LR than the heads).
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from setgame.classify_card.dataset import (
    ATTRIBUTES,
    SetCardDataset,
    build_sample_list,
    make_transforms,
    train_val_split,
)
from setgame.classify_card.model import SetCardClassifier


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _accuracy(logits_dict: dict, labels_dict: dict, device: torch.device) -> dict[str, float]:
    """Return per-attribute accuracy for a single batch."""
    return {
        attr: (logits_dict[attr].argmax(1) == labels_dict[attr].to(device)).float().mean().item()
        for attr in ATTRIBUTES
    }


def _run_epoch(
    model: SetCardClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training or validation epoch.

    Parameters
    ----------
    optimizer : Optimizer or None
        Pass an optimizer to train; pass None to validate.

    Returns
    -------
    mean_loss : float
    mean_accs : dict {attribute: accuracy}
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_accs = {attr: 0.0 for attr in ATTRIBUTES}
    n_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            preds = model(images)

            loss = sum(
                criterion(preds[attr], labels[attr].to(device))
                for attr in ATTRIBUTES
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            batch_accs = _accuracy(preds, labels, device)
            for attr in ATTRIBUTES:
                total_accs[attr] += batch_accs[attr]
            n_batches += 1

    mean_accs = {attr: total_accs[attr] / n_batches for attr in ATTRIBUTES}
    return total_loss / n_batches, mean_accs


def train(
    originals_dir: str,
    labels_dir: str,
    output_path: str,
    augmented_dir: str | None = None,
    val_fraction: float = 0.2,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    phase1_epochs: int = 5,
    phase2_epochs: int = 20,
    phase1_lr: float = 1e-3,
    phase2_backbone_lr: float = 2e-5,
    phase2_heads_lr: float = 1e-4,
    seed: int = 42,
) -> SetCardClassifier:
    """Train a SetCardClassifier and save the weights to output_path.

    Parameters
    ----------
    originals_dir : str
        Directory containing original segmented card images.
    labels_dir : str
        Directory containing JSON label files (one per card).
    output_path : str
        Path where the trained model weights (.pth) will be saved.
    augmented_dir : str or None
        Parent directory of aug0/, aug1/, … subdirectories. Pass None to
        train on originals only.
    val_fraction : float
        Fraction of original images to reserve for validation.
    img_size : int
        Spatial input size (height = width) for the model.
    batch_size : int
        Batch size for both phases.
    num_workers : int
        DataLoader worker processes.
    phase1_epochs : int
        Epochs with backbone frozen (head warm-up).
    phase2_epochs : int
        Epochs with full model fine-tuning.
    phase1_lr : float
        Learning rate for classification heads during phase 1.
    phase2_backbone_lr : float
        Backbone learning rate during phase 2.
    phase2_heads_lr : float
        Head learning rate during phase 2.
    seed : int
        Random seed for the train/val split.

    Returns
    -------
    SetCardClassifier
        The trained model (on CPU).
    """
    device = _get_device()
    logger.info(f"Training on {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    all_samples = build_sample_list(originals_dir, labels_dir, augmented_dir)
    train_samples, val_samples = train_val_split(all_samples, val_fraction, seed)
    logger.info(
        f"Dataset: {len(train_samples)} train samples, {len(val_samples)} val samples"
    )

    train_ds = SetCardDataset(train_samples, transform=make_transforms(train=True,  img_size=img_size))
    val_ds   = SetCardDataset(val_samples,   transform=make_transforms(train=False, img_size=img_size))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SetCardClassifier().to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: backbone frozen, heads only ─────────────────────────────────
    logger.info(f"Phase 1: training heads only for {phase1_epochs} epochs")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer1 = torch.optim.AdamW(model.heads.parameters(), lr=phase1_lr, weight_decay=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=phase1_epochs)

    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_accs = _run_epoch(model, train_dl, criterion, device, optimizer1)
        val_loss,   val_accs   = _run_epoch(model, val_dl,   criterion, device)
        scheduler1.step()
        _log_epoch("P1", epoch, phase1_epochs, train_loss, train_accs, val_loss, val_accs)

    # ── Phase 2: full fine-tuning with differential LR ───────────────────────
    logger.info(f"Phase 2: fine-tuning all layers for {phase2_epochs} epochs")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": phase2_backbone_lr},
        {"params": model.heads.parameters(),    "lr": phase2_heads_lr},
    ], weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=phase2_epochs)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, phase2_epochs + 1):
        train_loss, train_accs = _run_epoch(model, train_dl, criterion, device, optimizer2)
        val_loss,   val_accs   = _run_epoch(model, val_dl,   criterion, device)
        scheduler2.step()
        _log_epoch("P2", epoch, phase2_epochs, train_loss, train_accs, val_loss, val_accs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save best weights ────────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    model.cpu()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Best model saved to {output_path}  (val_loss={best_val_loss:.4f})")
    return model


def _log_epoch(phase, epoch, total, train_loss, train_accs, val_loss, val_accs):
    acc_str = "  ".join(f"{a}={val_accs[a]:.3f}" for a in ATTRIBUTES)
    logger.info(
        f"[{phase} {epoch:02d}/{total}]  "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  |  {acc_str}"
    )
