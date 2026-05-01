"""
CNN training loop with 5-fold stratified cross-validation.

Usage:
    python -m app.ml.trainer --data-dir /path/to/dataset --epochs 30
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold

from app.ml.model import SkinAnomalyCNN
from app.ml.transforms import get_train_transforms, get_val_transforms
from app.core.config import settings


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataloader. Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train_with_cross_validation(
    data_dir: str,
    epochs: int = 30,
    n_folds: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    save_path: str = "app/ml/weights/skin_cnn.pt",
) -> None:
    """
    Train SkinAnomalyCNN with 5-fold stratified cross-validation.

    - Uses data augmentation on training folds
    - Saves the best-performing model checkpoint
    - Reports per-fold and average accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = ImageFolder(data_dir, transform=get_train_transforms())
    labels = [s[1] for s in dataset.samples]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    best_accuracy = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.samples, labels), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        train_loader = DataLoader(
            dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx), num_workers=4
        )
        val_loader = DataLoader(
            dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(val_idx), num_workers=4
        )

        model = SkinAnomalyCNN(
            num_classes=settings.NUM_CLASSES, freeze_backbone=True
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_fold_acc = 0.0
        patience, patience_counter = 5, 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"  Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
            )

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                patience_counter = 0
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"  Saved best model with val_acc={val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        fold_accuracies.append(best_fold_acc)
        print(f"  Fold {fold} best acc: {best_fold_acc:.4f}")

    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\n=== Cross-Validation Results ===")
    print(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
    print(f"Average accuracy: {avg_acc:.4f}")
    print(f"Best overall accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin anomaly CNN")
    parser.add_argument("--data-dir", required=True, help="Path to ImageFolder dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", default="app/ml/weights/skin_cnn.pt")
    args = parser.parse_args()

    train_with_cross_validation(
        data_dir=args.data_dir,
        epochs=args.epochs,
        n_folds=args.folds,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
