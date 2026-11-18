"""Train the small CNN baseline from scratch on train/validation rows."""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from classification_metrics import classification_metrics
from model_config import CLASS_NAMES, CLASS_TO_INDEX
from small_cnn import (
    SMALL_CNN_ARCHITECTURE,
    SMALL_CNN_BATCH_SIZE,
    SMALL_CNN_CHECKPOINT_PATH,
    SMALL_CNN_EPOCHS,
    SMALL_CNN_LEARNING_RATE,
    SMALL_CNN_PREPROCESSING,
    SMALL_CNN_SEED,
    SmallCNN,
    small_cnn_inference_transform,
    small_cnn_training_transform,
)
from waste_dataset import DatasetError, WasteManifestDataset


def set_random_seeds(seed: int) -> None:
    """Seed model initialization, augmentation, shuffling, and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_loaders() -> tuple[DataLoader, DataLoader]:
    """Load train and validation rows only; never instantiate the test split."""
    train_dataset = WasteManifestDataset(
        "train", small_cnn_training_transform()
    )
    validation_dataset = WasteManifestDataset(
        "validation", small_cnn_inference_transform()
    )
    shuffle_generator = torch.Generator().manual_seed(SMALL_CNN_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=SMALL_CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=SMALL_CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    example_count = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        example_count += batch_size

    return total_loss / example_count


def validate(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    metrics = classification_metrics(targets, predictions, CLASS_NAMES)
    return metrics["accuracy"], metrics["macro_f1"]


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    validation_accuracy: float,
    validation_macro_f1: float,
) -> None:
    SMALL_CNN_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": SMALL_CNN_ARCHITECTURE,
        "pretrained_weights": None,
        "initialization": "random_from_scratch",
        "class_to_index": CLASS_TO_INDEX,
        "preprocessing": SMALL_CNN_PREPROCESSING,
        "epoch": epoch,
        "validation_metrics": {
            "accuracy": validation_accuracy,
            "macro_f1": validation_macro_f1,
        },
        "training": {
            "random_seeds": {
                "python": SMALL_CNN_SEED,
                "torch": SMALL_CNN_SEED,
                "cuda": SMALL_CNN_SEED,
            },
            "epochs_requested": SMALL_CNN_EPOCHS,
            "batch_size": SMALL_CNN_BATCH_SIZE,
            "learning_rate": SMALL_CNN_LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "training_split": "train",
            "checkpoint_selection_split": "validation",
        },
    }
    torch.save(checkpoint, SMALL_CNN_CHECKPOINT_PATH)


def run_training() -> None:
    set_random_seeds(SMALL_CNN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = create_loaders()

    # SmallCNN uses PyTorch's seeded random initialization; no weights are loaded.
    model = SmallCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SMALL_CNN_LEARNING_RATE
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print("Model: SmallCNN (random initialization; trained from scratch)")
    print(f"Parameters: {parameter_count}")
    print(f"Random seed: {SMALL_CNN_SEED}")
    print(
        f"Optimizer: Adam, learning rate: {SMALL_CNN_LEARNING_RATE}, "
        f"batch size: {SMALL_CNN_BATCH_SIZE}, loss: CrossEntropyLoss"
    )
    print(f"Preprocessing: {SMALL_CNN_PREPROCESSING}")
    print(
        f"Manifest rows loaded: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)} (test is not loaded)"
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, SMALL_CNN_EPOCHS + 1):
        training_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_accuracy, validation_macro_f1 = validate(
            model, validation_loader, device
        )
        print(
            f"Epoch {epoch}/{SMALL_CNN_EPOCHS} - "
            f"train loss: {training_loss:.4f} - "
            f"validation accuracy: {validation_accuracy:.4f} - "
            f"validation macro F1: {validation_macro_f1:.4f}"
        )

        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            save_checkpoint(
                model, epoch, validation_accuracy, validation_macro_f1
            )
            print(f"Saved best checkpoint: {SMALL_CNN_CHECKPOINT_PATH}")

    print(
        f"Best validation macro F1: {best_macro_f1:.4f} "
        f"(epoch {best_epoch})"
    )


def main() -> None:
    try:
        run_training()
    except DatasetError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
