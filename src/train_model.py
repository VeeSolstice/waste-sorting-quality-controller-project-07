"""Fine-tune ResNet-18 using only the train and validation manifest rows."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from model_config import (
    CHECKPOINT_PATH,
    CLASS_TO_INDEX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    LEARNING_RATE,
    PREPROCESSING_INFO,
    RANDOM_SEED,
    SMOKE_CHECKPOINT_PATH,
)
from waste_dataset import DatasetError, create_training_loaders
from waste_model import (
    ARCHITECTURE_NAME,
    PRETRAINED_WEIGHTS_NAME,
    PretrainedWeightsError,
    create_training_model,
    keep_frozen_layers_in_eval_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one epoch with two train and two validation batches, writing a "
            "separate smoke-test checkpoint."
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def set_random_seeds(seed: int) -> None:
    """Seed Python and PyTorch and request deterministic CUDA behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1_score(
    predictions: list[int], targets: list[int], class_count: int
) -> float:
    """Calculate unweighted mean per-class F1 directly from predictions."""
    class_scores: list[float] = []
    for class_index in range(class_count):
        true_positives = sum(
            prediction == class_index and target == class_index
            for prediction, target in zip(predictions, targets)
        )
        false_positives = sum(
            prediction == class_index and target != class_index
            for prediction, target in zip(predictions, targets)
        )
        false_negatives = sum(
            prediction != class_index and target == class_index
            for prediction, target in zip(predictions, targets)
        )
        denominator = 2 * true_positives + false_positives + false_negatives
        class_scores.append(
            0.0 if denominator == 0 else 2 * true_positives / denominator
        )
    return sum(class_scores) / len(class_scores)


def train_one_epoch(
    model: nn.Module,
    data_loader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    maximum_batches: int | None = None,
) -> float:
    model.train()
    keep_frozen_layers_in_eval_mode(model)
    total_loss = 0.0
    example_count = 0

    for batch_number, (images, labels) in enumerate(data_loader, start=1):
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
        if maximum_batches is not None and batch_number >= maximum_batches:
            break

    return total_loss / example_count


def validate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    maximum_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(data_loader, start=1):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            targets.extend(labels.tolist())
            if maximum_batches is not None and batch_number >= maximum_batches:
                break

    correct = sum(
        prediction == target
        for prediction, target in zip(predictions, targets)
    )
    accuracy = correct / len(targets)
    macro_f1 = macro_f1_score(predictions, targets, len(CLASS_TO_INDEX))
    return accuracy, macro_f1


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    validation_accuracy: float,
    validation_macro_f1: float,
    batch_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": ARCHITECTURE_NAME,
        "pretrained_weights": PRETRAINED_WEIGHTS_NAME,
        "class_to_index": CLASS_TO_INDEX,
        "preprocessing": PREPROCESSING_INFO,
        "epoch": epoch,
        "validation_metrics": {
            "accuracy": validation_accuracy,
            "macro_f1": validation_macro_f1,
        },
        "training": {
            "random_seeds": {
                "python": RANDOM_SEED,
                "torch": RANDOM_SEED,
                "cuda": RANDOM_SEED,
            },
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "training_split": "train",
            "checkpoint_selection_split": "validation",
        },
    }
    torch.save(checkpoint, path)


def run_training(args: argparse.Namespace) -> None:
    set_random_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1 if args.smoke_test else args.epochs
    maximum_batches = 2 if args.smoke_test else None
    checkpoint_path = SMOKE_CHECKPOINT_PATH if args.smoke_test else CHECKPOINT_PATH

    train_loader, validation_loader = create_training_loaders(
        args.batch_size, RANDOM_SEED
    )
    model = create_training_model().to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Pretrained weights: {PRETRAINED_WEIGHTS_NAME}")
    print(
        f"Manifest rows loaded: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)} (test is not loaded)"
    )
    if args.smoke_test:
        print("Smoke test: two train batches and two validation batches")

    best_macro_f1 = -1.0
    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            maximum_batches,
        )
        validation_accuracy, validation_macro_f1 = validate(
            model, validation_loader, device, maximum_batches
        )
        print(
            f"Epoch {epoch}/{epochs} - train loss: {training_loss:.4f} - "
            f"validation accuracy: {validation_accuracy:.4f} - "
            f"validation macro F1: {validation_macro_f1:.4f}"
        )

        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                validation_accuracy,
                validation_macro_f1,
                args.batch_size,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    print(f"Best validation macro F1: {best_macro_f1:.4f}")


def main() -> None:
    args = parse_args()
    try:
        run_training(args)
    except (DatasetError, PretrainedWeightsError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
