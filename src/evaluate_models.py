"""Evaluate validation-selected ResNet-18 and SmallCNN checkpoints once on test."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from classification_metrics import classification_metrics, most_common_confusions
from model_config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    PREPROCESSING_INFO,
    PROJECT_ROOT,
)
from small_cnn import (
    SMALL_CNN_ARCHITECTURE,
    SMALL_CNN_CHECKPOINT_PATH,
    SMALL_CNN_PREPROCESSING,
    SmallCNN,
    small_cnn_inference_transform,
)
from waste_dataset import DatasetError, WasteManifestDataset, inference_transform
from waste_model import ARCHITECTURE_NAME, create_inference_model


EVALUATION_BATCH_SIZE = 32
MODEL_CHECKPOINTS = (
    (ARCHITECTURE_NAME, CHECKPOINT_PATH),
    (SMALL_CNN_ARCHITECTURE, SMALL_CNN_CHECKPOINT_PATH),
)


class EvaluationError(Exception):
    """A clear final-evaluation configuration or checkpoint error."""


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_checkpoint_model(
    expected_architecture: str, checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any], Any]:
    """Load a model and the exact deterministic preprocessing it saved."""
    if not checkpoint_path.is_file():
        raise EvaluationError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except Exception as error:
        raise EvaluationError(
            f"Could not load checkpoint {checkpoint_path}: {error}"
        ) from error

    if checkpoint.get("architecture") != expected_architecture:
        raise EvaluationError(
            f"Expected {expected_architecture!r} in {checkpoint_path}, but found "
            f"{checkpoint.get('architecture')!r}."
        )
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise EvaluationError(
            f"Class mapping mismatch in checkpoint: {checkpoint_path}"
        )

    if expected_architecture == ARCHITECTURE_NAME:
        if checkpoint.get("preprocessing") != PREPROCESSING_INFO:
            raise EvaluationError("ResNet checkpoint preprocessing is unsupported.")
        model = create_inference_model()
        transform = inference_transform()
    elif expected_architecture == SMALL_CNN_ARCHITECTURE:
        if checkpoint.get("preprocessing") != SMALL_CNN_PREPROCESSING:
            raise EvaluationError("SmallCNN checkpoint preprocessing is unsupported.")
        if checkpoint.get("pretrained_weights") is not None:
            raise EvaluationError("SmallCNN baseline must not contain pretrained weights.")
        model = SmallCNN()
        transform = small_cnn_inference_transform()
    else:
        raise EvaluationError(f"Unsupported architecture: {expected_architecture}")

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise EvaluationError(
            f"Checkpoint weights are missing or incompatible: {error}"
        ) from error
    model.to(device).eval()
    return model, checkpoint, transform


def evaluate_checkpoint(
    model_name: str, checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    model, checkpoint, transform = load_checkpoint_model(
        model_name, checkpoint_path, device
    )
    test_dataset = WasteManifestDataset("test", transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    targets: list[int] = []
    predictions: list[int] = []
    prediction_rows: list[dict[str, str]] = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device, non_blocking=True))
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            batch_targets = labels.tolist()

            for target, prediction in zip(batch_targets, batch_predictions):
                image_path, manifest_target = test_dataset.samples[sample_index]
                if manifest_target != target:
                    raise EvaluationError("Test DataLoader order changed unexpectedly.")
                true_label = CLASS_NAMES[target]
                predicted_label = CLASS_NAMES[prediction]
                prediction_rows.append(
                    {
                        "relative_path": project_relative(image_path),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "correct": str(target == prediction).lower(),
                    }
                )
                sample_index += 1

            targets.extend(batch_targets)
            predictions.extend(batch_predictions)

    metrics = classification_metrics(targets, predictions, CLASS_NAMES)
    metrics.update(
        {
            "model": model_name,
            "checkpoint": project_relative(checkpoint_path),
            "checkpoint_selected_epoch": checkpoint.get("epoch"),
            "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
            "test_split": "test",
            "test_images": len(test_dataset),
            "confusion_matrix_orientation": (
                "Rows are true classes; columns are predicted classes."
            ),
            "class_order": list(CLASS_NAMES),
        }
    )
    return metrics, prediction_rows


def write_evaluation_outputs(
    model_name: str,
    metrics: dict[str, Any],
    prediction_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path]:
    metrics_path = PROJECT_ROOT / "artifacts" / f"{model_name}_test_metrics.json"
    matrix_path = (
        PROJECT_ROOT / "artifacts" / f"{model_name}_test_confusion_matrix.csv"
    )
    predictions_path = (
        PROJECT_ROOT / "artifacts" / f"{model_name}_test_predictions.csv"
    )

    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
        metrics_file.write("\n")

    with matrix_path.open("w", encoding="utf-8", newline="") as matrix_file:
        writer = csv.writer(matrix_file, lineterminator="\n")
        writer.writerow(("true_label", *CLASS_NAMES))
        for class_name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
            writer.writerow((class_name, *row))

    with predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as predictions_file:
        writer = csv.DictWriter(
            predictions_file,
            fieldnames=(
                "relative_path",
                "true_label",
                "predicted_label",
                "correct",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    return metrics_path, matrix_path, predictions_path


def print_results(
    model_name: str,
    metrics: dict[str, Any],
    prediction_rows: list[dict[str, str]],
    output_paths: tuple[Path, Path, Path],
) -> None:
    print(f"\n{model_name} held-out test results ({metrics['test_images']} images)")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print("  Per-class precision / recall / F1 / support:")
    for class_name in CLASS_NAMES:
        class_metrics = metrics["per_class"][class_name]
        print(
            f"    {class_name:<9} "
            f"{class_metrics['precision']:.4f} / "
            f"{class_metrics['recall']:.4f} / "
            f"{class_metrics['f1']:.4f} / "
            f"{class_metrics['support']}"
        )

    print("  Most common confusions (true -> predicted):")
    common = most_common_confusions(
        metrics["confusion_matrix"], CLASS_NAMES, limit=5
    )
    if common:
        for true_label, predicted_label, count in common:
            print(f"    {true_label} -> {predicted_label}: {count}")
    else:
        print("    none")

    print("  Example errors:")
    errors = [row for row in prediction_rows if row["correct"] == "false"][:5]
    if errors:
        for row in errors:
            print(
                f"    {row['relative_path']} "
                f"({row['true_label']} -> {row['predicted_label']})"
            )
    else:
        print("    none")

    for path in output_paths:
        print(f"  Saved: {project_relative(path)}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Final evaluation device: {device}")
    print("The test split is evaluated only; no training or checkpoint changes occur.")
    print("Confusion matrix rows are true classes; columns are predicted classes.")

    try:
        for model_name, checkpoint_path in MODEL_CHECKPOINTS:
            metrics, prediction_rows = evaluate_checkpoint(
                model_name, checkpoint_path, device
            )
            output_paths = write_evaluation_outputs(
                model_name, metrics, prediction_rows
            )
            print_results(model_name, metrics, prediction_rows, output_paths)
    except (DatasetError, EvaluationError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
