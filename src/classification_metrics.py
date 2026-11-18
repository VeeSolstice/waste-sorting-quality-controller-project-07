"""Small standard-library helpers for multiclass classification metrics."""

from __future__ import annotations

from typing import Any, Sequence


def classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute a confusion matrix and macro/per-class classification metrics."""
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have the same length.")
    if not targets:
        raise ValueError("At least one prediction is required.")

    class_count = len(class_names)
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions):
        if not 0 <= target < class_count or not 0 <= prediction < class_count:
            raise ValueError("A class index is outside the configured class range.")
        matrix[target][prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for class_index, class_name in enumerate(class_names):
        true_positives = matrix[class_index][class_index]
        false_positives = (
            sum(matrix[row][class_index] for row in range(class_count))
            - true_positives
        )
        false_negatives = sum(matrix[class_index]) - true_positives
        support = sum(matrix[class_index])

        precision_denominator = true_positives + false_positives
        precision = (
            true_positives / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = true_positives / support if support else 0.0
        f1_denominator = precision + recall
        f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = sum(matrix[index][index] for index in range(class_count))
    return {
        "accuracy": correct / len(targets),
        "macro_precision": sum(precision_values) / class_count,
        "macro_recall": sum(recall_values) / class_count,
        "macro_f1": sum(f1_values) / class_count,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def most_common_confusions(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    limit: int = 5,
) -> list[tuple[str, str, int]]:
    """Return the most frequent off-diagonal true/predicted class pairs."""
    confusions = [
        (class_names[true_index], class_names[predicted_index], count)
        for true_index, row in enumerate(matrix)
        for predicted_index, count in enumerate(row)
        if true_index != predicted_index and count > 0
    ]
    return sorted(confusions, key=lambda item: (-item[2], item[0], item[1]))[
        :limit
    ]
