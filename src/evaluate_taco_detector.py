"""Select a TACO detector by validation AP50, then evaluate test exactly once.

The ``select`` command reads checkpoint metadata only and writes a decision
record before any test image is opened. The ``evaluate`` command accepts only
that fixed record, verifies hashes, and evaluates its selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from taco_detection_dataset import TacoDetectionDataset, detection_collate_fn
from train_taco_detector import (
    CLASS_TO_INDEX,
    evaluate_coco_bbox_metrics,
    load_checkpoint_model,
    sha256_file,
)


TIE_RULE = (
    "Select the highest saved validation overall AP50; an exact numeric tie "
    "selects the checkpoint with the lexicographically smaller SHA-256 hash."
)
DEFAULT_ROOT = Path("data/taco_detection")


def checkpoint_summary(path: Path) -> dict[str, Any]:
    """Read validation-selection evidence without constructing a model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    validation = checkpoint.get("validation")
    if not isinstance(metadata, dict) or not isinstance(validation, dict):
        raise ValueError(f"Checkpoint lacks metadata or validation metrics: {path}")
    if "overall_ap50" not in validation:
        raise ValueError(f"Checkpoint lacks validation overall_ap50: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "experiment_name": metadata.get("experiment_name", "legacy_320"),
        "best_epoch": int(checkpoint["epoch"]),
        "validation_ap50": float(validation["overall_ap50"]),
        "annotation_sha256": metadata["annotation_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "architecture": metadata["architecture"],
        "pretrained_weights": metadata["pretrained_weights"],
    }


def select_checkpoint_summaries(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the documented validation-only checkpoint rule."""
    if len(candidates) < 2:
        raise ValueError("Model selection requires at least two candidates")
    annotation_hashes = {item["annotation_sha256"] for item in candidates}
    manifest_hashes = {item["manifest_sha256"] for item in candidates}
    if len(annotation_hashes) != 1 or len(manifest_hashes) != 1:
        raise ValueError("Candidate checkpoints do not use the same data hashes")
    ranked = sorted(
        candidates,
        key=lambda item: (-float(item["validation_ap50"]), item["sha256"]),
    )
    return ranked[0]


def write_selection_record(
    checkpoint_paths: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    """Hash candidates and save the final decision before test data is loaded."""
    if output_path.exists():
        raise FileExistsError(
            f"Selection record already exists and will not be overwritten: {output_path}"
        )
    candidates = [checkpoint_summary(path) for path in checkpoint_paths]
    selected = select_checkpoint_summaries(candidates)
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "saved validation overall AP50 only",
        "tie_rule": TIE_RULE,
        "candidates": candidates,
        "selected_checkpoint_path": selected["path"],
        "selected_checkpoint_sha256": selected["sha256"],
        "selected_experiment_name": selected["experiment_name"],
        "selected_validation_ap50": selected["validation_ap50"],
        "selected_best_epoch": selected["best_epoch"],
        "annotation_sha256": selected["annotation_sha256"],
        "manifest_sha256": selected["manifest_sha256"],
        "test_images_loaded_during_selection": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def cpu_max_long_side_from_metadata(metadata: dict[str, Any]) -> int:
    """Read the saved CPU resize rule from current or legacy checkpoints."""
    resize = metadata["resize"]
    if "cpu" in resize:
        return int(resize["cpu"]["maximum_long_side"])
    return int(resize["maximum_long_side"])


def validate_manifest_split_isolation(manifest_path: Path) -> dict[str, set[int]]:
    """Require every image ID to belong to exactly one named manifest split."""
    split_ids: dict[str, set[int]] = {}
    owner: dict[int, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            image_id = int(row["image_id"])
            split = row["split"]
            previous = owner.get(image_id)
            if previous is not None:
                raise ValueError(
                    f"Manifest image {image_id} is repeated on row {row_number}; "
                    f"first split was {previous!r}, repeated split is {split!r}"
                )
            owner[image_id] = split
            split_ids.setdefault(split, set()).add(image_id)
    required = {"train", "validation", "test"}
    if not required.issubset(split_ids):
        raise ValueError(f"Manifest must contain splits {sorted(required)}")
    return split_ids


def _tensor_boxes(
    boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if scores is None:
        score_values: Sequence[float | None] = [None] * len(boxes)
    else:
        score_values = [float(value) for value in scores]
    index_to_class = {value: key for key, value in CLASS_TO_INDEX.items()}
    for box, label, score in zip(boxes, labels, score_values):
        record: dict[str, Any] = {
            "box_xyxy": [float(value) for value in box.tolist()],
            "label": int(label),
            "class_name": index_to_class[int(label)],
        }
        if score is not None:
            record["score"] = score
        records.append(record)
    return records


def detector_postprocessing_metadata(model: torch.nn.Module) -> dict[str, Any]:
    """Distinguish Torchvision's internal filter from application filtering."""
    return {
        "model_internal_score_threshold": float(model.roi_heads.score_thresh),
        "additional_application_score_threshold": None,
        "model_maximum_detections_per_image": int(
            model.roi_heads.detections_per_img
        ),
    }


@torch.no_grad()
def run_fixed_test_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the frozen decision, then load and evaluate only test images."""
    metrics_path = args.output_dir / "taco_detector_test_metrics.json"
    predictions_path = args.output_dir / "taco_detector_test_predictions.json"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError(
            "Final test outputs already exist. Refusing to rerun or overwrite the "
            "one-time evaluation."
        )

    selection_bytes_hash = sha256_file(args.selection_record)
    selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
    if selection.get("test_images_loaded_during_selection") is not False:
        raise ValueError("Selection record does not certify validation-only selection")
    checkpoint_path = Path(selection["selected_checkpoint_path"])
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != selection["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash no longer matches the decision record")

    model, checkpoint = load_checkpoint_model(checkpoint_path, args.device)
    metadata = checkpoint["metadata"]
    annotation_hash = sha256_file(args.annotations)
    manifest_hash = sha256_file(args.manifest)
    if annotation_hash != metadata["annotation_sha256"]:
        raise ValueError("Annotation hash does not match the selected checkpoint")
    if manifest_hash != metadata["manifest_sha256"]:
        raise ValueError("Manifest hash does not match the selected checkpoint")
    if annotation_hash != selection["annotation_sha256"]:
        raise ValueError("Annotation hash does not match the selection record")
    if manifest_hash != selection["manifest_sha256"]:
        raise ValueError("Manifest hash does not match the selection record")

    splits = validate_manifest_split_isolation(args.manifest)
    dataset = TacoDetectionDataset(
        args.annotations,
        args.images,
        args.manifest,
        split="test",
        max_long_side=cpu_max_long_side_from_metadata(metadata),
    )
    if len(dataset) != len(splits["test"]):
        raise ValueError("Derived test dataset does not match manifest test IDs")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    amp_enabled = bool(metadata.get("amp", False) and args.device.type == "cuda")
    predictions: list[dict[str, torch.Tensor]] = []
    targets_for_metric: list[dict[str, torch.Tensor]] = []
    prediction_records: list[dict[str, Any]] = []
    file_name_by_id = {
        int(record["id"]): record["file_name"] for record in dataset.images
    }
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)

    for images, targets in loader:
        with torch.amp.autocast(
            device_type=args.device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model([image.to(args.device) for image in images])
        for output, target in zip(outputs, targets):
            cpu_output = {
                key: value.detach().cpu()
                for key, value in output.items()
                if key in {"boxes", "labels", "scores"}
            }
            cpu_target = {
                key: value.detach().cpu() for key, value in target.items()
            }
            predictions.append(cpu_output)
            targets_for_metric.append(cpu_target)
            image_id = int(cpu_target["image_id"].item())
            prediction_records.append(
                {
                    "image_id": image_id,
                    "file_name": file_name_by_id[image_id],
                    "resized_size_hw": [int(value) for value in cpu_target["size"]],
                    "ground_truth": _tensor_boxes(
                        cpu_target["boxes"], cpu_target["labels"]
                    ),
                    "predictions": _tensor_boxes(
                        cpu_output["boxes"],
                        cpu_output["labels"],
                        cpu_output["scores"],
                    ),
                }
            )

    metrics = evaluate_coco_bbox_metrics(predictions, targets_for_metric)
    postprocessing = detector_postprocessing_metadata(model)
    metrics["evaluator"].update(postprocessing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        json.dumps(prediction_records, indent=2), encoding="utf-8"
    )
    result = {
        "schema_version": 1,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "exact_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "selection_record_path": str(args.selection_record.resolve()),
        "selection_record_sha256": selection_bytes_hash,
        "selected_checkpoint_path": str(checkpoint_path.resolve()),
        "selected_checkpoint_sha256": actual_checkpoint_hash,
        "selected_validation_ap50": selection["selected_validation_ap50"],
        "annotation_sha256": annotation_hash,
        "manifest_sha256": manifest_hash,
        "split": "test",
        "test_image_count": len(dataset),
        **postprocessing,
        "predictions_used_after_internal_model_filter_only": True,
        "cpu_max_long_side": cpu_max_long_side_from_metadata(metadata),
        "model_transform": metadata["resize"].get("model_transform"),
        "amp": amp_enabled,
        "device": str(args.device),
        "metrics": metrics,
        "predictions_path": str(predictions_path.resolve()),
        "peak_gpu_allocated_gib": (
            torch.cuda.max_memory_allocated(args.device) / 1024**3
            if args.device.type == "cuda"
            else None
        ),
        "test_results_used_for_model_selection": False,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    select_parser.add_argument("--selection-record", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--selection-record", type=Path, required=True)
    evaluate_parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ROOT / "annotations_policy_b.json",
    )
    evaluate_parser.add_argument(
        "--images", type=Path, default=DEFAULT_ROOT / "images"
    )
    evaluate_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT / "manifests" / "dataset_splits.csv",
    )
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=1)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.device == "auto":
            args.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            args.device = torch.device(args.device)
        if args.device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        if args.batch_size < 1 or args.num_workers < 0:
            parser.error("batch size must be positive and num-workers nonnegative")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "select":
        record = write_selection_record(args.candidate, args.selection_record)
        print(json.dumps(record, indent=2))
    else:
        result = run_fixed_test_evaluation(args)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
