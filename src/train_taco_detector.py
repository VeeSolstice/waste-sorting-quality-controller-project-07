"""Train a Faster R-CNN detector on the local Policy B TACO cohort.

Only train and validation manifest rows are opened here. The held-out test
split is deliberately reserved for a later, one-time final evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from taco_detection_dataset import TacoDetectionDataset, detection_collate_fn


CLASS_TO_INDEX = {
    "cardboard": 1,
    "glass": 2,
    "metal": 3,
    "paper": 4,
    "plastic": 5,
    "other": 6,
}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}
ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
PRETRAINED_WEIGHTS = "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"
NUM_CLASSES_WITH_BACKGROUND = 7
DEFAULT_ROOT = Path("data/taco_detection")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without promising bitwise GPU identity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(*, pretrained: bool, detector_max_size: int) -> torch.nn.Module:
    """Build the named detector and replace its COCO predictor with 7 outputs."""
    weights = (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        if pretrained
        else None
    )
    try:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            weights_backbone=None,
        )
    except Exception as error:
        if pretrained:
            raise RuntimeError(
                "Could not load the named pretrained COCO weights "
                f"{PRETRAINED_WEIGHTS}. Check the network/cache and retry; "
                "training will not silently continue with random weights."
            ) from error
        raise

    input_features = model.roi_heads.box_predictor.cls_score.in_features
    # The replacement head is randomly initialized. Its project labels differ
    # from the original COCO predictor labels.
    model.roi_heads.box_predictor = FastRCNNPredictor(
        input_features, NUM_CLASSES_WITH_BACKGROUND
    )
    model.transform.min_size = (min(320, detector_max_size),)
    model.transform.max_size = detector_max_size
    return model


def _xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, x2 - x1, y2 - y1]


def evaluate_coco_ap50(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Calculate bbox AP50 with the official COCO evaluator.

    Prediction and ground-truth boxes must both be xyxy in the same resized
    image coordinate system. Classes with no ground-truth support return null.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    if not targets:
        raise ValueError("AP50 requires at least one validation target")

    images: list[dict[str, int]] = []
    annotations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    support: Counter[int] = Counter()
    annotation_id = 1

    for prediction, target in zip(predictions, targets):
        image_id = int(target["image_id"].reshape(-1)[0])
        height, width = (int(value) for value in target["size"].tolist())
        images.append({"id": image_id, "height": height, "width": width})

        gt_boxes = target["boxes"].detach().cpu()
        gt_labels = target["labels"].detach().cpu()
        gt_crowds = target.get(
            "iscrowd", torch.zeros(len(gt_boxes), dtype=torch.int64)
        ).detach().cpu()
        for box, label, iscrowd in zip(gt_boxes, gt_labels, gt_crowds):
            xywh = _xyxy_to_xywh(box.tolist())
            category_id = int(label)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": int(iscrowd),
                }
            )
            support[category_id] += 1
            annotation_id += 1

        pred_boxes = prediction["boxes"].detach().cpu()
        pred_labels = prediction["labels"].detach().cpu()
        pred_scores = prediction["scores"].detach().cpu()
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": _xyxy_to_xywh(box.tolist()),
                    "score": float(score),
                }
            )

    categories = [
        {"id": category_id, "name": INDEX_TO_CLASS[category_id]}
        for category_id in sorted(INDEX_TO_CLASS)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        ground_truth.createIndex()
        if results:
            detections = ground_truth.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {
                "images": images,
                "annotations": [],
                "categories": categories,
            }
            detections.createIndex()

        evaluator = COCOeval(ground_truth, detections, iouType="bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.params.catIds = sorted(INDEX_TO_CLASS)
        evaluator.params.iouThrs = np.array([0.5], dtype=np.float64)
        evaluator.evaluate()
        evaluator.accumulate()

    # Shape: IoU thresholds, recall thresholds, classes, areas, max detections.
    precision = evaluator.eval["precision"]
    all_valid = precision[0, :, :, 0, -1]
    valid_values = all_valid[all_valid > -1]
    overall = float(np.mean(valid_values)) if valid_values.size else 0.0

    per_class: dict[str, dict[str, Any]] = {}
    for class_position, category_id in enumerate(evaluator.params.catIds):
        class_values = precision[0, :, class_position, 0, -1]
        class_valid = class_values[class_values > -1]
        per_class[INDEX_TO_CLASS[category_id]] = {
            "ap50": float(np.mean(class_valid)) if class_valid.size else None,
            "support": int(support[category_id]),
        }
    return {"overall_ap50": overall, "per_class_ap50": per_class}


def _move_targets(
    targets: Iterable[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            key: value.to(device)
            for key, value in target.items()
            if key in {"boxes", "labels", "image_id", "area", "iscrowd"}
        }
        for target in targets
    ]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    totals: Counter[str] = Counter()
    batches = 0
    for batch_number, (images, targets) in enumerate(loader, start=1):
        if max_batches is not None and batch_number > max_batches:
            break
        device_images = [image.to(device) for image in images]
        device_targets = _move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss_components = model(device_images, device_targets)
                total_loss = sum(loss_components.values())
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite training loss: {total_loss.item()}")
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory. No experiment setting was changed. "
                    "Retry explicitly with --batch-size 1 and/or a smaller "
                    "--max-long-side value."
                ) from error
            raise

        totals["total_loss"] += float(total_loss.detach())
        for name, value in loss_components.items():
            totals[name] += float(value.detach())
        batches += 1

    if batches == 0:
        raise RuntimeError("No training batches were processed")
    result = {name: value / batches for name, value in sorted(totals.items())}
    result["batches"] = batches
    return result


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_for_metric: list[dict[str, torch.Tensor]] = []
    for batch_number, (images, targets) in enumerate(loader, start=1):
        if max_batches is not None and batch_number > max_batches:
            break
        device_images = [image.to(device) for image in images]
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(device_images)
        predictions.extend(
            {
                key: value.detach().cpu()
                for key, value in output.items()
                if key in {"boxes", "labels", "scores"}
            }
            for output in outputs
        )
        targets_for_metric.extend(
            {key: value.detach().cpu() for key, value in target.items()}
            for target in targets
        )
    return evaluate_coco_ap50(predictions, targets_for_metric)


def checkpoint_paths(output_dir: Path, smoke_test: bool) -> tuple[Path, Path]:
    suffix = "smoke" if smoke_test else "best"
    return (
        output_dir / f"{ARCHITECTURE}_{suffix}.pt",
        output_dir / f"{ARCHITECTURE}_{suffix}_history.json",
    )


def _checkpoint_metadata(
    args: argparse.Namespace,
    annotation_path: Path,
    manifest_path: Path,
    train_images: int,
    validation_images: int,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "new_box_predictor_untrained_at_start": True,
        "class_to_index": CLASS_TO_INDEX,
        "background_index": 0,
        "policy": "B: five materials plus other annotated waste",
        "resize": {
            "cpu_rule": "EXIF transpose, RGB, aspect-preserving downscale only; no crop",
            "maximum_long_side": args.max_long_side,
            "model_min_side": min(320, args.max_long_side),
            "model_max_side": args.max_long_side,
        },
        "seed": args.seed,
        "optimizer": {
            "name": "SGD",
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        },
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "annotation_sha256": sha256_file(annotation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_image_count": train_images,
        "validation_image_count": validation_images,
        "test_split_loaded": False,
        "smoke_test": bool(args.smoke_test),
    }


@torch.no_grad()
def strict_reload_and_sanity_check(
    checkpoint_path: Path,
    dataset: TacoDetectionDataset,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resize_limit = int(checkpoint["metadata"]["resize"]["maximum_long_side"])
    model = build_model(pretrained=False, detector_max_size=resize_limit).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    image, target = dataset[0]
    output = model([image.to(device)])[0]
    if not all(key in output for key in ("boxes", "labels", "scores")):
        raise RuntimeError("Reloaded detector returned an incomplete prediction")
    if not torch.isfinite(output["boxes"]).all() or not torch.isfinite(
        output["scores"]
    ).all():
        raise RuntimeError("Reloaded detector returned non-finite values")
    return {
        "image_id": int(target["image_id"].item()),
        "detections": int(len(output["boxes"])),
        "top_label": int(output["labels"][0]) if len(output["labels"]) else None,
        "top_score": float(output["scores"][0]) if len(output["scores"]) else None,
        "strict_state_dict_load": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ROOT / "annotations_policy_b.json",
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_ROOT / "images")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT / "manifests" / "dataset_splits.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT / "training"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--max-long-side", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-train-batches", type=int, default=3)
    parser.add_argument("--smoke-validation-batches", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_long_side < 1:
        parser.error("epochs, batch-size, and max-long-side must be positive")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_dataset = TacoDetectionDataset(
        args.annotations,
        args.images,
        args.manifest,
        split="train",
        max_long_side=args.max_long_side,
    )
    validation_dataset = TacoDetectionDataset(
        args.annotations,
        args.images,
        args.manifest,
        split="validation",
        max_long_side=args.max_long_side,
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Train and validation splits must both contain images")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    print(f"device={device} amp={amp_enabled} seed={args.seed}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(
            f"gpu={properties.name} total_memory_gib="
            f"{properties.total_memory / 1024**3:.2f}"
        )
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"train_images={len(train_dataset)} validation_images={len(validation_dataset)} "
        "test_images_loaded=0"
    )
    print(
        f"architecture={ARCHITECTURE} weights={PRETRAINED_WEIGHTS} "
        "new_7_output_head=untrained"
    )
    print(
        f"batch_size={args.batch_size} lr={args.learning_rate} "
        f"momentum={args.momentum} weight_decay={args.weight_decay} "
        f"max_long_side={args.max_long_side}"
    )

    model = build_model(pretrained=True, detector_max_size=args.max_long_side).to(
        device
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, history_path = checkpoint_paths(args.output_dir, args.smoke_test)
    metadata = _checkpoint_metadata(
        args,
        args.annotations,
        args.manifest,
        len(train_dataset),
        len(validation_dataset),
    )
    history: list[dict[str, Any]] = []
    best_ap50 = -math.inf
    epochs = 1 if args.smoke_test else args.epochs
    train_limit = args.smoke_train_batches if args.smoke_test else None
    validation_limit = args.smoke_validation_batches if args.smoke_test else None

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            max_batches=train_limit,
        )
        metrics = validate(
            model,
            validation_loader,
            device,
            amp_enabled,
            max_batches=validation_limit,
        )
        elapsed = time.perf_counter() - start
        epoch_record = {
            "epoch": epoch,
            "seconds": elapsed,
            "training_losses": losses,
            "validation": metrics,
        }
        history.append(epoch_record)
        print(f"epoch={epoch} seconds={elapsed:.2f}")
        print("training_losses=" + json.dumps(losses, sort_keys=True))
        print("validation=" + json.dumps(metrics, sort_keys=True))

        if metrics["overall_ap50"] > best_ap50:
            best_ap50 = float(metrics["overall_ap50"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": metrics,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        history_path.write_text(
            json.dumps({"metadata": metadata, "epochs": history}, indent=2),
            encoding="utf-8",
        )

    if device.type == "cuda":
        print(
            f"peak_gpu_allocated_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.3f} "
            f"peak_gpu_reserved_gib={torch.cuda.max_memory_reserved(device) / 1024**3:.3f}"
        )
    sanity = strict_reload_and_sanity_check(
        checkpoint_path, validation_dataset, device
    )
    print(f"best_checkpoint={checkpoint_path} best_validation_ap50={best_ap50:.6f}")
    print("reload_sanity=" + json.dumps(sanity, sort_keys=True))
    print(f"history={history_path}")


if __name__ == "__main__":
    main()
