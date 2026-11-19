"""Minimal PyTorch Dataset for a locally prepared TACO detection cohort."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TacoDetectionDataset(Dataset):
    """Load RGB images and COCO xywh annotations as torchvision-style targets."""

    def __init__(
        self,
        annotation_path: Path,
        image_root: Path,
        manifest_path: Path | None = None,
        split: str | None = None,
    ) -> None:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = data.get("images")
        annotations = data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError("Derived COCO images and annotations must be lists")

        allowed_ids: set[int] | None = None
        if manifest_path is not None:
            with manifest_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            allowed_ids = {
                int(row["image_id"])
                for row in rows
                if split is None or row.get("split") == split
            }

        self.image_root = image_root
        self.images = [
            image
            for image in images
            if allowed_ids is None or int(image["id"]) in allowed_ids
        ]
        self.images.sort(key=lambda image: int(image["id"]))
        self.annotations_by_image: defaultdict[int, list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for annotation in annotations:
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_record = self.images[index]
        path = self.image_root / image_record["file_name"]
        with Image.open(path) as source:
            source.load()
            rgb = source.convert("RGB")
            image_array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        crowds: list[int] = []
        for annotation in self.annotations_by_image[int(image_record["id"])]:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            if width <= 0 or height <= 0:
                raise ValueError("Prepared annotations must have positive-area boxes")
            boxes.append([x, y, x + width, y + height])
            label = int(annotation["category_id"])
            if label < 1:
                raise ValueError("Detection labels must start at 1")
            labels.append(label)
            areas.append(width * height)
            crowds.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(image_record["id"])], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Keep variable-size images and variable-length target dictionaries separate."""
    images, targets = zip(*batch)
    return images, targets
