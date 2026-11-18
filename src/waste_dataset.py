"""Manifest-backed datasets and image preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model_config import (
    CLASS_MAPPING_PATH,
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RESIZE_SIZE,
)


class DatasetError(Exception):
    """A clear dataset or manifest configuration error."""


def load_saved_class_mapping() -> dict[str, int]:
    """Load and validate the one fixed class-to-index mapping."""
    try:
        with CLASS_MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
            mapping = json.load(mapping_file)
    except FileNotFoundError as error:
        raise DatasetError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"Could not read the class mapping: {error}") from error

    if mapping != CLASS_TO_INDEX:
        raise DatasetError(
            "The saved class mapping does not match the fixed project mapping."
        )
    return mapping


def training_transform() -> transforms.Compose:
    """Return randomized preprocessing used only for training images."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def inference_transform() -> transforms.Compose:
    """Return deterministic preprocessing shared by validation and inference."""
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class WasteManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load one manifest split while leaving all image files read-only."""

    def __init__(
        self,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        manifest_path: Path = MANIFEST_PATH,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise DatasetError(f"Unknown dataset split: {split!r}")

        self.split = split
        self.transform = transform
        self.class_to_index = load_saved_class_mapping()
        self.samples: list[tuple[Path, int]] = []

        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                required_columns = {"relative_path", "class_label", "split"}
                if set(reader.fieldnames or ()) != required_columns:
                    raise DatasetError(
                        "The split manifest must contain relative_path, class_label, "
                        "and split columns."
                    )

                for row in reader:
                    if row["split"] != split:
                        continue
                    class_label = row["class_label"]
                    if class_label not in self.class_to_index:
                        raise DatasetError(
                            f"Unknown class {class_label!r} in the split manifest."
                        )
                    image_path = PROJECT_ROOT / Path(row["relative_path"])
                    if not image_path.is_file():
                        raise DatasetError(f"Manifest image not found: {image_path}")
                    self.samples.append(
                        (image_path, self.class_to_index[class_label])
                    )
        except FileNotFoundError as error:
            raise DatasetError(f"Split manifest not found: {manifest_path}") from error
        except OSError as error:
            raise DatasetError(f"Could not read the split manifest: {error}") from error

        if not self.samples:
            raise DatasetError(f"No images found for split {split!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, class_index = self.samples[index]
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except OSError as error:
            raise DatasetError(f"Could not open image {image_path}: {error}") from error
        return self.transform(image), class_index


def create_training_loaders(
    batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders; the test split is never loaded here."""
    train_dataset = WasteManifestDataset("train", training_transform())
    validation_dataset = WasteManifestDataset("validation", inference_transform())

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader
