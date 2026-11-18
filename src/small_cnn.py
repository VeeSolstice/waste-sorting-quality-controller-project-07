"""A compact convolutional classifier trained from random initialization."""

from __future__ import annotations

from pathlib import Path

from torch import nn
from torchvision import transforms

from model_config import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_ROOT,
)


SMALL_CNN_ARCHITECTURE = "small_cnn"
SMALL_CNN_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "waste_small_cnn.pt"
SMALL_CNN_SEED = 42
SMALL_CNN_EPOCHS = 5
SMALL_CNN_BATCH_SIZE = 32
SMALL_CNN_LEARNING_RATE = 0.001
SMALL_CNN_RESIZE_SIZE = 144
SMALL_CNN_IMAGE_SIZE = 128

SMALL_CNN_PREPROCESSING = {
    "input_color_mode": "RGB",
    "image_size": SMALL_CNN_IMAGE_SIZE,
    "resize_size": SMALL_CNN_RESIZE_SIZE,
    "training": (
        "RandomResizedCrop(128), RandomHorizontalFlip, ToTensor, "
        "ImageNet normalization"
    ),
    "validation_and_inference": (
        "Resize shorter side to 144, CenterCrop(128), ToTensor, "
        "ImageNet normalization"
    ),
    "imagenet_mean": list(IMAGENET_MEAN),
    "imagenet_std": list(IMAGENET_STD),
}


class SmallCNN(nn.Module):
    """Four convolution blocks followed by a small fully connected head."""

    def __init__(self, class_count: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, class_count),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def small_cnn_training_transform() -> transforms.Compose:
    """Randomized preprocessing used for scratch-CNN training only."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(SMALL_CNN_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def small_cnn_inference_transform() -> transforms.Compose:
    """Deterministic preprocessing used for validation and final evaluation."""
    return transforms.Compose(
        [
            transforms.Resize(SMALL_CNN_RESIZE_SIZE),
            transforms.CenterCrop(SMALL_CNN_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def checkpoint_relative_path() -> str:
    return SMALL_CNN_CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix()
