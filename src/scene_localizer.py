"""Explainable foreground-region localization for controlled-background scenes.

This is classical image processing, not YOLO and not a learned object detector.
It is intended for separated objects photographed against a plain, contrasting
background. Touching objects can merge into one region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_DIMENSION = 6_000
MIN_COLOR_DISTANCE = 24
MIN_REGION_AREA = 700
MIN_REGION_AREA_RATIO = 0.0015
MAX_REGION_AREA_RATIO = 0.88
MAX_OBJECTS = 12


class LocalizationError(Exception):
    """A clear scene-localization input or complexity error."""


@dataclass(frozen=True)
class BoundingBox:
    """A pixel-aligned bounding box using an exclusive lower-right corner."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class LocalizationResult:
    boxes: tuple[BoundingBox, ...]
    foreground_mask: np.ndarray
    estimated_background_rgb: tuple[int, int, int]
    color_distance_threshold: float
    minimum_region_area: int
    message: str


def validate_image_dimensions(width: int, height: int) -> None:
    """Reject empty or oversized images before expensive processing."""
    if width < 1 or height < 1:
        raise LocalizationError("The image has invalid dimensions.")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise LocalizationError(
            f"Image dimensions {width}×{height} exceed the supported maximum "
            f"of {MAX_IMAGE_DIMENSION} pixels per side."
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise LocalizationError(
            f"Image contains {width * height:,} pixels; the supported maximum is "
            f"{MAX_IMAGE_PIXELS:,}."
        )


def _odd_kernel_size(short_side: int, fraction: float, maximum: int) -> int:
    size = max(3, min(maximum, round(short_side * fraction)))
    return size if size % 2 else size + 1


def _estimate_background(rgb: np.ndarray) -> np.ndarray:
    """Estimate the plain background from a narrow border around the image."""
    height, width = rgb.shape[:2]
    border = max(2, round(min(height, width) * 0.025))
    border_pixels = np.concatenate(
        (
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(border_pixels, axis=0).astype(np.float32)


def localize_objects(image: Image.Image) -> LocalizationResult:
    """Return separated foreground regions for a controlled-background image."""
    if not isinstance(image, Image.Image):
        raise LocalizationError("Scene input must be a PIL image.")

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    validate_image_dimensions(width, height)
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    background = _estimate_background(rgb)

    difference = rgb.astype(np.float32) - background.reshape(1, 1, 3)
    color_distance = np.linalg.norm(difference, axis=2)
    distance_u8 = np.clip(color_distance, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(float(MIN_COLOR_DISTANCE), float(otsu_threshold))
    mask = np.where(color_distance >= threshold, 255, 0).astype(np.uint8)

    short_side = min(width, height)
    open_size = _odd_kernel_size(short_side, 0.004, 7)
    close_size = _odd_kernel_size(short_side, 0.009, 13)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    image_area = width * height
    minimum_area = max(
        MIN_REGION_AREA, round(image_area * MIN_REGION_AREA_RATIO)
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates: list[tuple[int, int, int, int, int]] = []
    oversized_foreground = False
    for component in range(1, component_count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component]
        )
        if area < minimum_area or component_width < 18 or component_height < 18:
            continue
        if area / image_area > MAX_REGION_AREA_RATIO:
            oversized_foreground = True
            continue
        candidates.append((x, y, component_width, component_height, area))

    if len(candidates) > MAX_OBJECTS:
        raise LocalizationError(
            f"The foreground mask produced {len(candidates)} substantial regions, "
            f"above the supported limit of {MAX_OBJECTS}. Use a plainer background, "
            "remove clutter, or photograph fewer separated items."
        )

    padding = max(4, round(short_side * 0.012))
    boxes = [
        BoundingBox(
            x_min=max(0, x - padding),
            y_min=max(0, y - padding),
            x_max=min(width, x + component_width + padding),
            y_max=min(height, y + component_height + padding),
            area=area,
        )
        for x, y, component_width, component_height, area in candidates
    ]
    boxes.sort(key=lambda box: (box.y_min, box.x_min, box.y_max, box.x_max))

    if boxes:
        message = f"Localized {len(boxes)} separated foreground region(s)."
    elif oversized_foreground:
        message = (
            "Localization failed because foreground covered almost the entire image. "
            "Use separated items on a plain, contrasting background."
        )
    else:
        message = (
            "No foreground object met the contrast and minimum-area rules. "
            "No classification was attempted."
        )

    return LocalizationResult(
        boxes=tuple(boxes),
        foreground_mask=mask,
        estimated_background_rgb=tuple(int(round(value)) for value in background),
        color_distance_threshold=threshold,
        minimum_region_area=minimum_area,
        message=message,
    )
