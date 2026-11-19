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
MIN_FRAGMENT_OVERLAP_RATIO = 0.65
MAX_NEARBY_FRAGMENT_AREA_RATIO = 0.18


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


def _bounding_area(box: BoundingBox) -> int:
    return box.width * box.height


def _axis_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _axis_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0


def _should_merge_fragments(
    first: BoundingBox,
    second: BoundingBox,
    distance_tolerance: int,
) -> bool:
    """Return True only when two component boxes strongly suggest one object.

    Components whose boxes substantially overlap or contain one another are likely
    foreground fragments separated by low-contrast pixels. A tiny gap is allowed
    only when the smaller component is much smaller and the boxes are strongly
    aligned on the other axis. Peer-sized objects are therefore not joined merely
    because they sit close together on the same row.
    """
    x_overlap = _axis_overlap(
        first.x_min, first.x_max, second.x_min, second.x_max
    )
    y_overlap = _axis_overlap(
        first.y_min, first.y_max, second.y_min, second.y_max
    )
    first_box_area = _bounding_area(first)
    second_box_area = _bounding_area(second)
    smaller_box_area = min(first_box_area, second_box_area)
    larger_box_area = max(first_box_area, second_box_area)
    overlap_ratio = (x_overlap * y_overlap) / smaller_box_area

    if overlap_ratio >= MIN_FRAGMENT_OVERLAP_RATIO:
        return True

    smaller, larger = (
        (first, second)
        if first_box_area <= second_box_area
        else (second, first)
    )
    almost_contained = (
        smaller.x_min >= larger.x_min - distance_tolerance
        and smaller.y_min >= larger.y_min - distance_tolerance
        and smaller.x_max <= larger.x_max + distance_tolerance
        and smaller.y_max <= larger.y_max + distance_tolerance
    )
    if almost_contained and overlap_ratio >= 0.50:
        return True

    if smaller_box_area / larger_box_area > MAX_NEARBY_FRAGMENT_AREA_RATIO:
        return False

    x_gap = _axis_gap(first.x_min, first.x_max, second.x_min, second.x_max)
    y_gap = _axis_gap(first.y_min, first.y_max, second.y_min, second.y_max)
    vertical_alignment = y_overlap / min(first.height, second.height)
    horizontal_alignment = x_overlap / min(first.width, second.width)
    return (
        x_gap <= distance_tolerance
        and vertical_alignment >= 0.85
    ) or (
        y_gap <= distance_tolerance
        and horizontal_alignment >= 0.85
    )


def _merge_component_fragments(
    boxes: list[BoundingBox], distance_tolerance: int
) -> list[BoundingBox]:
    """Combine component boxes linked by conservative fragment evidence."""
    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            if _should_merge_fragments(
                first, boxes[second_index], distance_tolerance
            ):
                union(first_index, second_index)

    groups: dict[int, list[BoundingBox]] = {}
    for index, box in enumerate(boxes):
        groups.setdefault(find(index), []).append(box)

    return [
        BoundingBox(
            x_min=min(box.x_min for box in group),
            y_min=min(box.y_min for box in group),
            x_max=max(box.x_max for box in group),
            y_max=max(box.y_max for box in group),
            area=sum(box.area for box in group),
        )
        for group in groups.values()
    ]


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

    candidates: list[BoundingBox] = []
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
        candidates.append(
            BoundingBox(
                x_min=x,
                y_min=y,
                x_max=x + component_width,
                y_max=y + component_height,
                area=area,
            )
        )

    if len(candidates) > MAX_OBJECTS * 4:
        raise LocalizationError(
            f"The foreground mask produced {len(candidates)} substantial regions, "
            "too many to reconcile safely. Use a plainer background, "
            "remove clutter, or photograph fewer separated items."
        )

    fragment_tolerance = max(2, round(short_side * 0.003))
    candidates = _merge_component_fragments(candidates, fragment_tolerance)
    if len(candidates) > MAX_OBJECTS:
        raise LocalizationError(
            f"The foreground mask produced {len(candidates)} substantial regions, "
            f"above the supported limit of {MAX_OBJECTS}. Use a plainer background, "
            "remove clutter, or photograph fewer separated items."
        )

    padding = max(4, round(short_side * 0.012))
    boxes = [
        BoundingBox(
            x_min=max(0, box.x_min - padding),
            y_min=max(0, box.y_min - padding),
            x_max=min(width, box.x_max + padding),
            y_max=min(height, box.y_max + padding),
            area=box.area,
        )
        for box in candidates
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
