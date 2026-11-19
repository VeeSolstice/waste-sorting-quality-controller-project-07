"""Two-stage multi-item scene analysis and downloadable result helpers."""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from model_config import CLASS_NAMES, QUALITY_THRESHOLD
from predictor import PredictionError, WastePredictor
from scene_localizer import (
    BoundingBox,
    LocalizationError,
    LocalizationResult,
    localize_objects,
    validate_image_dimensions,
)


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class SceneAnalysisError(Exception):
    """A user-facing uploaded-image or scene-analysis error."""


@dataclass(frozen=True)
class SceneItem:
    item_number: int
    box: BoundingBox
    predicted_class: str
    top_score: float
    scores: dict[str, float]
    suggested_category: str
    quality_control_result: str
    manual_review: bool
    review_required: bool


@dataclass(frozen=True)
class SceneAnalysisResult:
    original_image: Image.Image
    annotated_image: Image.Image
    localization: LocalizationResult
    items: tuple[SceneItem, ...]
    selected_bin: str | None

    @property
    def counts_by_suggestion(self) -> dict[str, int]:
        return dict(Counter(item.suggested_category for item in self.items))

    @property
    def review_count(self) -> int:
        return sum(item.review_required for item in self.items)


def load_uploaded_image(data: bytes) -> Image.Image:
    """Fully decode uploaded bytes as an RGB image with conservative limits."""
    if not data:
        raise SceneAnalysisError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise SceneAnalysisError(
            f"The uploaded file is {len(data) / (1024 * 1024):.1f} MB; the maximum "
            f"supported size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            validate_image_dimensions(*source.size)
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except LocalizationError as error:
        raise SceneAnalysisError(str(error)) from error
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise SceneAnalysisError(f"Could not decode the uploaded image: {error}") from error

    return image


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "Arial Bold.ttf") if bold else ("arial.ttf", "Arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate(image: Image.Image, items: list[SceneItem]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    short_side = min(annotated.size)
    line_width = max(3, round(short_side * 0.006))
    font = _font(max(16, round(short_side * 0.033)), bold=True)

    for item in items:
        box = item.box
        color = "#D97706" if item.review_required else "#0F766E"
        draw.rounded_rectangle(
            (box.x_min, box.y_min, box.x_max, box.y_max),
            radius=max(5, line_width * 2),
            outline=color,
            width=line_width,
        )
        label = f"{item.item_number}  {item.predicted_class}  {item.top_score:.1%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = box.x_min
        label_y = max(0, box.y_min - text_height - line_width * 3)
        draw.rounded_rectangle(
            (
                label_x,
                label_y,
                min(annotated.width, label_x + text_width + line_width * 4),
                label_y + text_height + line_width * 3,
            ),
            radius=max(3, line_width),
            fill=color,
        )
        draw.text(
            (label_x + line_width * 2, label_y + line_width),
            label,
            fill="white",
            font=font,
        )
    return annotated


def analyze_scene(
    image: Image.Image,
    predictor: WastePredictor,
    selected_bin: str | None = None,
) -> SceneAnalysisResult:
    """Localize separated regions, then classify only those in-memory crops."""
    if selected_bin is not None and selected_bin not in CLASS_NAMES:
        raise SceneAnalysisError(f"Unknown target bin: {selected_bin!r}")

    try:
        localization = localize_objects(image)
    except LocalizationError as error:
        raise SceneAnalysisError(str(error)) from error

    rgb_image = image.convert("RGB")
    items: list[SceneItem] = []
    for item_number, box in enumerate(localization.boxes, start=1):
        crop = rgb_image.crop((box.x_min, box.y_min, box.x_max, box.y_max))
        try:
            prediction = predictor.predict_image(crop, selected_bin)
        except PredictionError as error:
            raise SceneAnalysisError(
                f"Could not classify localized item {item_number}: {error}"
            ) from error

        manual_review = prediction.top_score < QUALITY_THRESHOLD
        suggested_category = (
            "Manual review" if manual_review else prediction.predicted_class
        )
        review_required = manual_review or (
            selected_bin is not None and prediction.predicted_class != selected_bin
        )
        items.append(
            SceneItem(
                item_number=item_number,
                box=box,
                predicted_class=prediction.predicted_class,
                top_score=prediction.top_score,
                scores=prediction.scores,
                suggested_category=suggested_category,
                quality_control_result=prediction.quality_control_result,
                manual_review=manual_review,
                review_required=review_required,
            )
        )

    return SceneAnalysisResult(
        original_image=rgb_image,
        annotated_image=_annotate(rgb_image, items),
        localization=localization,
        items=tuple(items),
        selected_bin=selected_bin,
    )


def annotated_png_bytes(result: SceneAnalysisResult) -> bytes:
    output = io.BytesIO()
    result.annotated_image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def item_csv_bytes(result: SceneAnalysisResult) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = [
        "item_number",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "predicted_class",
        "top_score",
        "suggested_category",
        "manual_review",
        "review_required",
        "target_bin",
        "quality_control_result",
        *(f"score_{class_name}" for class_name in CLASS_NAMES),
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in result.items:
        writer.writerow(
            {
                "item_number": item.item_number,
                "x_min": item.box.x_min,
                "y_min": item.box.y_min,
                "x_max": item.box.x_max,
                "y_max": item.box.y_max,
                "predicted_class": item.predicted_class,
                "top_score": f"{item.top_score:.6f}",
                "suggested_category": item.suggested_category,
                "manual_review": str(item.manual_review).lower(),
                "review_required": str(item.review_required).lower(),
                "target_bin": result.selected_bin or "",
                "quality_control_result": item.quality_control_result,
                **{
                    f"score_{class_name}": f"{item.scores[class_name]:.6f}"
                    for class_name in CLASS_NAMES
                },
            }
        )
    return output.getvalue().encode("utf-8")
