"""Local inference for the validation-selected experimental TACO detector."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model_config import PROJECT_ROOT


SELECTION_RECORD_PATH = (
    PROJECT_ROOT
    / "data"
    / "taco_detection"
    / "final_detector_evaluation"
    / "selection_record.json"
)
EXPECTED_CLASS_TO_INDEX = {
    "cardboard": 1,
    "glass": 2,
    "metal": 3,
    "paper": 4,
    "plastic": 5,
    "other": 6,
}
INDEX_TO_CLASS = {value: key for key, value in EXPECTED_CLASS_TO_INDEX.items()}
DISPLAY_NAMES = {
    **{name: name for name in EXPECTED_CLASS_TO_INDEX if name != "other"},
    "other": "annotated waste needing review",
}
EXPECTED_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
DISPLAY_SCORE_THRESHOLD = 0.30
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_DIMENSION = 6_000


class DetectorSetupError(Exception):
    """A clear local-checkpoint or dependency setup failure."""


class DetectorInputError(Exception):
    """A clear uploaded-image validation failure."""


@dataclass(frozen=True)
class PreparedDetectorImage:
    original_image: Image.Image
    resized_image: Image.Image
    tensor: torch.Tensor
    x_scale: float
    y_scale: float


@dataclass(frozen=True)
class LearnedDetection:
    class_name: str
    display_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorAnalysisResult:
    original_image: Image.Image
    annotated_image: Image.Image
    detections: tuple[LearnedDetection, ...]
    display_score_threshold: float
    model_internal_score_threshold: float
    maximum_detections_per_image: int


@dataclass(frozen=True)
class LoadedTacoDetector:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    cpu_max_long_side: int
    detector_min_side: int
    detector_max_side: int
    model_internal_score_threshold: float
    maximum_detections_per_image: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _saved_resize_settings(metadata: dict[str, Any]) -> tuple[int, int, int]:
    resize = metadata.get("resize")
    if not isinstance(resize, dict):
        raise DetectorSetupError("The selected checkpoint has no saved resize settings.")
    try:
        cpu_max = int(resize["cpu"]["maximum_long_side"])
        detector_min = int(resize["model_transform"]["minimum_side"])
        detector_max = int(resize["model_transform"]["maximum_side"])
    except (KeyError, TypeError, ValueError) as error:
        raise DetectorSetupError(
            "The selected checkpoint's CPU and model resize settings are incomplete."
        ) from error
    if cpu_max < 1 or detector_min < 1 or detector_max < detector_min:
        raise DetectorSetupError("The selected checkpoint has invalid resize settings.")
    return cpu_max, detector_min, detector_max


def load_selected_detector(
    selection_record_path: Path = SELECTION_RECORD_PATH,
    device: torch.device | None = None,
) -> LoadedTacoDetector:
    """Load exactly the locally selected checkpoint without downloading weights."""
    if not selection_record_path.is_file():
        raise DetectorSetupError(
            "The local TACO selection record is missing. Complete detector selection "
            f"first; expected {selection_record_path}."
        )
    try:
        selection = json.loads(selection_record_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(selection["selected_checkpoint_path"])
        expected_hash = str(selection["selected_checkpoint_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DetectorSetupError("The local TACO selection record is invalid.") from error
    if not checkpoint_path.is_file():
        raise DetectorSetupError(
            "The selected local TACO checkpoint is missing. Restore the ignored "
            f"checkpoint at {checkpoint_path}; no model will be downloaded."
        )
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise DetectorSetupError(
            "The selected TACO checkpoint failed SHA-256 verification. Refusing "
            "to load a different or modified model."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        state_dict = checkpoint["model_state_dict"]
    except (OSError, KeyError, TypeError, RuntimeError) as error:
        raise DetectorSetupError(f"Could not read the selected checkpoint: {error}") from error
    if metadata.get("architecture") != EXPECTED_ARCHITECTURE:
        raise DetectorSetupError("The selected checkpoint architecture is not supported.")
    if metadata.get("class_to_index") != EXPECTED_CLASS_TO_INDEX:
        raise DetectorSetupError(
            "The selected checkpoint class mapping does not match the six TACO labels."
        )
    cpu_max, detector_min, detector_max = _saved_resize_settings(metadata)

    target_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    try:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None
        )
        input_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            input_features, len(EXPECTED_CLASS_TO_INDEX) + 1
        )
        model.transform.min_size = (detector_min,)
        model.transform.max_size = detector_max
        model.load_state_dict(state_dict, strict=True)
        model.to(target_device).eval()
    except (RuntimeError, ValueError) as error:
        raise DetectorSetupError(
            f"Could not strictly load the selected TACO detector: {error}"
        ) from error

    return LoadedTacoDetector(
        model=model,
        device=target_device,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=actual_hash,
        cpu_max_long_side=cpu_max,
        detector_min_side=detector_min,
        detector_max_side=detector_max,
        model_internal_score_threshold=float(model.roi_heads.score_thresh),
        maximum_detections_per_image=int(model.roi_heads.detections_per_img),
    )


def prepare_uploaded_image(data: bytes, cpu_max_long_side: int) -> PreparedDetectorImage:
    """Decode, EXIF-orient, and downscale an upload without cropping."""
    if not data:
        raise DetectorInputError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DetectorInputError(
            f"The upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if (
                width < 1
                or height < 1
                or width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise DetectorInputError(
                    "The image exceeds the 20-million-pixel or 6000-pixel-side limit."
                )
            source.load()
            original = ImageOps.exif_transpose(source).convert("RGB")
    except DetectorInputError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise DetectorInputError(f"Could not decode the uploaded image: {error}") from error

    original_width, original_height = original.size
    scale = min(1.0, cpu_max_long_side / max(original.size))
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = (
        original.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        if (resized_width, resized_height) != original.size
        else original.copy()
    )
    image_array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    return PreparedDetectorImage(
        original_image=original,
        resized_image=resized,
        tensor=tensor,
        x_scale=resized_width / original_width,
        y_scale=resized_height / original_height,
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate(
    image: Image.Image, detections: tuple[LearnedDetection, ...]
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, round(min(image.size) * 0.005))
    font = _font(max(15, round(min(image.size) * 0.026)))
    for detection in detections:
        box = detection.box_xyxy
        color = "#B45309" if detection.class_name == "other" else "#0F766E"
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{detection.display_name} {detection.score:.1%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_height = text_box[3] - text_box[1] + line_width * 2
        label_width = text_box[2] - text_box[0] + line_width * 2
        label_x = max(0.0, box[0])
        label_y = max(0.0, box[1] - label_height)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color,
        )
        draw.text(
            (label_x + line_width, label_y + line_width // 2),
            label,
            fill="white",
            font=font,
        )
    return annotated


def analyze_prepared_image(
    prepared: PreparedDetectorImage,
    detector: LoadedTacoDetector,
    display_score_threshold: float = DISPLAY_SCORE_THRESHOLD,
) -> DetectorAnalysisResult:
    """Run the model once and map its resized-input boxes to the displayed image."""
    if not 0.0 <= display_score_threshold <= 1.0:
        raise ValueError("display_score_threshold must be between 0 and 1")
    with torch.inference_mode():
        output = detector.model([prepared.tensor.to(detector.device)])[0]

    detections: list[LearnedDetection] = []
    for box, label, score in zip(
        output["boxes"].detach().cpu(),
        output["labels"].detach().cpu(),
        output["scores"].detach().cpu(),
    ):
        confidence = float(score)
        if confidence < display_score_threshold:
            continue
        label_index = int(label)
        if label_index not in INDEX_TO_CLASS:
            raise DetectorSetupError(
                f"The detector returned unknown class index {label_index}."
            )
        x1, y1, x2, y2 = (float(value) for value in box.tolist())
        display_box = (
            max(0.0, min(prepared.original_image.width, x1 / prepared.x_scale)),
            max(0.0, min(prepared.original_image.height, y1 / prepared.y_scale)),
            max(0.0, min(prepared.original_image.width, x2 / prepared.x_scale)),
            max(0.0, min(prepared.original_image.height, y2 / prepared.y_scale)),
        )
        class_name = INDEX_TO_CLASS[label_index]
        detections.append(
            LearnedDetection(
                class_name=class_name,
                display_name=DISPLAY_NAMES[class_name],
                score=confidence,
                box_xyxy=display_box,
            )
        )
    frozen_detections = tuple(detections)
    return DetectorAnalysisResult(
        original_image=prepared.original_image,
        annotated_image=_annotate(prepared.original_image, frozen_detections),
        detections=frozen_detections,
        display_score_threshold=display_score_threshold,
        model_internal_score_threshold=detector.model_internal_score_threshold,
        maximum_detections_per_image=detector.maximum_detections_per_image,
    )


def analyze_uploaded_image(
    data: bytes,
    detector: LoadedTacoDetector,
    display_score_threshold: float = DISPLAY_SCORE_THRESHOLD,
) -> DetectorAnalysisResult:
    prepared = prepare_uploaded_image(data, detector.cpu_max_long_side)
    return analyze_prepared_image(prepared, detector, display_score_threshold)
