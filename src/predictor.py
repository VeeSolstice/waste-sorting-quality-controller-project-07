"""Shared checkpoint loading, prediction, and quality-control rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from model_config import (
    CHECKPOINT_PATH,
    CLASS_TO_INDEX,
    QUALITY_THRESHOLD,
    SOFTMAX_NOTICE,
)
from waste_dataset import inference_transform
from waste_model import ARCHITECTURE_NAME, create_inference_model


class PredictionError(Exception):
    """A user-facing checkpoint or image prediction error."""


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    top_score: float
    scores: dict[str, float]
    quality_control_result: str
    notice: str = SOFTMAX_NOTICE


def quality_control_result(
    predicted_class: str,
    top_score: float,
    selected_bin: str,
    threshold: float = QUALITY_THRESHOLD,
) -> str:
    """Apply the initial prediction-based quality-control rule."""
    # Softmax is not a calibrated guarantee; this is only a prediction rule.
    if top_score < threshold:
        return "Manual review"
    if predicted_class == selected_bin:
        return "Matches selected bin"
    return "Potential contaminant—review"


class WastePredictor:
    """Load one checkpoint once and reuse it for deterministic predictions."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
        if not checkpoint_path.is_file():
            raise PredictionError(
                f"Checkpoint not found: {checkpoint_path}. Run src/train_model.py first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
        except Exception as error:
            raise PredictionError(f"Could not load checkpoint: {error}") from error

        if checkpoint.get("architecture") != ARCHITECTURE_NAME:
            raise PredictionError(
                f"Unsupported checkpoint architecture: {checkpoint.get('architecture')!r}"
            )
        if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
            raise PredictionError("Checkpoint class mapping does not match this project.")

        self.class_to_index = checkpoint["class_to_index"]
        self.index_to_class = {
            class_index: class_name
            for class_name, class_index in self.class_to_index.items()
        }
        self.transform = inference_transform()
        self.model = create_inference_model().to(self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError) as error:
            raise PredictionError(
                f"Checkpoint model weights are missing or incompatible: {error}"
            ) from error
        self.model.eval()

    def predict(self, image_path: Path, selected_bin: str) -> PredictionResult:
        if selected_bin not in self.class_to_index:
            choices = ", ".join(self.class_to_index)
            raise PredictionError(
                f"Unknown target bin {selected_bin!r}. Choose one of: {choices}."
            )
        if not image_path.is_file():
            raise PredictionError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise PredictionError(f"Could not open image {image_path}: {error}") from error

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(image_tensor), dim=1)[0].cpu()

        scores = {
            self.index_to_class[index]: probabilities[index].item()
            for index in range(len(self.index_to_class))
        }
        predicted_class, top_score = max(scores.items(), key=lambda item: item[1])
        decision = quality_control_result(
            predicted_class, top_score, selected_bin
        )
        return PredictionResult(predicted_class, top_score, scores, decision)
