"""Predict one waste image and apply the selected-bin quality rule."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_config import CHECKPOINT_PATH, CLASS_NAMES
from predictor import PredictionError, WastePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image with one main item")
    parser.add_argument("target_bin", choices=CLASS_NAMES, help="Selected target bin")
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Checkpoint path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        predictor = WastePredictor(args.checkpoint)
        result = predictor.predict(args.image, args.target_bin)
    except PredictionError as error:
        raise SystemExit(f"ERROR: {error}") from error

    print(f"Image: {args.image}")
    print(f"Selected target bin: {args.target_bin}")
    print(f"Predicted class: {result.predicted_class}")
    print(f"Top score: {result.top_score:.4f}")
    print("Softmax scores:")
    for class_name, score in result.scores.items():
        print(f"  {class_name}: {score:.4f}")
    print(f"Quality control: {result.quality_control_result}")
    print(result.notice)


if __name__ == "__main__":
    main()
