from __future__ import annotations

import csv
import io
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_config import CLASS_NAMES  # noqa: E402
from predictor import PredictionResult, quality_control_result  # noqa: E402
from scene_analysis import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    SceneAnalysisError,
    analyze_scene,
    annotated_png_bytes,
    item_csv_bytes,
    load_uploaded_image,
)


class FakePredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict_image(
        self, image: Image.Image, selected_bin: str | None = None
    ) -> PredictionResult:
        del image
        self.calls += 1
        predicted_class = "cardboard" if self.calls == 1 else "glass"
        top_score = 0.92 if self.calls == 1 else 0.55
        scores = {name: 0.0 for name in CLASS_NAMES}
        scores[predicted_class] = top_score
        scores["trash"] += 1.0 - top_score
        return PredictionResult(
            predicted_class=predicted_class,
            top_score=top_score,
            scores=scores,
            quality_control_result=quality_control_result(
                predicted_class, top_score, selected_bin
            ),
        )


class SceneAnalysisTests(unittest.TestCase):
    def controlled_scene(self) -> Image.Image:
        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((70, 80, 230, 270), fill="#222222")
        draw.rectangle((390, 90, 560, 270), fill="#2563A8")
        return image

    def test_analysis_and_downloads_match_item_records(self) -> None:
        result = analyze_scene(
            self.controlled_scene(), FakePredictor(), selected_bin="cardboard"
        )
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].suggested_category, "cardboard")
        self.assertEqual(result.items[1].suggested_category, "Manual review")
        self.assertEqual(result.review_count, 1)

        rows = list(
            csv.DictReader(io.StringIO(item_csv_bytes(result).decode("utf-8")))
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["predicted_class"], "cardboard")
        self.assertEqual(rows[1]["manual_review"], "true")
        self.assertEqual(rows[0]["target_bin"], "cardboard")

        with Image.open(io.BytesIO(annotated_png_bytes(result))) as annotated:
            self.assertEqual(annotated.size, self.controlled_scene().size)
            self.assertEqual(annotated.format, "PNG")

    def test_invalid_upload_fails_clearly(self) -> None:
        with self.assertRaisesRegex(SceneAnalysisError, "Could not decode"):
            load_uploaded_image(b"this is not an image")

    def test_oversized_upload_fails_before_decode(self) -> None:
        with self.assertRaisesRegex(SceneAnalysisError, "maximum supported size"):
            load_uploaded_image(b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_blank_scene_does_not_call_the_classifier(self) -> None:
        predictor = FakePredictor()
        result = analyze_scene(Image.new("RGB", (600, 400), "white"), predictor)
        self.assertEqual(result.items, ())
        self.assertEqual(predictor.calls, 0)
        self.assertIn("No classification was attempted", result.localization.message)

    def test_optional_target_returns_sorting_suggestion(self) -> None:
        self.assertEqual(
            quality_control_result("metal", 0.91, None),
            "Suggested category: metal",
        )
        self.assertEqual(quality_control_result("metal", 0.40, None), "Manual review")


if __name__ == "__main__":
    unittest.main()
