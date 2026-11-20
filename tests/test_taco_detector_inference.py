from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_mode_state import (  # noqa: E402
    CONTROLLED_MODE,
    EXPERIMENTAL_MODE,
    isolate_mode_state,
)
from taco_detector_inference import (  # noqa: E402
    DetectorSetupError,
    LoadedTacoDetector,
    PreparedDetectorImage,
    analyze_prepared_image,
    load_selected_detector,
    prepare_uploaded_image,
)


class FakeDetectorModel:
    def __init__(self, output: dict[str, torch.Tensor]) -> None:
        self.output = output
        self.inference_mode_observed = False

    def __call__(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        self.inference_mode_observed = torch.is_inference_mode_enabled()
        self.input_shape = tuple(images[0].shape)
        return [self.output]


def loaded_fake(model: FakeDetectorModel) -> LoadedTacoDetector:
    return LoadedTacoDetector(
        model=model,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        checkpoint_path=Path("mock.pt"),
        checkpoint_sha256="mock",
        cpu_max_long_side=50,
        detector_min_side=32,
        detector_max_side=50,
        model_internal_score_threshold=0.05,
        maximum_detections_per_image=100,
    )


class TacoDetectorInferenceTests(unittest.TestCase):
    def test_exif_orientation_is_applied_before_tensor_conversion(self) -> None:
        raw = Image.new("RGB", (4, 2), "blue")
        exif = Image.Exif()
        exif[274] = 6
        encoded = io.BytesIO()
        raw.save(encoded, format="JPEG", exif=exif)

        prepared = prepare_uploaded_image(encoded.getvalue(), cpu_max_long_side=20)

        self.assertEqual(prepared.original_image.size, (2, 4))
        self.assertEqual(tuple(prepared.tensor.shape), (3, 4, 2))

    def test_boxes_map_to_display_coordinates_and_other_name_is_explicit(self) -> None:
        original = Image.new("RGB", (100, 50), "white")
        resized = original.resize((50, 25))
        prepared = PreparedDetectorImage(
            original_image=original,
            resized_image=resized,
            tensor=torch.zeros((3, 25, 50), dtype=torch.float32),
            x_scale=0.5,
            y_scale=0.5,
        )
        model = FakeDetectorModel(
            {
                "boxes": torch.tensor([[5.0, 5.0, 25.0, 20.0]]),
                "labels": torch.tensor([6]),
                "scores": torch.tensor([0.80]),
            }
        )

        result = analyze_prepared_image(prepared, loaded_fake(model), 0.30)

        self.assertTrue(model.inference_mode_observed)
        self.assertEqual(model.input_shape, (3, 25, 50))
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.class_name, "other")
        self.assertEqual(detection.display_name, "annotated waste needing review")
        self.assertEqual(detection.box_xyxy, (10.0, 10.0, 50.0, 40.0))
        self.assertEqual(result.annotated_image.size, original.size)

    def test_output_below_display_cutoff_is_empty(self) -> None:
        image = Image.new("RGB", (20, 20), "white")
        prepared = PreparedDetectorImage(
            original_image=image,
            resized_image=image,
            tensor=torch.zeros((3, 20, 20), dtype=torch.float32),
            x_scale=1.0,
            y_scale=1.0,
        )
        model = FakeDetectorModel(
            {
                "boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.20]),
            }
        )

        result = analyze_prepared_image(prepared, loaded_fake(model), 0.30)

        self.assertEqual(result.detections, ())

    def test_checkpoint_hash_failure_stops_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "selected.pt"
            checkpoint.write_bytes(b"not the selected model")
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "selected_checkpoint_path": str(checkpoint),
                        "selected_checkpoint_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DetectorSetupError, "SHA-256"):
                load_selected_detector(selection, torch.device("cpu"))

    def test_switching_modes_clears_both_result_namespaces(self) -> None:
        state: dict[str, object] = {
            "active_analysis_mode": CONTROLLED_MODE,
            "scene_analysis": object(),
            "scene_analysis_key": "controlled",
            "detector_analysis": object(),
            "detector_analysis_key": "experimental",
        }

        changed = isolate_mode_state(state, EXPERIMENTAL_MODE)

        self.assertTrue(changed)
        self.assertEqual(state["active_analysis_mode"], EXPERIMENTAL_MODE)
        self.assertNotIn("scene_analysis", state)
        self.assertNotIn("detector_analysis", state)


if __name__ == "__main__":
    unittest.main()
