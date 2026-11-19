from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_taco_detector import (  # noqa: E402
    NUM_CLASSES_WITH_BACKGROUND,
    _checkpoint_metadata,
    build_model,
    checkpoint_paths,
    evaluate_coco_ap50,
    model_resize_settings,
)


class TacoDetectorTests(unittest.TestCase):
    def test_new_predictor_has_seven_outputs_without_downloading(self) -> None:
        model = build_model(
            pretrained=False,
            detector_min_size=384,
            detector_max_size=768,
        )

        self.assertEqual(
            model.roi_heads.box_predictor.cls_score.out_features,
            NUM_CLASSES_WITH_BACKGROUND,
        )
        self.assertEqual(model.transform.min_size, (384,))
        self.assertEqual(model.transform.max_size, 768)

    def test_checkpoint_metadata_records_both_resize_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "annotations.json"
            manifest_path = root / "manifest.csv"
            annotation_path.write_text("{}", encoding="utf-8")
            manifest_path.write_text("image_id,split\n", encoding="utf-8")
            args = SimpleNamespace(
                cpu_max_long_side=960,
                detector_min_side=640,
                detector_max_side=960,
                experiment_name="highres_640_960",
                seed=42,
                learning_rate=0.0025,
                momentum=0.9,
                weight_decay=0.0005,
                batch_size=1,
                amp=True,
                smoke_test=False,
            )

            metadata = _checkpoint_metadata(
                args, annotation_path, manifest_path, 544, 117
            )

            self.assertEqual(
                metadata["resize"]["cpu"]["maximum_long_side"], 960
            )
            self.assertEqual(
                metadata["resize"]["model_transform"],
                {"minimum_side": 640, "maximum_side": 960},
            )
            self.assertEqual(metadata["experiment_name"], "highres_640_960")

    def test_legacy_320_resize_metadata_remains_readable(self) -> None:
        metadata = {
            "resize": {
                "maximum_long_side": 320,
                "model_min_side": 320,
                "model_max_side": 320,
            }
        }

        self.assertEqual(model_resize_settings(metadata), (320, 320))

    def test_coco_ap50_is_one_for_perfect_detection(self) -> None:
        targets = [
            {
                "image_id": torch.tensor([17]),
                "size": torch.tensor([80, 100]),
                "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
                "labels": torch.tensor([3]),
                "iscrowd": torch.tensor([0]),
            }
        ]
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
                "labels": torch.tensor([3]),
                "scores": torch.tensor([0.99]),
            }
        ]

        metrics = evaluate_coco_ap50(predictions, targets)

        self.assertAlmostEqual(metrics["overall_ap50"], 1.0, places=6)
        self.assertAlmostEqual(
            metrics["per_class_ap50"]["metal"]["ap50"], 1.0, places=6
        )
        self.assertEqual(metrics["per_class_ap50"]["metal"]["support"], 1)
        self.assertIsNone(metrics["per_class_ap50"]["glass"]["ap50"])

    def test_smoke_checkpoint_cannot_overwrite_full_checkpoint(self) -> None:
        full_checkpoint, full_history = checkpoint_paths(
            Path("output"), False, "highres_640_960"
        )
        smoke_checkpoint, smoke_history = checkpoint_paths(
            Path("output"), True, "highres_640_960"
        )

        self.assertNotEqual(full_checkpoint, smoke_checkpoint)
        self.assertNotEqual(full_history, smoke_history)
        self.assertIn("smoke", smoke_checkpoint.name)
        self.assertIn("best", full_checkpoint.name)

    def test_high_resolution_output_is_isolated_from_320_experiment(self) -> None:
        old_checkpoint, _ = checkpoint_paths(
            Path("training_320"), False, "baseline_320"
        )
        highres_checkpoint, _ = checkpoint_paths(
            Path("training_highres"), False, "highres_640_960"
        )

        self.assertNotEqual(old_checkpoint, highres_checkpoint)
        self.assertIn("highres_640_960", highres_checkpoint.name)
        self.assertNotEqual(old_checkpoint.parent, highres_checkpoint.parent)


if __name__ == "__main__":
    unittest.main()
