from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_taco_detector import (  # noqa: E402
    cpu_max_long_side_from_metadata,
    detector_postprocessing_metadata,
    select_checkpoint_summaries,
    validate_manifest_split_isolation,
)
from train_taco_detector import evaluate_coco_bbox_metrics  # noqa: E402


class TacoDetectorEvaluationTests(unittest.TestCase):
    def test_evaluator_distinguishes_internal_and_application_thresholds(self) -> None:
        roi_heads = type(
            "RoiHeads", (), {"score_thresh": 0.05, "detections_per_img": 100}
        )()
        model = type("Model", (), {"roi_heads": roi_heads})()

        metadata = detector_postprocessing_metadata(model)

        self.assertEqual(metadata["model_internal_score_threshold"], 0.05)
        self.assertIsNone(metadata["additional_application_score_threshold"])
        self.assertEqual(metadata["model_maximum_detections_per_image"], 100)

    def test_checkpoint_selection_uses_validation_ap50_and_hash_tie_rule(self) -> None:
        common = {
            "annotation_sha256": "annotation",
            "manifest_sha256": "manifest",
        }
        lower = {**common, "validation_ap50": 0.2, "sha256": "aaa"}
        higher = {**common, "validation_ap50": 0.3, "sha256": "zzz"}
        tied_smaller_hash = {
            **common,
            "validation_ap50": 0.3,
            "sha256": "bbb",
        }

        selected = select_checkpoint_summaries(
            [lower, higher, tied_smaller_hash]
        )

        self.assertIs(selected, tied_smaller_hash)

    def test_saved_cpu_resize_settings_support_current_and_legacy_metadata(self) -> None:
        current = {
            "resize": {
                "cpu": {"maximum_long_side": 960},
                "model_transform": {"minimum_side": 640, "maximum_side": 960},
            }
        }
        legacy = {"resize": {"maximum_long_side": 320}}

        self.assertEqual(cpu_max_long_side_from_metadata(current), 960)
        self.assertEqual(cpu_max_long_side_from_metadata(legacy), 320)

    def test_manifest_split_isolation_rejects_repeated_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.csv"
            manifest.write_text(
                "image_id,file_name,split\n"
                "1,a.jpg,train\n"
                "2,b.jpg,validation\n"
                "3,c.jpg,test\n",
                encoding="utf-8",
            )
            splits = validate_manifest_split_isolation(manifest)
            self.assertEqual(splits["test"], {3})

            manifest.write_text(
                "image_id,file_name,split\n"
                "1,a.jpg,train\n"
                "2,b.jpg,validation\n"
                "1,a.jpg,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "repeated"):
                validate_manifest_split_isolation(manifest)

    def test_standard_coco_ap_is_one_for_a_perfect_box(self) -> None:
        targets = [
            {
                "image_id": torch.tensor([9]),
                "size": torch.tensor([100, 120]),
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]]),
                "labels": torch.tensor([2]),
                "iscrowd": torch.tensor([0]),
            }
        ]
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]]),
                "labels": torch.tensor([2]),
                "scores": torch.tensor([0.99]),
            }
        ]

        metrics = evaluate_coco_bbox_metrics(predictions, targets)

        self.assertAlmostEqual(metrics["overall_ap"], 1.0, places=6)
        self.assertAlmostEqual(metrics["overall_ap50"], 1.0, places=6)
        self.assertAlmostEqual(metrics["per_class"]["glass"]["ap"], 1.0, places=6)
        self.assertEqual(metrics["per_class"]["glass"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
