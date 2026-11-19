from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_taco_detector import (  # noqa: E402
    NUM_CLASSES_WITH_BACKGROUND,
    build_model,
    checkpoint_paths,
    evaluate_coco_ap50,
)


class TacoDetectorTests(unittest.TestCase):
    def test_new_predictor_has_seven_outputs_without_downloading(self) -> None:
        model = build_model(pretrained=False, detector_max_size=480)

        self.assertEqual(
            model.roi_heads.box_predictor.cls_score.out_features,
            NUM_CLASSES_WITH_BACKGROUND,
        )
        self.assertEqual(model.transform.max_size, 480)

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
        full_checkpoint, full_history = checkpoint_paths(Path("output"), False)
        smoke_checkpoint, smoke_history = checkpoint_paths(Path("output"), True)

        self.assertNotEqual(full_checkpoint, smoke_checkpoint)
        self.assertNotEqual(full_history, smoke_history)
        self.assertIn("smoke", smoke_checkpoint.name)
        self.assertIn("best", full_checkpoint.name)


if __name__ == "__main__":
    unittest.main()
