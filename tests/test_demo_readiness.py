from __future__ import annotations

import hashlib
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

from check_demo_readiness import (  # noqa: E402
    EXPECTED_CLASS_TO_INDEX,
    check_image,
    check_selected_detector,
)


def detector_metadata() -> dict[str, object]:
    return {
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "class_to_index": EXPECTED_CLASS_TO_INDEX,
        "resize": {
            "cpu": {"maximum_long_side": 960},
            "model_transform": {"minimum_side": 640, "maximum_side": 960},
        },
    }


class DemoReadinessTests(unittest.TestCase):
    def test_selected_detector_hash_and_metadata_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "detector.pt"
            torch.save(
                {"metadata": detector_metadata(), "model_state_dict": {"weight": 1}},
                checkpoint,
            )
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "selected_checkpoint_path": str(checkpoint),
                        "selected_checkpoint_sha256": digest,
                        "selected_validation_ap50": 0.2,
                    }
                ),
                encoding="utf-8",
            )

            checks = check_selected_detector(selection)

            self.assertEqual(len(checks), 2)
            self.assertTrue(all(check.passed for check in checks))
            self.assertIn("model min/max 640/960", checks[1].detail)

    def test_selected_detector_hash_mismatch_fails_before_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "detector.pt"
            checkpoint.write_bytes(b"different bytes")
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

            checks = check_selected_detector(selection)

            self.assertTrue(checks[0].passed)
            self.assertFalse(checks[1].passed)
            self.assertIn("SHA-256 mismatch", checks[1].detail)

    def test_demo_image_is_fully_decoded_and_exif_oriented(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "oriented.jpg"
            image = Image.new("RGB", (4, 2), "blue")
            exif = Image.Exif()
            exif[274] = 6
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", exif=exif)
            path.write_bytes(buffer.getvalue())

            check = check_image(path, "Demo image")

            self.assertTrue(check.passed)
            self.assertIn("RGB 2x4", check.detail)


if __name__ == "__main__":
    unittest.main()
