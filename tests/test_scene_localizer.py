from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_localizer import (  # noqa: E402
    LocalizationError,
    localize_objects,
    validate_image_dimensions,
)


class SceneLocalizerTests(unittest.TestCase):
    def test_separated_regions_are_localized_in_stable_order(self) -> None:
        image = Image.new("RGB", (720, 420), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 80, 180, 250), fill="#202020")
        draw.rectangle((290, 110, 410, 300), fill="#2563A8")
        draw.rectangle((530, 60, 660, 220), fill="#B45309")

        result = localize_objects(image)

        self.assertEqual(len(result.boxes), 3)
        self.assertEqual(
            list(result.boxes),
            list(localize_objects(image).boxes),
        )
        self.assertLessEqual(result.boxes[0].y_min, result.boxes[1].y_min)
        self.assertLessEqual(result.boxes[1].y_min, result.boxes[2].y_min)
        self.assertIn("Localized 3", result.message)

    def test_plain_background_returns_zero_objects(self) -> None:
        result = localize_objects(Image.new("RGB", (600, 400), "#F8F8F8"))
        self.assertEqual(result.boxes, ())
        self.assertIn("No classification was attempted", result.message)

    def test_touching_objects_merge_into_one_region(self) -> None:
        image = Image.new("RGB", (500, 350), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 90, 240, 260), fill="#111111")
        draw.rectangle((240, 110, 390, 260), fill="#4F46E5")
        result = localize_objects(image)
        self.assertEqual(len(result.boxes), 1)

    def test_overlapping_fragments_of_one_object_are_reconciled(self) -> None:
        image = Image.new("RGB", (500, 350), "white")
        draw = ImageDraw.Draw(image)
        # A U-shaped main component surrounds a disconnected inner highlight.
        # Their pixels do not touch, but the smaller component's box is contained
        # by the main component's box, matching a fragmented reflective object.
        draw.rectangle((140, 75, 165, 270), fill="#202020")
        draw.rectangle((285, 75, 310, 270), fill="#202020")
        draw.rectangle((140, 235, 310, 270), fill="#202020")
        draw.rectangle((185, 120, 265, 165), fill="#202020")

        result = localize_objects(image)

        self.assertEqual(len(result.boxes), 1)
        self.assertLessEqual(result.boxes[0].x_min, 140)
        self.assertGreaterEqual(result.boxes[0].x_max, 311)

    def test_two_nearby_peer_sized_objects_remain_separate(self) -> None:
        image = Image.new("RGB", (500, 350), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((90, 90, 215, 270), fill="#202020")
        draw.rectangle((221, 90, 346, 270), fill="#315FA8")

        result = localize_objects(image)

        self.assertEqual(len(result.boxes), 2)

    def test_small_noise_is_removed(self) -> None:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        for x in range(20, 580, 40):
            draw.rectangle((x, 50, x + 2, 52), fill="black")
        result = localize_objects(image)
        self.assertEqual(result.boxes, ())

    def test_too_many_substantial_regions_fail_clearly(self) -> None:
        image = Image.new("RGB", (900, 600), "white")
        draw = ImageDraw.Draw(image)
        for index in range(13):
            column = index % 7
            row = index // 7
            x = 35 + column * 120
            y = 90 + row * 220
            draw.rectangle((x, y, x + 54, y + 74), fill="#202020")
        with self.assertRaisesRegex(LocalizationError, "above the supported limit"):
            localize_objects(image)

    def test_oversized_dimensions_fail_before_processing(self) -> None:
        with self.assertRaisesRegex(LocalizationError, "supported maximum"):
            validate_image_dimensions(7000, 4000)


if __name__ == "__main__":
    unittest.main()
