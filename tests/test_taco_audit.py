from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_taco_dataset import CocoAuditError, audit_coco  # noqa: E402


def base_document() -> dict[str, object]:
    return {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}


class TacoAuditTests(unittest.TestCase):
    def test_category_mapping_is_explicit_and_unlabeled_is_excluded(self) -> None:
        data = base_document()
        data["images"] = [
            {"id": 1, "width": 100, "height": 80, "file_name": "one.jpg"},
            {"id": 2, "width": 100, "height": 80, "file_name": "two.jpg"},
        ]
        data["categories"] = [
            {"id": 10, "name": "Corrugated carton", "supercategory": "Carton"},
            {"id": 11, "name": "Glass bottle", "supercategory": "Bottle"},
            {"id": 12, "name": "Unlabeled litter", "supercategory": "Litter"},
        ]
        data["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [1, 2, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 11, "bbox": [30, 2, 20, 20]},
            {"id": 3, "image_id": 2, "category_id": 12, "bbox": [1, 2, 20, 20]},
        ]

        result = audit_coco(data)
        mapping = result["proposed_material_mapping"]

        self.assertEqual(mapping["instance_counts_by_material"]["cardboard"], 1)
        self.assertEqual(mapping["instance_counts_by_material"]["glass"], 1)
        self.assertEqual(mapping["mapped_instances"], 2)
        review_names = {
            category["name"] for category in mapping["review_or_exclusion"]
        }
        self.assertIn("Unlabeled litter", review_names)

    def test_malformed_records_are_reported_without_silent_acceptance(self) -> None:
        data = base_document()
        data["images"] = [
            {
                "id": 1,
                "width": 100,
                "height": 80,
                "file_name": "same.jpg",
                "flickr_url": "https://example.test/same.jpg",
            },
            {
                "id": 1,
                "width": 100,
                "height": 80,
                "file_name": "same.jpg",
                "flickr_url": "https://example.test/same.jpg",
            },
            {"id": 2, "width": 0, "height": 80, "file_name": "bad.jpg"},
        ]
        data["categories"] = [
            {"id": 10, "name": "Glass bottle", "supercategory": "Bottle"}
        ]
        data["annotations"] = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 10,
                "bbox": [1, 2, 20, 20],
                "segmentation": [[1, 2, 3, 4, 5, 6]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 10,
                "bbox": [1, 2, -2, 20],
                "segmentation": {"counts": "encoded", "size": [80, 100]},
            },
            {"id": 3, "image_id": 999, "category_id": 10, "bbox": [1, 2, 3, 4]},
            {"id": 4, "image_id": 1, "category_id": 999, "bbox": [1, 2, 3, 4]},
            "not-an-annotation-object",
        ]

        result = audit_coco(data)

        self.assertEqual(result["bounding_boxes"]["valid"], 2)
        self.assertEqual(result["bounding_boxes"]["invalid"], 3)
        self.assertEqual(result["record_counts"]["malformed_annotation_records"], 1)
        self.assertEqual(result["missing_references"]["annotation_to_image"], 1)
        self.assertEqual(result["missing_references"]["annotation_to_category"], 1)
        self.assertEqual(len(result["duplicates"]["image_ids"]), 1)
        self.assertEqual(len(result["duplicates"]["file_names"]), 1)
        self.assertEqual(
            len(result["duplicates"]["image_urls_across_all_url_fields"]), 1
        )
        self.assertEqual(result["segmentations"]["types"]["polygon_list"], 1)
        self.assertEqual(result["segmentations"]["types"]["rle_compressed"], 1)

    def test_non_list_coco_sections_fail_clearly(self) -> None:
        data = base_document()
        data["images"] = {"id": 1}
        with self.assertRaisesRegex(CocoAuditError, "images.*must be a list"):
            audit_coco(data)


if __name__ == "__main__":
    unittest.main()
