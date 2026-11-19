from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_taco_detection import (  # noqa: E402
    POLICY_A,
    POLICY_B,
    build_derived_coco,
    make_grouped_splits,
    plan_policy,
)
from taco_detection_dataset import (  # noqa: E402
    TacoDetectionDataset,
    detection_collate_fn,
)


def sample_coco() -> dict[str, object]:
    return {
        "images": [
            {"id": 1, "width": 20, "height": 16, "file_name": "one.jpg", "license": "CC"},
            {"id": 2, "width": 20, "height": 16, "file_name": "two.jpg", "license": "CC"},
            {"id": 3, "width": 20, "height": 16, "file_name": "three.jpg", "license": "CC"},
        ],
        "categories": [
            {"id": 10, "name": "Glass bottle"},
            {"id": 11, "name": "Unlabeled litter"},
        ],
        "annotations": [
            {"id": 7, "image_id": 1, "category_id": 10, "bbox": [1, 2, 5, 6]},
            {"id": 7, "image_id": 2, "category_id": 10, "bbox": [1, 2, 5, 6]},
            {"id": 8, "image_id": 2, "category_id": 11, "bbox": [8, 2, 5, 6]},
            {"id": 9, "image_id": 3, "category_id": 10, "bbox": [-1, 2, 5, 6]},
        ],
    }


class TacoDetectionPreparationTests(unittest.TestCase):
    def test_policy_a_excludes_mixed_image_and_policy_b_uses_other(self) -> None:
        data = sample_coco()
        policy_a, summary_a = plan_policy(data, POLICY_A)
        policy_b, summary_b = plan_policy(data, POLICY_B)

        self.assertEqual([item.image_id for item in policy_a], [1])
        self.assertEqual([item.image_id for item in policy_b], [1, 2])
        self.assertEqual(summary_a["box_counts_by_class"], {"glass": 1})
        self.assertEqual(summary_b["box_counts_by_class"], {"glass": 2, "other": 1})

    def test_invalid_box_excludes_entire_image(self) -> None:
        data = sample_coco()
        for policy in (POLICY_A, POLICY_B):
            planned, summary = plan_policy(data, policy)
            self.assertNotIn(3, [item.image_id for item in planned])
            self.assertEqual(summary["excluded_image_counts"]["invalid_box_or_reference_image"], 1)

    def test_duplicate_source_annotation_ids_receive_unique_derived_ids(self) -> None:
        planned, _ = plan_policy(sample_coco(), POLICY_B)
        verification = {
            item.image_id: {
                "width": 20,
                "height": 16,
                "sha256": f"hash-{item.image_id}",
                "difference_hash": "0" * 16,
                "average_hash": "0" * 16,
            }
            for item in planned
        }

        derived = build_derived_coco(planned, verification, POLICY_B)

        derived_ids = [annotation["id"] for annotation in derived["annotations"]]
        source_ids = [
            annotation["source_annotation_id"] for annotation in derived["annotations"]
        ]
        self.assertEqual(len(derived_ids), len(set(derived_ids)))
        self.assertEqual(source_ids.count(7), 2)
        self.assertEqual(len(derived["annotations"]), 3)

    def test_grouped_split_keeps_duplicates_isolated(self) -> None:
        records = []
        classes = ("cardboard", "glass")
        image_id = 1
        for group_number in range(12):
            group_id = f"g-{group_number}"
            group_size = 2 if group_number == 0 else 1
            for _ in range(group_size):
                records.append(
                    {
                        "image_id": image_id,
                        "group_id": group_id,
                        "box_counts": {"cardboard": 1, "glass": 1},
                    }
                )
                image_id += 1

        image_split, report = make_grouped_splits(records, classes, seed=42)

        self.assertIsNotNone(image_split, report)
        group_splits = {
            group_id: {image_split[record["image_id"]] for record in records if record["group_id"] == group_id}
            for group_id in {record["group_id"] for record in records}
        }
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        split_sets = [
            {image_id for image_id, value in image_split.items() if value == split}
            for split in ("train", "validation", "test")
        ]
        self.assertTrue(split_sets[0].isdisjoint(split_sets[1]))
        self.assertTrue(split_sets[0].isdisjoint(split_sets[2]))
        self.assertTrue(split_sets[1].isdisjoint(split_sets[2]))

    def test_dataset_batch_has_detection_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (20, 16), "red").save(root / "one.jpg")
            Image.new("RGB", (20, 16), "blue").save(root / "two.jpg")
            annotations = {
                "images": [
                    {"id": 1, "width": 20, "height": 16, "file_name": "one.jpg"},
                    {"id": 2, "width": 20, "height": 16, "file_name": "two.jpg"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 5, 6]},
                    {"id": 2, "image_id": 2, "category_id": 2, "bbox": [3, 4, 7, 8]},
                ],
                "categories": [
                    {"id": 1, "name": "cardboard"},
                    {"id": 2, "name": "glass"},
                ],
            }
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
            dataset = TacoDetectionDataset(annotation_path, root)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=detection_collate_fn,
            )

            images, targets = next(iter(loader))

            self.assertEqual(len(images), 2)
            self.assertEqual(images[0].dtype, torch.float32)
            self.assertEqual(tuple(images[0].shape), (3, 16, 20))
            self.assertGreaterEqual(float(images[0].min()), 0.0)
            self.assertLessEqual(float(images[0].max()), 1.0)
            self.assertEqual(targets[0]["boxes"].dtype, torch.float32)
            self.assertEqual(targets[0]["labels"].dtype, torch.int64)
            self.assertEqual(targets[0]["image_id"].dtype, torch.int64)
            self.assertTrue(torch.all(targets[0]["boxes"][:, 2:] > targets[0]["boxes"][:, :2]))
            self.assertTrue(torch.all(targets[0]["labels"] >= 1))


if __name__ == "__main__":
    unittest.main()
