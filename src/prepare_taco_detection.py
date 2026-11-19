"""Prepare a verified local TACO detection dataset without training a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from audit_taco_dataset import (
    DEFAULT_ANNOTATIONS,
    MATERIAL_CATEGORY_NAMES,
    TACO_ANNOTATION_SHA256,
    TACO_ANNOTATION_URL,
    TACO_COMMIT,
    TACO_REPOSITORY_URL,
    _bbox_problem,
)


POLICY_A = "A"
POLICY_B = "B"
MATERIAL_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic")
OTHER_CLASS = "other"
DEFAULT_LICENSES = ("CC", "ODBL (c) OpenLitterMap & Contributors")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_SEED = 42
MIN_CLASS_IMAGES_PER_SPLIT = 2


class TacoPreparationError(Exception):
    """Raised when local TACO preparation cannot continue safely."""


@dataclass(frozen=True)
class PlannedObject:
    source_annotation: dict[str, Any]
    class_name: str


@dataclass(frozen=True)
class PlannedImage:
    source_image: dict[str, Any]
    objects: tuple[PlannedObject, ...]

    @property
    def image_id(self) -> int:
        return int(self.source_image["id"])


def _license_value(image: dict[str, Any]) -> str:
    value = image.get("license")
    return "null" if value is None else str(value)


def _material_lookup() -> dict[str, str]:
    return {
        category_name: material
        for material, category_names in MATERIAL_CATEGORY_NAMES.items()
        for category_name in category_names
    }


def load_pinned_annotations(path: Path) -> dict[str, Any]:
    """Load only the exact official annotation bytes audited by this project."""
    try:
        file_bytes = path.read_bytes()
    except OSError as error:
        raise TacoPreparationError(f"Could not read annotations {path}: {error}") from error
    actual_hash = hashlib.sha256(file_bytes).hexdigest()
    if actual_hash != TACO_ANNOTATION_SHA256:
        raise TacoPreparationError(
            "Annotation SHA-256 does not match the pinned official source: "
            f"expected {TACO_ANNOTATION_SHA256}, received {actual_hash}."
        )
    try:
        data = json.loads(file_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TacoPreparationError(f"Could not decode annotations: {error}") from error
    return data


def plan_policy(
    data: dict[str, Any], policy: str
) -> tuple[list[PlannedImage], dict[str, Any]]:
    """Apply whole-image Policy A or B before downloading any photographs."""
    if policy not in {POLICY_A, POLICY_B}:
        raise TacoPreparationError("policy must be A or B")
    images = data.get("images")
    annotations = data.get("annotations")
    categories = data.get("categories")
    if not all(isinstance(value, list) for value in (images, annotations, categories)):
        raise TacoPreparationError("images, annotations, and categories must be lists")

    category_names = {
        category.get("id"): str(category.get("name", ""))
        for category in categories
        if isinstance(category, dict)
    }
    material_by_name = _material_lookup()
    annotations_by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    orphan_annotations = 0
    image_ids = {
        image.get("id") for image in images if isinstance(image, dict)
    }
    for annotation in annotations:
        if not isinstance(annotation, dict):
            orphan_annotations += 1
            continue
        if annotation.get("image_id") not in image_ids:
            orphan_annotations += 1
            continue
        annotations_by_image[annotation.get("image_id")].append(annotation)

    planned: list[PlannedImage] = []
    box_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    multi_object_images = 0

    for image in images:
        if not isinstance(image, dict):
            excluded["malformed_image_record"] += 1
            continue
        image_annotations = annotations_by_image.get(image.get("id"), [])
        if not image_annotations:
            excluded["no_annotations"] += 1
            continue

        objects: list[PlannedObject] = []
        invalid_image = False
        ambiguous_image = False
        for annotation in image_annotations:
            category_name = category_names.get(annotation.get("category_id"))
            if category_name is None:
                invalid_image = True
                break
            if _bbox_problem(annotation, image) is not None:
                invalid_image = True
                break
            material = material_by_name.get(category_name)
            if material is None:
                ambiguous_image = True
                class_name = OTHER_CLASS
            else:
                class_name = material
            objects.append(PlannedObject(annotation, class_name))

        if invalid_image:
            excluded["invalid_box_or_reference_image"] += 1
            continue
        if policy == POLICY_A and ambiguous_image:
            excluded["contains_review_or_exclusion_category"] += 1
            continue

        planned_image = PlannedImage(image, tuple(objects))
        planned.append(planned_image)
        box_counts.update(obj.class_name for obj in objects)
        license_counts[_license_value(image)] += 1
        if len(objects) >= 2:
            multi_object_images += 1

    summary = {
        "policy": policy,
        "image_count": len(planned),
        "box_count": sum(box_counts.values()),
        "box_counts_by_class": dict(sorted(box_counts.items())),
        "multi_object_image_count": multi_object_images,
        "image_counts_by_license": dict(sorted(license_counts.items())),
        "excluded_image_counts": dict(sorted(excluded.items())),
        "orphan_or_malformed_annotation_count": orphan_annotations,
    }
    return planned, summary


def filter_license_cohort(
    planned: list[PlannedImage], allowed_licenses: set[str]
) -> tuple[list[PlannedImage], dict[str, Any]]:
    """Select explicitly license-tagged images without interpreting the license."""
    selected = [
        item
        for item in planned
        if _license_value(item.source_image) in allowed_licenses
    ]
    box_counts = Counter(
        obj.class_name for item in selected for obj in item.objects
    )
    license_counts = Counter(_license_value(item.source_image) for item in selected)
    return selected, {
        "image_count": len(selected),
        "box_count": sum(box_counts.values()),
        "box_counts_by_class": dict(sorted(box_counts.items())),
        "multi_object_image_count": sum(len(item.objects) >= 2 for item in selected),
        "image_counts_by_license": dict(sorted(license_counts.items())),
        "allowed_license_values": sorted(allowed_licenses),
        "license_caution": (
            "License strings are preserved metadata, not a legal conclusion. "
            "CC is not interpreted as a specific Creative Commons variant, and "
            "null is not treated as permission to redistribute."
        ),
    }


def print_plan(summary: dict[str, Any], title: str) -> None:
    print(title)
    print(f"  Images: {summary['image_count']}")
    print(f"  Boxes: {summary['box_count']} {summary['box_counts_by_class']}")
    print(f"  Multi-object images: {summary['multi_object_image_count']}")
    print(f"  Image license values: {summary['image_counts_by_license']}")
    if "excluded_image_counts" in summary:
        print(f"  Excluded images: {summary['excluded_image_counts']}")


def _safe_image_path(image_root: Path, file_name: str) -> Path:
    root = image_root.resolve()
    candidate = (root / Path(file_name)).resolve()
    if candidate != root and root not in candidate.parents:
        raise TacoPreparationError(f"Unsafe image file name: {file_name!r}")
    return candidate


def _download_file(
    url: str,
    destination: Path,
    retries: int,
    timeout_seconds: float,
) -> bool:
    """Download atomically and return True; existing files return False."""
    if destination.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "WasteSorting-TACO-local-preparer/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if getattr(response, "status", 200) != 200:
                    raise OSError(f"HTTP {getattr(response, 'status', 'unknown')}")
                with part_path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            part_path.replace(destination)
            return True
        except Exception as error:  # Network libraries expose several exception types.
            last_error = error
            if part_path.exists():
                part_path.unlink()
            if attempt < retries:
                time.sleep(min(4.0, float(2 ** (attempt - 1))))
    raise TacoPreparationError(
        f"Download failed after {retries} attempts for {url}: {last_error}"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    gray = ImageOps.grayscale(image)
    difference = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(difference.tobytes())
    difference_bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            difference_bits = (difference_bits << 1) | int(left > right)

    average_image = gray.resize((8, 8), Image.Resampling.LANCZOS)
    average_pixels = list(average_image.tobytes())
    average_value = sum(average_pixels) / len(average_pixels)
    average_bits = 0
    for value in average_pixels:
        average_bits = (average_bits << 1) | int(value >= average_value)
    return f"{difference_bits:016x}", f"{average_bits:016x}"


def download_and_verify_one(
    planned: PlannedImage,
    image_root: Path,
    retries: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    image = planned.source_image
    file_name = str(image.get("file_name", ""))
    url = image.get("flickr_url")
    if not file_name or not isinstance(url, str) or not url:
        raise TacoPreparationError("Image record lacks file_name or flickr_url")
    path = _safe_image_path(image_root, file_name)
    downloaded = _download_file(url, path, retries, timeout_seconds)
    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            oriented.load()
            actual_size = oriented.size
            rgb = oriented.convert("RGB")
            difference_hash, average_hash = _perceptual_hashes(rgb)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise TacoPreparationError(f"Could not fully decode {path}: {error}") from error

    expected_size = (int(image.get("width", 0)), int(image.get("height", 0)))
    if actual_size != expected_size:
        raise TacoPreparationError(
            f"Size mismatch for {file_name}: expected {expected_size}, got {actual_size}"
        )
    return {
        "image_id": planned.image_id,
        "file_name": file_name,
        "source_url": url,
        "license": image.get("license"),
        "width": actual_size[0],
        "height": actual_size[1],
        "sha256": _hash_file(path),
        "difference_hash": difference_hash,
        "average_hash": average_hash,
        "downloaded_now": downloaded,
        "byte_size": path.stat().st_size,
    }


def download_and_verify(
    planned: list[PlannedImage],
    image_root: Path,
    workers: int,
    retries: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Download concurrently within a fixed worker limit and retain all failures."""
    verified: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                download_and_verify_one,
                item,
                image_root,
                retries,
                timeout_seconds,
            ): item
            for item in planned
        }
        completed = 0
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            completed += 1
            try:
                verified.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "image_id": item.image_id,
                        "file_name": item.source_image.get("file_name"),
                        "source_url": item.source_image.get("flickr_url"),
                        "license": item.source_image.get("license"),
                        "error": str(error),
                    }
                )
            if completed % 25 == 0 or completed == len(planned):
                print(
                    f"Download verification: {completed}/{len(planned)}; "
                    f"verified={len(verified)}; failed={len(failures)}",
                    flush=True,
                )
    verified.sort(key=lambda item: item["image_id"])
    failures.sort(key=lambda item: item["image_id"])
    return verified, failures


def _hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def group_duplicate_images(
    verified: list[dict[str, Any]],
) -> tuple[dict[int, str], dict[str, Any]]:
    """Group exact bytes and conservative likely visual duplicates."""
    parent = list(range(len(verified)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    exact_by_hash: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(verified):
        exact_by_hash[record["sha256"]].append(index)
    exact_groups = []
    for indices in exact_by_hash.values():
        if len(indices) >= 2:
            exact_groups.append([verified[index]["image_id"] for index in indices])
            for index in indices[1:]:
                union(indices[0], index)

    likely_pairs: list[dict[str, Any]] = []
    for first_index, first in enumerate(verified):
        first_ratio = first["width"] / first["height"]
        for second_index in range(first_index + 1, len(verified)):
            second = verified[second_index]
            if first["sha256"] == second["sha256"]:
                continue
            second_ratio = second["width"] / second["height"]
            if abs(first_ratio - second_ratio) / max(first_ratio, second_ratio) > 0.01:
                continue
            difference_distance = _hash_distance(
                first["difference_hash"], second["difference_hash"]
            )
            average_distance = _hash_distance(
                first["average_hash"], second["average_hash"]
            )
            if difference_distance <= 4 and average_distance <= 4:
                union(first_index, second_index)
                likely_pairs.append(
                    {
                        "image_ids": [first["image_id"], second["image_id"]],
                        "difference_hash_distance": difference_distance,
                        "average_hash_distance": average_distance,
                    }
                )

    grouped_indices: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(verified)):
        grouped_indices[find(index)].append(index)
    group_map: dict[int, str] = {}
    duplicate_groups: list[dict[str, Any]] = []
    for group_number, indices in enumerate(
        sorted(grouped_indices.values(), key=lambda values: verified[values[0]]["image_id"]),
        start=1,
    ):
        group_id = f"group-{group_number:05d}"
        image_ids = [verified[index]["image_id"] for index in indices]
        for image_id in image_ids:
            group_map[image_id] = group_id
        if len(indices) >= 2:
            duplicate_groups.append({"group_id": group_id, "image_ids": image_ids})

    return group_map, {
        "exact_duplicate_groups": exact_groups,
        "likely_visual_duplicate_pairs": likely_pairs,
        "combined_duplicate_groups": duplicate_groups,
        "visual_hash_rule": (
            "Aspect ratio within 1%, dHash Hamming distance <=4, and average-hash "
            "Hamming distance <=4. This is a screening heuristic, not proof."
        ),
    }


def make_grouped_splits(
    records: list[dict[str, Any]],
    class_names: tuple[str, ...],
    seed: int = SPLIT_SEED,
) -> tuple[dict[int, str] | None, dict[str, Any]]:
    """Create deterministic duplicate-grouped, multi-label-aware image splits."""
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["group_id"])].append(record)

    class_group_counts = {
        class_name: sum(
            any(member["box_counts"].get(class_name, 0) > 0 for member in members)
            for members in groups.values()
        )
        for class_name in class_names
    }
    insufficient = {
        name: count
        for name, count in class_group_counts.items()
        if count < len(SPLIT_NAMES) * MIN_CLASS_IMAGES_PER_SPLIT
    }
    if insufficient:
        return None, {
            "created": False,
            "reason": "Insufficient independent duplicate groups for each split.",
            "class_group_counts": class_group_counts,
            "insufficient_classes": insufficient,
        }

    group_rows: list[dict[str, Any]] = []
    total_class_boxes: Counter[str] = Counter()
    for group_id, members in groups.items():
        boxes = Counter()
        for member in members:
            boxes.update(member["box_counts"])
        total_class_boxes.update(boxes)
        group_rows.append(
            {
                "group_id": group_id,
                "members": members,
                "image_count": len(members),
                "box_counts": boxes,
            }
        )

    randomizer = random.Random(seed)
    randomizer.shuffle(group_rows)
    group_rows.sort(
        key=lambda group: (
            -max(
                (
                    group["box_counts"].get(name, 0)
                    / max(1, total_class_boxes[name])
                )
                for name in class_names
            ),
            -group["image_count"],
        )
    )

    total_images = len(records)
    target_images = {
        split: total_images * SPLIT_PROPORTIONS[split] for split in SPLIT_NAMES
    }
    target_boxes = {
        split: {
            name: total_class_boxes[name] * SPLIT_PROPORTIONS[split]
            for name in class_names
        }
        for split in SPLIT_NAMES
    }
    split_images: Counter[str] = Counter()
    split_boxes: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_NAMES
    }
    group_split: dict[str, str] = {}

    for group in group_rows:
        costs: list[tuple[float, str]] = []
        for split in SPLIT_NAMES:
            image_cost_after = (
                split_images[split] + group["image_count"] - target_images[split]
            ) ** 2 / max(1.0, target_images[split])
            image_cost_before = (
                split_images[split] - target_images[split]
            ) ** 2 / max(1.0, target_images[split])
            class_cost_change = sum(
                (
                    (
                        split_boxes[split][name]
                        + group["box_counts"].get(name, 0)
                        - target_boxes[split][name]
                    )
                    ** 2
                    - (split_boxes[split][name] - target_boxes[split][name]) ** 2
                )
                / max(1.0, target_boxes[split][name])
                for name in class_names
            )
            costs.append(
                (image_cost_after - image_cost_before + class_cost_change, split)
            )
        _, selected_split = min(costs, key=lambda item: (item[0], item[1]))
        group_split[group["group_id"]] = selected_split
        split_images[selected_split] += group["image_count"]
        split_boxes[selected_split].update(group["box_counts"])

    image_split = {
        int(member["image_id"]): group_split[group_id]
        for group_id, members in groups.items()
        for member in members
    }
    split_summary: dict[str, Any] = {}
    inadequate_split_classes: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        split_records = [
            record for record in records if image_split[int(record["image_id"])] == split
        ]
        image_counts_by_class = {
            name: sum(record["box_counts"].get(name, 0) > 0 for record in split_records)
            for name in class_names
        }
        boxes = Counter()
        for record in split_records:
            boxes.update(record["box_counts"])
        multi_count = sum(sum(record["box_counts"].values()) >= 2 for record in split_records)
        split_summary[split] = {
            "images": len(split_records),
            "boxes": sum(boxes.values()),
            "box_counts_by_class": dict(sorted(boxes.items())),
            "image_counts_by_class": image_counts_by_class,
            "multi_object_images": multi_count,
        }
        for name, count in image_counts_by_class.items():
            if count < MIN_CLASS_IMAGES_PER_SPLIT:
                inadequate_split_classes.append(
                    {"split": split, "class_name": name, "image_count": count}
                )

    if inadequate_split_classes:
        return None, {
            "created": False,
            "reason": "A class was not meaningfully represented in every split.",
            "class_group_counts": class_group_counts,
            "candidate_split_summary": split_summary,
            "inadequate_split_classes": inadequate_split_classes,
        }
    return image_split, {
        "created": True,
        "seed": seed,
        "proportions": SPLIT_PROPORTIONS,
        "class_group_counts": class_group_counts,
        "splits": split_summary,
    }


def build_derived_coco(
    planned: list[PlannedImage],
    verified_by_id: dict[int, dict[str, Any]],
    policy: str,
) -> dict[str, Any]:
    """Assign unique derived annotation IDs while preserving every source ID."""
    classes = MATERIAL_CLASSES + ((OTHER_CLASS,) if policy == POLICY_B else ())
    label_by_name = {name: index for index, name in enumerate(classes, start=1)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    derived_annotation_id = 1

    for planned_image in sorted(planned, key=lambda item: item.image_id):
        verification = verified_by_id.get(planned_image.image_id)
        if verification is None:
            continue
        source = planned_image.source_image
        images.append(
            {
                "id": planned_image.image_id,
                "width": verification["width"],
                "height": verification["height"],
                "file_name": source["file_name"],
                "license": source.get("license"),
                "source_image_id": source.get("id"),
                "source_url": source.get("flickr_url"),
                "source_resized_url": source.get("flickr_640_url"),
                "sha256": verification["sha256"],
                "difference_hash": verification["difference_hash"],
                "average_hash": verification["average_hash"],
            }
        )
        for obj in planned_image.objects:
            source_annotation = obj.source_annotation
            annotations.append(
                {
                    "id": derived_annotation_id,
                    "image_id": planned_image.image_id,
                    "category_id": label_by_name[obj.class_name],
                    "bbox": source_annotation["bbox"],
                    "area": source_annotation.get("area"),
                    "iscrowd": source_annotation.get("iscrowd", 0),
                    "segmentation": source_annotation.get("segmentation"),
                    "source_annotation_id": source_annotation.get("id"),
                    "source_category_id": source_annotation.get("category_id"),
                }
            )
            derived_annotation_id += 1

    return {
        "info": {
            "description": "Local derived TACO detection annotations",
            "policy": policy,
            "source_repository": TACO_REPOSITORY_URL,
            "source_commit": TACO_COMMIT,
            "source_annotation_url": TACO_ANNOTATION_URL,
            "source_annotation_sha256": TACO_ANNOTATION_SHA256,
            "license_caution": (
                "Image license values are preserved without legal interpretation."
            ),
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": label_by_name[name], "name": name, "supercategory": "material"}
            for name in classes
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "file_name", "source_url", "license", "error"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def write_manifests(
    output_dir: Path,
    records: list[dict[str, Any]],
    image_split: dict[int, str],
) -> None:
    fieldnames = [
        "image_id",
        "file_name",
        "split",
        "license",
        "source_url",
        "group_id",
        "sha256",
    ]
    combined_path = output_dir / "manifests" / "dataset_splits.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "split": image_split[int(record["image_id"])],
            "license": "" if record["license"] is None else record["license"],
            "source_url": record["source_url"],
            "group_id": record["group_id"],
            "sha256": record["sha256"],
        }
        for record in sorted(records, key=lambda item: item["image_id"])
    ]
    with combined_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for split in SPLIT_NAMES:
        split_path = output_dir / "manifests" / f"{split}.csv"
        with split_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taco_detection"))
    parser.add_argument("--policy", choices=(POLICY_A, POLICY_B), default=POLICY_B)
    parser.add_argument("--license", action="append", dest="licenses")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.timeout <= 0:
        raise TacoPreparationError("workers, retries, and timeout must be positive")
    data = load_pinned_annotations(args.annotations)
    policy_a, policy_a_summary = plan_policy(data, POLICY_A)
    policy_b, policy_b_summary = plan_policy(data, POLICY_B)
    print_plan(policy_a_summary, "Policy A — five-material-only complete images")
    print_plan(policy_b_summary, "Policy B — five materials plus explicit OTHER")
    if args.plan_only:
        print("Plan-only mode: no images downloaded and no manifests created.")
        return

    selected_plan = policy_a if args.policy == POLICY_A else policy_b
    allowed_licenses = set(args.licenses or DEFAULT_LICENSES)
    selected, selected_summary = filter_license_cohort(selected_plan, allowed_licenses)
    print_plan(selected_summary, f"Selected local cohort — Policy {args.policy}")
    print(selected_summary["license_caution"])

    verified, failures = download_and_verify(
        selected,
        args.output_dir / "images",
        args.workers,
        args.retries,
        args.timeout,
    )
    _write_failures(args.output_dir / "download_failures.csv", failures)
    verified_by_id = {int(record["image_id"]): record for record in verified}
    verified_plan = [item for item in selected if item.image_id in verified_by_id]

    group_map, duplicate_report = group_duplicate_images(verified)
    plan_by_id = {item.image_id: item for item in verified_plan}
    split_records: list[dict[str, Any]] = []
    for record in verified:
        planned_image = plan_by_id[int(record["image_id"])]
        box_counts = Counter(obj.class_name for obj in planned_image.objects)
        split_records.append(
            {
                **record,
                "group_id": group_map[int(record["image_id"])],
                "box_counts": dict(box_counts),
            }
        )

    class_names = MATERIAL_CLASSES + ((OTHER_CLASS,) if args.policy == POLICY_B else ())
    image_split, split_report = make_grouped_splits(split_records, class_names)
    derived_coco = build_derived_coco(verified_plan, verified_by_id, args.policy)
    _write_json(args.output_dir / f"annotations_policy_{args.policy.lower()}.json", derived_coco)
    if image_split is not None:
        write_manifests(args.output_dir, split_records, image_split)

    verified_box_counts = Counter(
        obj.class_name for item in verified_plan for obj in item.objects
    )
    verified_license_counts = Counter(
        _license_value(item.source_image) for item in verified_plan
    )
    source_annotation_counts = Counter(
        annotation.get("id")
        for annotation in data["annotations"]
        if isinstance(annotation, dict)
    )
    derived_annotations_by_source: defaultdict[Any, list[int]] = defaultdict(list)
    for annotation in derived_coco["annotations"]:
        derived_annotations_by_source[annotation["source_annotation_id"]].append(
            annotation["id"]
        )
    repeated_source_id_provenance = [
        {
            "source_annotation_id": source_id,
            "official_source_occurrences": source_count,
            "selected_derived_occurrences": len(
                derived_annotations_by_source.get(source_id, [])
            ),
            "derived_annotation_ids": derived_annotations_by_source.get(source_id, []),
        }
        for source_id, source_count in sorted(source_annotation_counts.items())
        if source_count > 1
    ]
    report = {
        "source": {
            "repository": TACO_REPOSITORY_URL,
            "commit": TACO_COMMIT,
            "annotation_url": TACO_ANNOTATION_URL,
            "annotation_sha256": TACO_ANNOTATION_SHA256,
        },
        "policy_a_plan": policy_a_summary,
        "policy_b_plan": policy_b_summary,
        "selected_policy": args.policy,
        "selected_cohort_before_download": selected_summary,
        "download": {
            "requested_images": len(selected),
            "verified_images": len(verified),
            "downloaded_now": sum(record["downloaded_now"] for record in verified),
            "reused_existing": sum(not record["downloaded_now"] for record in verified),
            "failed_images": len(failures),
            "verified_boxes": sum(verified_box_counts.values()),
            "verified_box_counts_by_class": dict(sorted(verified_box_counts.items())),
            "verified_image_counts_by_license": dict(sorted(verified_license_counts.items())),
            "verified_multi_object_images": sum(
                len(item.objects) >= 2 for item in verified_plan
            ),
        },
        "duplicates": duplicate_report,
        "split": split_report,
        "derived_coco": {
            "images": len(derived_coco["images"]),
            "annotations": len(derived_coco["annotations"]),
            "unique_annotation_ids": len(
                {annotation["id"] for annotation in derived_coco["annotations"]}
            ),
            "source_annotation_ids_preserved": True,
            "repeated_source_id_provenance": repeated_source_id_provenance,
        },
    }
    _write_json(args.output_dir / "preparation_report.json", report)

    print(
        f"Verified local images: {len(verified)}/{len(selected)}; failures: {len(failures)}"
    )
    print(
        f"Verified boxes: {sum(verified_box_counts.values())} "
        f"{dict(sorted(verified_box_counts.items()))}"
    )
    print(f"Verified licenses: {dict(sorted(verified_license_counts.items()))}")
    print(
        f"Exact duplicate groups: {len(duplicate_report['exact_duplicate_groups'])}; "
        "likely visual pairs: "
        f"{len(duplicate_report['likely_visual_duplicate_pairs'])}"
    )
    if image_split is None:
        print(f"SPLIT NOT CREATED: {split_report['reason']}")
        print(json.dumps(split_report, indent=2))
    else:
        print("Deterministic grouped split created with seed 42:")
        print(json.dumps(split_report["splits"], indent=2))
    print("No detector was trained or evaluated.")


if __name__ == "__main__":
    main()
