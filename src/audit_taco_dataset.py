"""Audit official TACO COCO annotations without downloading or training images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TACO_REPOSITORY_URL = "https://github.com/pedropro/TACO"
TACO_COMMIT = "70479ad9adc3da971a8ae75ac97eb1026910eec7"
TACO_ANNOTATION_URL = (
    f"{TACO_REPOSITORY_URL}/blob/{TACO_COMMIT}/data/annotations.json"
)
DEFAULT_ANNOTATIONS = Path("data/external/TACO-official/data/annotations.json")
DEFAULT_OUTPUT = Path("data/audits/taco_annotation_audit.json")


# Only exact official names with a sufficiently clear material are mapped.
# Composite or broad categories stay in REVIEW_OR_EXCLUSION_REASONS.
MATERIAL_CATEGORY_NAMES: dict[str, frozenset[str]] = {
    "cardboard": frozenset({"Corrugated carton", "Pizza box"}),
    "glass": frozenset({"Glass bottle", "Broken glass", "Glass cup", "Glass jar"}),
    "metal": frozenset(
        {
            "Aluminium foil",
            "Metal bottle cap",
            "Food Can",
            "Drink can",
            "Metal lid",
            "Pop tab",
            "Scrap metal",
        }
    ),
    "paper": frozenset(
        {
            "Paper cup",
            "Magazine paper",
            "Tissues",
            "Wrapping paper",
            "Normal paper",
            "Paper bag",
            "Paper straw",
        }
    ),
    "plastic": frozenset(
        {
            "Other plastic bottle",
            "Clear plastic bottle",
            "Plastic bottle cap",
            "Disposable plastic cup",
            "Other plastic cup",
            "Plastic lid",
            "Other plastic",
            "Plastic film",
            "Other plastic wrapper",
            "Polypropylene bag",
            "Other plastic container",
            "Plastic glooves",
            "Plastic utensils",
            "Plastic straw",
        }
    ),
}


REVIEW_OR_EXCLUSION_REASONS: dict[str, str] = {
    "Battery": "Hazardous composite object; no single target material is explicit.",
    "Aluminium blister pack": "Blister packs may combine aluminium and plastic.",
    "Carded blister pack": "Composite card-and-plastic packaging.",
    "Aerosol": "Container material is not explicit in the category name.",
    "Toilet tube": "Likely paperboard, but the official name does not state material.",
    "Other carton": "Broad carton category; composition is not explicit.",
    "Egg carton": "May be fibre, foam, or plastic; material is not explicit.",
    "Drink carton": "Commonly multilayer paper, plastic, and sometimes aluminium.",
    "Meal carton": "Composition is not explicit in the category name.",
    "Foam cup": "Foam is distinct/ambiguous for the proposed plastic target.",
    "Food waste": "Outside the five proposed recyclable material labels.",
    "Plastified paper bag": "Explicit paper/plastic composite.",
    "Six pack rings": "Material is not explicit in the category name.",
    "Garbage bag": "Material is not explicit in the category name.",
    "Single-use carrier bag": "Material is not explicit in the category name.",
    "Crisp packet": "Often multilayer packaging; material is not explicit.",
    "Spread tub": "Material is not explicit in the category name.",
    "Tupperware": "Brand/type name rather than an explicit material label.",
    "Disposable food container": "Material is not explicit in the category name.",
    "Foam food container": "Foam is distinct/ambiguous for the plastic target.",
    "Rope & strings": "Material may be natural fibre, plastic, or mixed.",
    "Shoe": "Composite object outside the five material labels.",
    "Squeezable tube": "May be plastic, metal, or multilayer packaging.",
    "Styrofoam piece": "Kept separate because the official taxonomy names foam.",
    "Unlabeled litter": "Broad unlabeled object; never silently mapped to trash.",
    "Cigarette": "Composite object outside the five material labels.",
}


class CocoAuditError(Exception):
    """Raised when the top-level COCO document cannot be audited safely."""


def _records(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise CocoAuditError(f"COCO field {key!r} must be a list.")
    return value


def _stable_value(value: Any) -> str:
    """Return a deterministic printable form for identifiers and metadata."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _duplicate_groups(values: list[Any]) -> list[dict[str, Any]]:
    counts = Counter(_stable_value(value) for value in values if value is not None)
    return [
        {"value": json.loads(value), "count": count}
        for value, count in sorted(counts.items())
        if count > 1
    ]


def _record_id_sort_key(record: dict[str, Any]) -> tuple[int, float, str]:
    value = record.get("id")
    if _finite_number(value):
        return (0, float(value), "")
    return (1, 0.0, _stable_value(value))


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _dimension_summary(images: list[dict[str, Any]]) -> dict[str, Any]:
    valid_dimensions: list[tuple[float, float]] = []
    for image in images:
        width = image.get("width")
        height = image.get("height")
        if _finite_number(width) and _finite_number(height) and width > 0 and height > 0:
            valid_dimensions.append((float(width), float(height)))

    if not valid_dimensions:
        return {"valid_images": 0, "invalid_images": len(images)}

    widths = [item[0] for item in valid_dimensions]
    heights = [item[1] for item in valid_dimensions]
    pixel_counts = [width * height for width, height in valid_dimensions]
    return {
        "valid_images": len(valid_dimensions),
        "invalid_images": len(images) - len(valid_dimensions),
        "width": {
            "minimum": min(widths),
            "median": statistics.median(widths),
            "maximum": max(widths),
        },
        "height": {
            "minimum": min(heights),
            "median": statistics.median(heights),
            "maximum": max(heights),
        },
        "megapixels": {
            "minimum": min(pixel_counts) / 1_000_000,
            "median": statistics.median(pixel_counts) / 1_000_000,
            "maximum": max(pixel_counts) / 1_000_000,
        },
    }


def _segmentation_type(segmentation: Any) -> str:
    if isinstance(segmentation, list):
        return "polygon_list"
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        return "rle_compressed" if isinstance(counts, str) else "rle_uncompressed"
    if segmentation is None:
        return "missing_or_null"
    return "other_or_malformed"


def _bbox_problem(
    annotation: dict[str, Any], image: dict[str, Any] | None
) -> str | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "bbox_not_four_values"
    if not all(_finite_number(value) for value in bbox):
        return "bbox_non_finite_or_non_numeric"
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        return "bbox_non_positive_size"
    if image is None:
        return "bbox_missing_image_reference"
    image_width = image.get("width")
    image_height = image.get("height")
    if not (
        _finite_number(image_width)
        and _finite_number(image_height)
        and image_width > 0
        and image_height > 0
    ):
        return "bbox_invalid_image_dimensions"
    tolerance = 1e-6
    if x < -tolerance or y < -tolerance:
        return "bbox_negative_origin"
    if x + width > float(image_width) + tolerance:
        return "bbox_exceeds_image_width"
    if y + height > float(image_height) + tolerance:
        return "bbox_exceeds_image_height"
    return None


def _material_for_name(category_name: str) -> str | None:
    for material, names in MATERIAL_CATEGORY_NAMES.items():
        if category_name in names:
            return material
    return None


def audit_coco(data: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable evidence audit for a COCO annotation object."""
    if not isinstance(data, dict):
        raise CocoAuditError("The COCO document must be a JSON object.")

    raw_images = _records(data, "images")
    raw_annotations = _records(data, "annotations")
    raw_categories = _records(data, "categories")
    images = [record for record in raw_images if isinstance(record, dict)]
    annotations = [record for record in raw_annotations if isinstance(record, dict)]
    categories = [record for record in raw_categories if isinstance(record, dict)]

    image_ids = [image.get("id") for image in images]
    category_ids = [category.get("id") for category in categories]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    image_by_key: dict[str, dict[str, Any]] = {}
    for image in images:
        image_by_key.setdefault(_stable_value(image.get("id")), image)
    category_by_key: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_by_key.setdefault(_stable_value(category.get("id")), category)

    instance_counts: Counter[str] = Counter()
    category_image_keys: defaultdict[str, set[str]] = defaultdict(set)
    image_annotation_counts: Counter[str] = Counter()
    segmentation_counts: Counter[str] = Counter()
    bbox_problem_counts: Counter[str] = Counter()
    invalid_bbox_records: list[dict[str, Any]] = []
    valid_bbox_indices: set[int] = set()
    missing_image_references = 0
    missing_category_references = 0
    polygon_parts = 0
    invalid_polygon_parts = 0
    malformed_annotation_count = len(raw_annotations) - len(annotations)
    if malformed_annotation_count:
        bbox_problem_counts["malformed_annotation_record"] = (
            malformed_annotation_count
        )

    for index, annotation in enumerate(annotations):
        image_key = _stable_value(annotation.get("image_id"))
        category_key = _stable_value(annotation.get("category_id"))
        image = image_by_key.get(image_key)
        if image is None:
            missing_image_references += 1
        else:
            image_annotation_counts[image_key] += 1
        if category_key not in category_by_key:
            missing_category_references += 1
        else:
            instance_counts[category_key] += 1
            if image is not None:
                category_image_keys[category_key].add(image_key)

        segmentation = annotation.get("segmentation")
        segmentation_counts[_segmentation_type(segmentation)] += 1
        if isinstance(segmentation, list):
            polygon_parts += len(segmentation)
            invalid_polygon_parts += sum(
                not isinstance(part, list)
                or len(part) < 6
                or len(part) % 2 != 0
                or not all(_finite_number(value) for value in part)
                for part in segmentation
            )

        problem = _bbox_problem(annotation, image)
        if problem is None:
            valid_bbox_indices.add(index)
        else:
            bbox_problem_counts[problem] += 1
            invalid_bbox_records.append(
                {
                    "annotation_id": annotation.get("id"),
                    "image_id": annotation.get("image_id"),
                    "category_id": annotation.get("category_id"),
                    "bbox": annotation.get("bbox"),
                    "reason": problem,
                }
            )

    category_rows: list[dict[str, Any]] = []
    mapped_categories: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    review_categories: list[dict[str, Any]] = []
    material_instance_counts: Counter[str] = Counter()
    material_valid_instance_counts: Counter[str] = Counter()
    material_image_keys: defaultdict[str, set[str]] = defaultdict(set)
    mapped_valid_image_keys: set[str] = set()
    mapped_valid_annotation_counts: Counter[str] = Counter()
    mapped_annotation_indices: set[int] = set()
    categories_without_predefined_decision: list[str] = []

    annotations_by_category: defaultdict[str, list[tuple[int, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for index, annotation in enumerate(annotations):
        annotations_by_category[_stable_value(annotation.get("category_id"))].append(
            (index, annotation)
        )

    for category in sorted(categories, key=_record_id_sort_key):
        category_key = _stable_value(category.get("id"))
        name = str(category.get("name", ""))
        row = {
            "id": category.get("id"),
            "name": name,
            "supercategory": category.get("supercategory"),
            "instance_count": instance_counts[category_key],
            "image_count": len(category_image_keys[category_key]),
        }
        category_rows.append(row)
        material = _material_for_name(name)
        if material is None:
            if name not in REVIEW_OR_EXCLUSION_REASONS:
                categories_without_predefined_decision.append(name)
            review_categories.append(
                {
                    **row,
                    "reason": REVIEW_OR_EXCLUSION_REASONS.get(
                        name,
                        "No explicit reviewed mapping exists for this category name.",
                    ),
                }
            )
            continue

        mapped_categories[material].append(row)
        for annotation_index, annotation in annotations_by_category[category_key]:
            mapped_annotation_indices.add(annotation_index)
            material_instance_counts[material] += 1
            if annotation_index in valid_bbox_indices:
                material_valid_instance_counts[material] += 1
                image_key = _stable_value(annotation.get("image_id"))
                material_image_keys[material].add(image_key)
                mapped_valid_image_keys.add(image_key)
                mapped_valid_annotation_counts[image_key] += 1

    url_fields = sorted(
        {
            key
            for image in images
            for key in image
            if key.endswith("_url")
        }
    )
    all_urls: list[Any] = []
    url_presence: dict[str, int] = {}
    url_hosts: Counter[str] = Counter()
    for field in url_fields:
        values = [image.get(field) for image in images if image.get(field)]
        all_urls.extend(values)
        url_presence[field] = len(values)
        for value in values:
            if isinstance(value, str):
                url_hosts[urlparse(value).netloc] += 1

    filename_values = [image.get("file_name") for image in images]
    image_license_values = Counter(
        "null" if image.get("license") is None else str(image.get("license"))
        for image in images
    )
    mapped_instance_total = len(mapped_annotation_indices)
    mapped_valid_instance_total = sum(material_valid_instance_counts.values())

    return {
        "record_counts": {
            "images": len(raw_images),
            "annotations": len(raw_annotations),
            "categories": len(raw_categories),
            "malformed_image_records": len(raw_images) - len(images),
            "malformed_annotation_records": len(raw_annotations) - len(annotations),
            "malformed_category_records": len(raw_categories) - len(categories),
            "images_with_multiple_annotated_objects": sum(
                count >= 2 for count in image_annotation_counts.values()
            ),
        },
        "categories": category_rows,
        "bounding_boxes": {
            "valid": len(valid_bbox_indices),
            "invalid": len(raw_annotations) - len(valid_bbox_indices),
            "invalid_reasons": dict(sorted(bbox_problem_counts.items())),
            "invalid_records": invalid_bbox_records,
        },
        "segmentations": {
            "types": dict(sorted(segmentation_counts.items())),
            "polygon_parts": polygon_parts,
            "invalid_polygon_parts": invalid_polygon_parts,
        },
        "image_dimensions": _dimension_summary(images),
        "missing_references": {
            "annotation_to_image": missing_image_references,
            "annotation_to_category": missing_category_references,
        },
        "duplicates": {
            "image_ids": _duplicate_groups(image_ids),
            "annotation_ids": _duplicate_groups(annotation_ids),
            "category_ids": _duplicate_groups(category_ids),
            "file_names": _duplicate_groups(filename_values),
            "image_urls_across_all_url_fields": _duplicate_groups(all_urls),
        },
        "source_metadata_present": {
            "info": data.get("info"),
            "licenses": data.get("licenses"),
            "image_license_values": dict(sorted(image_license_values.items())),
            "url_field_non_null_counts": url_presence,
            "url_host_occurrences": dict(sorted(url_hosts.items())),
        },
        "proposed_material_mapping": {
            "rule": (
                "Exact official category names only; ambiguous/composite/broad "
                "categories remain in review_or_exclusion."
            ),
            "categories_by_material": {
                material: mapped_categories[material]
                for material in MATERIAL_CATEGORY_NAMES
            },
            "review_or_exclusion": review_categories,
            "categories_without_predefined_decision": sorted(
                categories_without_predefined_decision
            ),
            "instance_counts_by_material": {
                material: material_instance_counts[material]
                for material in MATERIAL_CATEGORY_NAMES
            },
            "valid_bbox_instance_counts_by_material": {
                material: material_valid_instance_counts[material]
                for material in MATERIAL_CATEGORY_NAMES
            },
            "image_counts_by_material": {
                material: len(material_image_keys[material])
                for material in MATERIAL_CATEGORY_NAMES
            },
            "mapped_instances": mapped_instance_total,
            "mapped_valid_bbox_instances": mapped_valid_instance_total,
            "usable_detector_images": len(mapped_valid_image_keys),
            "usable_images_with_multiple_mapped_objects": sum(
                count >= 2 for count in mapped_valid_annotation_counts.values()
            ),
            "review_or_exclusion_instances": len(annotations)
            - mapped_instance_total,
        },
    }


def audit_file(annotation_path: Path) -> dict[str, Any]:
    """Load, hash, and audit one COCO JSON file."""
    try:
        file_bytes = annotation_path.read_bytes()
        data = json.loads(file_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CocoAuditError(f"Could not read COCO JSON {annotation_path}: {error}") from error

    result = audit_coco(data)
    result["source"] = {
        "repository_url": TACO_REPOSITORY_URL,
        "commit": TACO_COMMIT,
        "official_annotation_url": TACO_ANNOTATION_URL,
        "local_annotation_path": annotation_path.as_posix(),
        "annotation_sha256": hashlib.sha256(file_bytes).hexdigest(),
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print the evidence needed to assess a future detector milestone."""
    source = result["source"]
    records = result["record_counts"]
    boxes = result["bounding_boxes"]
    mapping = result["proposed_material_mapping"]
    duplicates = result["duplicates"]

    print(f"Official source: {source['official_annotation_url']}")
    print(f"Pinned commit: {source['commit']}")
    print(f"Annotation SHA-256: {source['annotation_sha256']}")
    print(
        f"Images: {records['images']}; annotations: {records['annotations']}; "
        f"categories: {records['categories']}"
    )
    print(
        "Images with multiple annotated objects: "
        f"{records['images_with_multiple_annotated_objects']}"
    )
    print(f"Bounding boxes: {boxes['valid']} valid; {boxes['invalid']} invalid")
    print(f"Invalid box reasons: {boxes['invalid_reasons']}")
    print(f"Segmentation types: {result['segmentations']['types']}")
    print(f"Missing references: {result['missing_references']}")
    dimensions = result["image_dimensions"]
    print(
        "Image dimensions: "
        f"width {dimensions['width']['minimum']:.0f}–"
        f"{dimensions['width']['maximum']:.0f} "
        f"(median {dimensions['width']['median']:.0f}); "
        f"height {dimensions['height']['minimum']:.0f}–"
        f"{dimensions['height']['maximum']:.0f} "
        f"(median {dimensions['height']['median']:.0f})"
    )
    metadata = result["source_metadata_present"]
    print(f"COCO licenses array: {metadata['licenses']}")
    print(f"Image license values: {metadata['image_license_values']}")
    print(f"Non-null image URL fields: {metadata['url_field_non_null_counts']}")
    print("\nOfficial categories and instance counts:")
    for category in result["categories"]:
        print(
            f"  {category['id']:>2}  {category['name']}: "
            f"{category['instance_count']} instances"
        )

    print("\nProposed conservative material mapping:")
    for material in MATERIAL_CATEGORY_NAMES:
        names = [
            category["name"]
            for category in mapping["categories_by_material"][material]
        ]
        print(
            f"  {material}: {mapping['instance_counts_by_material'][material]} "
            "instances; "
            f"{mapping['valid_bbox_instance_counts_by_material'][material]} "
            "valid boxes in "
            f"{mapping['image_counts_by_material'][material]} images"
        )
        print(f"    categories: {', '.join(names) or '(none)'}")
    print(
        f"Mapped valid boxes: {mapping['mapped_valid_bbox_instances']}; "
        f"usable detector images: {mapping['usable_detector_images']}"
    )
    print(
        "Usable images with multiple mapped objects: "
        f"{mapping['usable_images_with_multiple_mapped_objects']}"
    )
    print("\nReview/exclusion categories:")
    for category in mapping["review_or_exclusion"]:
        print(
            f"  {category['id']:>2}  {category['name']} "
            f"({category['instance_count']}): {category['reason']}"
        )

    print("\nDuplicate checks before any split:")
    for key, groups in duplicates.items():
        print(f"  {key}: {len(groups)} repeated value group(s)")
    print("No detector was trained or evaluated by this audit.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_file(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(result)
    print(f"\nSaved ignored local audit: {args.output}")


if __name__ == "__main__":
    main()
