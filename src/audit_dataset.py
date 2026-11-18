"""Audit the waste-sorting image dataset without modifying its files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "dataset_audit.json"
EXPECTED_CLASSES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)
JPEG_SUFFIXES = {".jpg", ".jpeg"}
HASH_CHUNK_SIZE = 1024 * 1024


def project_relative(path: Path) -> str:
    """Return a portable path relative to the project root."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def jpeg_files(class_directory: Path) -> list[Path]:
    """Find JPEG files recursively and return them in a stable order."""
    paths = (
        path
        for path in class_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in JPEG_SUFFIXES
    )
    return sorted(
        paths,
        key=lambda path: (
            path.relative_to(class_directory).as_posix().casefold(),
            path.relative_to(class_directory).as_posix(),
        ),
    )


def decode_image(path: Path) -> None:
    """Open an image and force Pillow to decode all of its pixel data."""
    with Image.open(path) as image:
        image.load()


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest from the file's original bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset() -> dict[str, Any]:
    """Audit expected class folders, image decodability, and duplicates."""
    class_counts: dict[str, dict[str, int]] = {}
    unreadable_files: list[dict[str, str]] = []
    hash_errors: list[dict[str, str]] = []
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_class_folders: list[str] = []

    for class_label in EXPECTED_CLASSES:
        class_directory = DATASET_ROOT / class_label
        counts = {"jpeg_files": 0, "readable": 0, "unreadable": 0}
        class_counts[class_label] = counts

        if not class_directory.is_dir():
            missing_class_folders.append(project_relative(class_directory))
            continue

        for image_path in jpeg_files(class_directory):
            relative_path = project_relative(image_path)
            counts["jpeg_files"] += 1

            try:
                decode_image(image_path)
            except Exception as error:  # Continue after any per-file Pillow/I/O error.
                counts["unreadable"] += 1
                unreadable_files.append(
                    {
                        "path": relative_path,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                counts["readable"] += 1

            try:
                file_hash = sha256_file(image_path)
            except Exception as error:  # A hash failure must not stop the audit.
                hash_errors.append(
                    {
                        "path": relative_path,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                hashes[file_hash].append(
                    {"path": relative_path, "class_label": class_label}
                )

    duplicate_groups: list[dict[str, Any]] = []
    for file_hash in sorted(hashes):
        files = hashes[file_hash]
        if len(files) < 2:
            continue

        class_labels = sorted({file["class_label"] for file in files})
        duplicate_groups.append(
            {
                "sha256": file_hash,
                "files": files,
                "class_labels": class_labels,
                "crosses_class_labels": len(class_labels) > 1,
            }
        )

    total_jpeg_files = sum(
        counts["jpeg_files"] for counts in class_counts.values()
    )
    total_readable = sum(counts["readable"] for counts in class_counts.values())
    total_unreadable = sum(
        counts["unreadable"] for counts in class_counts.values()
    )
    cross_class_duplicate_groups = sum(
        group["crosses_class_labels"] for group in duplicate_groups
    )

    return {
        "dataset_root": project_relative(DATASET_ROOT),
        "expected_classes": list(EXPECTED_CLASSES),
        "missing_class_folders": missing_class_folders,
        "counts_per_class": class_counts,
        "summary": {
            "total_jpeg_files": total_jpeg_files,
            "total_readable": total_readable,
            "total_unreadable": total_unreadable,
            "duplicate_groups": len(duplicate_groups),
            "cross_class_duplicate_groups": cross_class_duplicate_groups,
            "hash_errors": len(hash_errors),
        },
        "unreadable_files": unreadable_files,
        "hash_errors": hash_errors,
        "duplicate_groups": duplicate_groups,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a concise, human-readable audit summary."""
    print(f"Dataset audit: {report['dataset_root']}")
    print("Counts per class:")
    for class_label in EXPECTED_CLASSES:
        counts = report["counts_per_class"][class_label]
        print(
            f"  {class_label}: {counts['jpeg_files']} JPEG files "
            f"({counts['readable']} readable, {counts['unreadable']} unreadable)"
        )

    missing_folders = report["missing_class_folders"]
    if missing_folders:
        print("Missing class folders:")
        for folder in missing_folders:
            print(f"  - {folder}")
    else:
        print("Missing class folders: none")

    summary = report["summary"]
    print(f"Total readable files: {summary['total_readable']}")
    print(f"Total unreadable files: {summary['total_unreadable']}")
    print(f"Duplicate groups: {summary['duplicate_groups']}")
    print(
        "Cross-class duplicate groups: "
        f"{summary['cross_class_duplicate_groups']}"
    )
    if summary["hash_errors"]:
        print(f"Hash errors: {summary['hash_errors']}")
    print(f"Report saved to: {project_relative(REPORT_PATH)}")


def main() -> None:
    report = audit_dataset()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")
    print_summary(report)


if __name__ == "__main__":
    main()
