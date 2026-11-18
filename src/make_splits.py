"""Create reproducible train, validation, and test dataset manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
AUDIT_PATH = PROJECT_ROOT / "artifacts" / "dataset_audit.json"
SPLITS_PATH = PROJECT_ROOT / "artifacts" / "dataset_splits.csv"
EXCLUSIONS_PATH = PROJECT_ROOT / "artifacts" / "excluded_images.csv"

EXPECTED_CLASSES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)
SPLIT_PERCENTAGES = (("train", 70), ("validation", 15), ("test", 15))
RANDOM_SEED = 42
JPEG_SUFFIXES = {".jpg", ".jpeg"}
HASH_CHUNK_SIZE = 1024 * 1024


class SplitError(Exception):
    """An expected problem that prevents safe manifest creation."""


def project_relative(path: Path) -> str:
    """Return a portable path relative to the project root."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def find_jpegs(class_directory: Path) -> list[Path]:
    """Find JPEG files recursively in a stable order."""
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


def sha256_file(path: Path) -> str:
    """Hash a file's bytes without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_audit() -> dict[str, Any]:
    """Load the audit report and reject conditions that make splitting unsafe."""
    try:
        with AUDIT_PATH.open("r", encoding="utf-8") as audit_file:
            audit = json.load(audit_file)
    except FileNotFoundError as error:
        raise SplitError(
            f"Audit report not found: {project_relative(AUDIT_PATH)}. "
            "Run src/audit_dataset.py first."
        ) from error
    except json.JSONDecodeError as error:
        raise SplitError(f"Audit report is not valid JSON: {error}") from error
    except OSError as error:
        raise SplitError(f"Could not read the audit report: {error}") from error

    if not isinstance(audit, dict):
        raise SplitError("Audit report must contain a JSON object.")

    if audit.get("expected_classes") != list(EXPECTED_CLASSES):
        raise SplitError(
            "Audit report does not list the six expected classes in the "
            "expected order. Re-run src/audit_dataset.py."
        )

    if audit.get("dataset_root") != project_relative(DATASET_ROOT):
        raise SplitError(
            "Audit report points to a different dataset root. "
            "Re-run src/audit_dataset.py."
        )

    missing_folders = audit.get("missing_class_folders")
    if not isinstance(missing_folders, list):
        raise SplitError("Audit report has an invalid missing_class_folders field.")
    if missing_folders:
        raise SplitError(
            "Audit reports missing class folders: " + ", ".join(missing_folders)
        )

    summary = audit.get("summary")
    if not isinstance(summary, dict):
        raise SplitError("Audit report is missing its summary object.")

    unreadable_count = summary.get("total_unreadable")
    if unreadable_count != 0 or audit.get("unreadable_files"):
        raise SplitError(
            f"Audit reports {unreadable_count!r} unreadable file(s). "
            "Resolve them and re-run the audit before making splits."
        )

    hash_error_count = summary.get("hash_errors")
    if hash_error_count != 0 or audit.get("hash_errors"):
        raise SplitError(
            f"Audit reports {hash_error_count!r} hashing error(s). "
            "Resolve them and re-run the audit before making splits."
        )

    if not isinstance(audit.get("counts_per_class"), dict):
        raise SplitError("Audit report is missing per-class counts.")
    if not isinstance(audit.get("duplicate_groups"), list):
        raise SplitError("Audit report is missing duplicate group details.")

    return audit


def scan_dataset(audit: dict[str, Any]) -> dict[str, str]:
    """Return relative JPEG paths mapped to labels and check audit counts."""
    path_labels: dict[str, str] = {}

    for class_label in EXPECTED_CLASSES:
        class_directory = DATASET_ROOT / class_label
        if not class_directory.is_dir():
            raise SplitError(
                f"Class folder is now missing: {project_relative(class_directory)}. "
                "Re-run the audit."
            )

        image_paths = find_jpegs(class_directory)
        audit_class_counts = audit["counts_per_class"].get(class_label)
        if not isinstance(audit_class_counts, dict):
            raise SplitError(f"Audit counts are missing for class {class_label!r}.")

        audited_count = audit_class_counts.get("readable")
        if audited_count != len(image_paths):
            raise SplitError(
                f"Class {class_label!r} now contains {len(image_paths)} JPEG(s), "
                f"but the audit reports {audited_count!r} readable JPEG(s). "
                "The audit is stale; re-run src/audit_dataset.py."
            )

        for image_path in image_paths:
            relative_path = project_relative(image_path)
            if relative_path in path_labels:
                raise SplitError(f"JPEG was scanned more than once: {relative_path}")
            path_labels[relative_path] = class_label

    return path_labels


def cross_class_exclusions(
    audit: dict[str, Any], path_labels: dict[str, str]
) -> list[dict[str, str]]:
    """Build exclusion records from cross-class duplicate groups."""
    exclusions: list[dict[str, str]] = []
    excluded_paths: set[str] = set()

    for group in audit["duplicate_groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("files"), list):
            raise SplitError("Audit report contains an invalid duplicate group.")

        group_labels = {
            file_record.get("class_label")
            for file_record in group["files"]
            if isinstance(file_record, dict)
        }
        crosses_classes = len(group_labels) > 1
        if group.get("crosses_class_labels") is not crosses_classes:
            raise SplitError(
                "Audit duplicate flags are inconsistent. Re-run src/audit_dataset.py."
            )
        if not crosses_classes:
            continue

        file_hash = group.get("sha256")
        if not isinstance(file_hash, str):
            raise SplitError("A duplicate group is missing its SHA-256 hash.")

        for file_record in group["files"]:
            if not isinstance(file_record, dict):
                raise SplitError("A duplicate group contains an invalid file record.")
            relative_path = file_record.get("path")
            class_label = file_record.get("class_label")
            if not isinstance(relative_path, str) or not isinstance(class_label, str):
                raise SplitError("A duplicate file record is missing path or class.")
            if path_labels.get(relative_path) != class_label:
                raise SplitError(
                    f"Audited duplicate is absent or has changed class: {relative_path}. "
                    "Re-run src/audit_dataset.py."
                )
            if relative_path in excluded_paths:
                raise SplitError(
                    f"Audit lists the same exclusion more than once: {relative_path}"
                )

            excluded_paths.add(relative_path)
            exclusions.append(
                {
                    "relative_path": relative_path,
                    "class_label": class_label,
                    "reason": (
                        "cross-class byte-identical duplicate; "
                        f"sha256={file_hash}"
                    ),
                }
            )

    return sorted(exclusions, key=lambda row: row["relative_path"])


def target_counts(total: int) -> dict[str, int]:
    """Use the largest-remainder method for whole-number split targets."""
    counts: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []

    for order, (split, percentage) in enumerate(SPLIT_PERCENTAGES):
        count, remainder = divmod(total * percentage, 100)
        counts[split] = count
        remainders.append((remainder, order, split))

    remaining = total - sum(counts.values())
    for _, _, split in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        counts[split] += 1
        remaining -= 1

    return counts


def hash_eligible_files(
    path_labels: dict[str, str], excluded_paths: set[str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Hash eligible files and group their relative paths by digest."""
    path_hashes: dict[str, str] = {}
    hash_paths: dict[str, list[str]] = defaultdict(list)

    # Exact hashes detect identical bytes, but cannot detect visually similar images.
    for relative_path in sorted(path_labels):
        if relative_path in excluded_paths:
            continue
        try:
            file_hash = sha256_file(PROJECT_ROOT / Path(relative_path))
        except OSError as error:
            raise SplitError(f"Could not hash {relative_path}: {error}") from error
        path_hashes[relative_path] = file_hash
        hash_paths[file_hash].append(relative_path)

    for file_hash, relative_paths in hash_paths.items():
        labels = {path_labels[path] for path in relative_paths}
        if len(labels) > 1:
            joined_paths = ", ".join(relative_paths)
            raise SplitError(
                "Found a cross-class byte-identical group that was not excluded by "
                f"the audit ({file_hash}): {joined_paths}. Re-run the audit."
            )

    return path_hashes, hash_paths


def build_split_rows(
    path_labels: dict[str, str], hash_paths: dict[str, list[str]]
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Assign exact-duplicate groups to splits reproducibly by class."""
    randomizer = random.Random(RANDOM_SEED)
    rows: list[dict[str, str]] = []
    counts_by_class: dict[str, dict[str, int]] = {}

    groups_by_class: dict[str, list[list[str]]] = {
        class_label: [] for class_label in EXPECTED_CLASSES
    }
    for relative_paths in hash_paths.values():
        class_label = path_labels[relative_paths[0]]
        groups_by_class[class_label].append(sorted(relative_paths))

    for class_label in EXPECTED_CLASSES:
        groups = sorted(groups_by_class[class_label], key=lambda group: group[0])
        randomizer.shuffle(groups)
        eligible_count = sum(len(group) for group in groups)
        targets = target_counts(eligible_count)

        if any(targets[split] == 0 for split, _ in SPLIT_PERCENTAGES):
            raise SplitError(
                f"Class {class_label!r} has too few eligible images to populate "
                "train, validation, and test splits."
            )

        split_counts = {split: 0 for split, _ in SPLIT_PERCENTAGES}
        counts_by_class[class_label] = split_counts

        for group in groups:
            # Keep byte-identical files together while filling each target evenly.
            chosen_split = min(
                (split for split, _ in SPLIT_PERCENTAGES),
                key=lambda split: (
                    (split_counts[split] + len(group)) / targets[split],
                    next(
                        index
                        for index, (name, _) in enumerate(SPLIT_PERCENTAGES)
                        if name == split
                    ),
                ),
            )
            for relative_path in group:
                rows.append(
                    {
                        "relative_path": relative_path,
                        "class_label": class_label,
                        "split": chosen_split,
                    }
                )
            split_counts[chosen_split] += len(group)

    rows.sort(key=lambda row: (row["class_label"], row["relative_path"]))
    return rows, counts_by_class


def validate_splits(
    rows: list[dict[str, str]],
    path_labels: dict[str, str],
    excluded_paths: set[str],
    path_hashes: dict[str, str],
) -> list[str]:
    """Run all required safety and completeness validations."""
    eligible_paths = set(path_labels) - excluded_paths
    manifested_paths = [row["relative_path"] for row in rows]
    path_counts = Counter(manifested_paths)

    if set(manifested_paths) != eligible_paths or any(
        count != 1 for count in path_counts.values()
    ):
        raise SplitError("Each eligible image must appear in the manifest exactly once.")

    if excluded_paths.intersection(manifested_paths):
        raise SplitError("An excluded image was found in the split manifest.")

    for row in rows:
        if path_labels.get(row["relative_path"]) != row["class_label"]:
            raise SplitError(f"Incorrect class label for {row['relative_path']}.")

    split_names = {split for split, _ in SPLIT_PERCENTAGES}
    for class_label in EXPECTED_CLASSES:
        class_splits = {
            row["split"] for row in rows if row["class_label"] == class_label
        }
        if class_splits != split_names:
            raise SplitError(
                f"Class {class_label!r} does not have images in every split."
            )

    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_hash[path_hashes[row["relative_path"]]].add(row["split"])
    hashes_across_splits = [
        file_hash for file_hash, splits in splits_by_hash.items() if len(splits) > 1
    ]
    if hashes_across_splits:
        raise SplitError(
            f"Found {len(hashes_across_splits)} byte-identical group(s) across splits."
        )

    return [
        f"all {len(eligible_paths)} eligible images appear exactly once",
        f"all {len(excluded_paths)} excluded images are absent from splits",
        "every class has train, validation, and test images",
        "no byte-identical files occur across different splits",
    ]


def write_csv_files(
    rows: list[dict[str, str]], exclusions: list[dict[str, str]]
) -> None:
    """Write the validated manifests with portable relative paths."""
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with SPLITS_PATH.open("w", encoding="utf-8", newline="") as split_file:
            writer = csv.DictWriter(
                split_file,
                fieldnames=("relative_path", "class_label", "split"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

        with EXCLUSIONS_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as exclusions_file:
            writer = csv.DictWriter(
                exclusions_file,
                fieldnames=("relative_path", "class_label", "reason"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(exclusions)
    except OSError as error:
        raise SplitError(f"Could not write manifest files: {error}") from error


def print_summary(
    counts_by_class: dict[str, dict[str, int]],
    exclusions: list[dict[str, str]],
    validation_results: list[str],
) -> None:
    """Print actual per-class totals, exclusions, and validation results."""
    excluded_by_class = Counter(row["class_label"] for row in exclusions)
    total_counts = {split: 0 for split, _ in SPLIT_PERCENTAGES}

    print(
        "Dataset splits (seed=42; targets: 70% train, 15% validation, 15% test)"
    )
    print("Class       Train  Validation  Test  Excluded")
    for class_label in EXPECTED_CLASSES:
        counts = counts_by_class[class_label]
        for split in total_counts:
            total_counts[split] += counts[split]
        print(
            f"{class_label:<11} {counts['train']:>5}  "
            f"{counts['validation']:>10}  {counts['test']:>4}  "
            f"{excluded_by_class[class_label]:>8}"
        )

    print(
        f"{'TOTAL':<11} {total_counts['train']:>5}  "
        f"{total_counts['validation']:>10}  {total_counts['test']:>4}  "
        f"{len(exclusions):>8}"
    )
    print("Validation passed:")
    for result in validation_results:
        print(f"  - {result}")
    print(f"Split manifest: {project_relative(SPLITS_PATH)}")
    print(f"Exclusions: {project_relative(EXCLUSIONS_PATH)}")


def main() -> None:
    try:
        audit = load_clean_audit()
        path_labels = scan_dataset(audit)
        exclusions = cross_class_exclusions(audit, path_labels)
        excluded_paths = {row["relative_path"] for row in exclusions}
        path_hashes, hash_paths = hash_eligible_files(path_labels, excluded_paths)
        rows, counts_by_class = build_split_rows(path_labels, hash_paths)
        validation_results = validate_splits(
            rows, path_labels, excluded_paths, path_hashes
        )
        write_csv_files(rows, exclusions)
    except SplitError as error:
        raise SystemExit(f"ERROR: {error}") from error

    print_summary(counts_by_class, exclusions, validation_results)


if __name__ == "__main__":
    main()
