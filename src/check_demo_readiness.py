"""Read-only checks for the default and optional offline demonstrations.

This command never downloads data, trains a model, or evaluates a dataset.  It
only verifies the local environment, saved checkpoint metadata, hashes, and a
chosen image for each Streamlit workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "waste_resnet18.pt"
DEFAULT_DEMO_IMAGE = PROJECT_ROOT / "data" / "demo" / "composite_validation_scene.png"
SELECTION_RECORD = (
    PROJECT_ROOT
    / "data"
    / "taco_detection"
    / "final_detector_evaluation"
    / "selection_record.json"
)
EXPERIMENTAL_DEMO_IMAGE = (
    PROJECT_ROOT / "data" / "taco_detection" / "images" / "batch_12" / "000018.jpg"
)

EXPECTED_CLASS_TO_INDEX = {
    "cardboard": 1,
    "glass": 2,
    "metal": 3,
    "paper": 4,
    "plastic": 5,
    "other": 6,
}
EXPECTED_CLASSIFIER_CLASSES = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
}

DEPENDENCIES = (
    ("NumPy", "numpy", "numpy"),
    ("Pillow", "PIL", "Pillow"),
    ("OpenCV", "cv2", "opencv-python"),
    ("Streamlit", "streamlit", "streamlit"),
    ("PyTorch", "torch", "torch"),
    ("Torchvision", "torchvision", "torchvision"),
    ("pycocotools", "pycocotools", "pycocotools"),
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dependencies() -> list[Check]:
    """Import the runtime dependencies and report their installed versions."""
    checks: list[Check] = []
    for display_name, module_name, distribution_name in DEPENDENCIES:
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution_name)
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            checks.append(Check(display_name, False, str(error)))
        else:
            checks.append(Check(display_name, True, version))
    return checks


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a dictionary")
    return checkpoint


def check_default_checkpoint(path: Path) -> Check:
    """Verify that the committed ResNet checkpoint is readable and compatible."""
    if not path.is_file():
        return Check("ResNet-18 checkpoint", False, f"missing: {path}")
    try:
        checkpoint = _load_torch_checkpoint(path)
        architecture = checkpoint.get("architecture")
        class_mapping = checkpoint.get("class_to_index")
        state_dict = checkpoint.get("model_state_dict")
        if architecture != "resnet18":
            raise ValueError(f"unexpected architecture: {architecture!r}")
        if class_mapping != EXPECTED_CLASSIFIER_CLASSES:
            raise ValueError("six-class mapping does not match the application")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("model_state_dict is missing or empty")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return Check("ResNet-18 checkpoint", False, str(error))
    digest = sha256_file(path)
    return Check(
        "ResNet-18 checkpoint",
        True,
        f"epoch {checkpoint.get('epoch')}; SHA-256 {digest}",
    )


def check_image(path: Path, label: str) -> Check:
    """Fully decode and EXIF-orient one local demonstration image."""
    if not path.is_file():
        return Check(label, False, f"missing: {path}")
    try:
        image_module = importlib.import_module("PIL.Image")
        image_ops = importlib.import_module("PIL.ImageOps")
        with image_module.open(path) as source:
            source.load()
            image = image_ops.exif_transpose(source).convert("RGB")
            size = image.size
    except (ImportError, OSError, ValueError) as error:
        return Check(label, False, f"decode failed: {error}")
    return Check(label, True, f"RGB {size[0]}x{size[1]}: {path}")


def check_selected_detector(selection_record_path: Path) -> list[Check]:
    """Verify the selection record, its selected hash, and checkpoint metadata."""
    if not selection_record_path.is_file():
        return [
            Check(
                "TACO selection record",
                False,
                f"missing ignored local file: {selection_record_path}",
            )
        ]
    try:
        record_bytes = selection_record_path.read_bytes()
        record = json.loads(record_bytes)
        checkpoint_path = Path(record["selected_checkpoint_path"])
        expected_checkpoint_hash = str(record["selected_checkpoint_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return [Check("TACO selection record", False, f"invalid: {error}")]

    record_hash = hashlib.sha256(record_bytes).hexdigest()
    checks = [
        Check(
            "TACO selection record",
            True,
            (
                f"SHA-256 {record_hash}; selected validation AP50 "
                f"{record.get('selected_validation_ap50')}"
            ),
        )
    ]
    if not checkpoint_path.is_file():
        checks.append(
            Check(
                "Selected TACO checkpoint",
                False,
                f"missing ignored local file: {checkpoint_path}",
            )
        )
        return checks

    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        checks.append(
            Check(
                "Selected TACO checkpoint",
                False,
                (
                    "SHA-256 mismatch: record expects "
                    f"{expected_checkpoint_hash}, found {actual_checkpoint_hash}"
                ),
            )
        )
        return checks

    try:
        checkpoint = _load_torch_checkpoint(checkpoint_path)
        metadata = checkpoint["metadata"]
        if metadata.get("architecture") != "fasterrcnn_mobilenet_v3_large_320_fpn":
            raise ValueError("unexpected detector architecture")
        if metadata.get("class_to_index") != EXPECTED_CLASS_TO_INDEX:
            raise ValueError("detector class mapping does not match the application")
        resize = metadata["resize"]
        cpu_max = int(resize["cpu"]["maximum_long_side"])
        model_min = int(resize["model_transform"]["minimum_side"])
        model_max = int(resize["model_transform"]["maximum_side"])
        if cpu_max < 1 or model_min < 1 or model_max < model_min:
            raise ValueError("invalid saved resize settings")
        if not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("model_state_dict is missing")
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        checks.append(Check("Selected TACO checkpoint", False, str(error)))
        return checks

    checks.append(
        Check(
            "Selected TACO checkpoint",
            True,
            (
                f"SHA-256 {actual_checkpoint_hash}; CPU long side {cpu_max}; "
                f"model min/max {model_min}/{model_max}"
            ),
        )
    )
    return checks


def print_checks(title: str, checks: list[Check]) -> bool:
    """Print a compact section and return whether every check passed."""
    ready = all(check.passed for check in checks)
    print(f"\n{title}: {'READY' if ready else 'NOT READY'}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")
    return ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check offline demo prerequisites without changing project files."
    )
    parser.add_argument("--default-image", type=Path, default=DEFAULT_DEMO_IMAGE)
    parser.add_argument(
        "--experimental-image", type=Path, default=EXPERIMENTAL_DEMO_IMAGE
    )
    parser.add_argument("--selection-record", type=Path, default=SELECTION_RECORD)
    parser.add_argument(
        "--require-experimental",
        action="store_true",
        help="Return a failure code when the optional TACO mode is not ready.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Offline demo readiness (read-only: no downloads, training, or evaluation)")
    dependency_ready = print_checks("Runtime dependencies", check_dependencies())
    default_ready = print_checks(
        "Default controlled-background workflow",
        [
            Check(
                "Python",
                sys.version_info >= (3, 13),
                sys.version.split()[0] + " (requires 3.13 or newer)",
            ),
            check_default_checkpoint(DEFAULT_CHECKPOINT),
            check_image(args.default_image, "Default demo image"),
        ],
    )
    experimental_ready = print_checks(
        "Optional experimental TACO workflow",
        [
            *check_selected_detector(args.selection_record),
            check_image(args.experimental_image, "Experimental demo image"),
        ],
    )
    if not dependency_ready or not default_ready:
        return 1
    if args.require_experimental and not experimental_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
