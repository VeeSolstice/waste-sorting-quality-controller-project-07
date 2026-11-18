"""Shared paths and model settings for the waste classifier."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "dataset_splits.csv"
CLASS_MAPPING_PATH = PROJECT_ROOT / "artifacts" / "class_to_index.json"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "waste_resnet18.pt"
SMOKE_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "waste_resnet18_smoke.pt"

CLASS_NAMES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RANDOM_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# This is a prediction-based operational rule. A softmax score is not a
# calibrated guarantee that a classification is correct.
QUALITY_THRESHOLD = 0.60
SOFTMAX_NOTICE = (
    "Prediction-based rule only: softmax scores are not calibrated guarantees."
)

PREPROCESSING_INFO = {
    "input_color_mode": "RGB",
    "image_size": IMAGE_SIZE,
    "training": (
        "RandomResizedCrop(224), RandomHorizontalFlip, ToTensor, "
        "ImageNet normalization"
    ),
    "validation_and_inference": (
        "Resize shorter side to 256, CenterCrop(224), ToTensor, "
        "ImageNet normalization"
    ),
    "imagenet_mean": list(IMAGENET_MEAN),
    "imagenet_std": list(IMAGENET_STD),
}
