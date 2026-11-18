"""ResNet-18 construction for training and checkpoint inference."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from model_config import CLASS_NAMES


ARCHITECTURE_NAME = "resnet18"
# DEFAULT currently identifies torchvision's documented ImageNet-1K V1 weights.
PRETRAINED_WEIGHTS = ResNet18_Weights.DEFAULT
PRETRAINED_WEIGHTS_NAME = f"ResNet18_Weights.{PRETRAINED_WEIGHTS.name}"


class PretrainedWeightsError(Exception):
    """Raised when the required pretrained weights cannot be obtained."""


def _replace_classifier(model: nn.Module) -> nn.Module:
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(CLASS_NAMES))
    return model


def create_training_model() -> nn.Module:
    """Load pretrained ResNet-18 and train only layer4 plus the new classifier."""
    try:
        model = resnet18(weights=PRETRAINED_WEIGHTS)
    except Exception as error:
        raise PretrainedWeightsError(
            f"Could not load {PRETRAINED_WEIGHTS_NAME}. Check the internet "
            "connection or torchvision cache. Training stopped; the program will "
            "not silently use an untrained model. Original error: {error}"
        ) from error

    for parameter in model.parameters():
        parameter.requires_grad = False

    model = _replace_classifier(model)
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def create_inference_model() -> nn.Module:
    """Build the architecture without downloading weights for checkpoint loading."""
    model = resnet18(weights=None)
    return _replace_classifier(model)


def keep_frozen_layers_in_eval_mode(model: nn.Module) -> None:
    """Prevent frozen batch-normalization statistics from changing during training."""
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()
