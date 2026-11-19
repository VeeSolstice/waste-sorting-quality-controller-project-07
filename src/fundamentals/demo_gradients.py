"""Check manual MLP gradients and train a deterministic tiny example."""

from __future__ import annotations

import numpy as np

try:
    from .numpy_mlp import TwoLayerMLP
except ImportError:  # Allows: python src/fundamentals/demo_gradients.py
    from numpy_mlp import TwoLayerMLP


DEMO_SEED = 7
FINITE_DIFFERENCE_EPSILON = 1e-5
SGD_LEARNING_RATE = 0.2
SGD_STEPS = 80


def make_tiny_example() -> tuple[np.ndarray, np.ndarray, TwoLayerMLP]:
    """Return examples ``(5, 3)``, labels ``(5,)``, and a seeded 3-4-3 MLP."""
    inputs = np.array(
        [
            [1.2, -0.7, 0.3],
            [-0.4, 1.1, 0.8],
            [0.5, 0.9, -1.3],
            [-1.0, -0.5, 1.4],
            [0.8, 0.2, 0.6],
        ],
        dtype=np.float64,
    )  # (batch_size=5, input_size=3)
    targets = np.array([0, 1, 2, 1, 0], dtype=np.int64)  # (batch_size=5,)
    model = TwoLayerMLP(3, 4, 3, seed=DEMO_SEED, weight_scale=0.25)
    return inputs, targets, model


def finite_difference_gradients(
    model: TwoLayerMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    epsilon: float = FINITE_DIFFERENCE_EPSILON,
) -> dict[str, np.ndarray]:
    """Estimate every parameter gradient with centered finite differences.

    Each returned array has the same shape as its corresponding model parameter.
    """
    numerical: dict[str, np.ndarray] = {}
    for name, parameter in model.parameters().items():
        gradient = np.zeros_like(parameter)  # Same shape as w1, b1, w2, or b2.
        for index in np.ndindex(parameter.shape):
            original_value = parameter[index]
            parameter[index] = original_value + epsilon
            loss_plus = model.loss(inputs, targets)
            parameter[index] = original_value - epsilon
            loss_minus = model.loss(inputs, targets)
            parameter[index] = original_value
            gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)
        numerical[name] = gradient
    return numerical


def maximum_relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Return the largest element-wise relative error for same-shaped arrays."""
    denominator = np.maximum(1e-12, np.abs(analytical) + np.abs(numerical))
    return float(np.max(np.abs(analytical - numerical) / denominator))


def main() -> None:
    inputs, targets, model = make_tiny_example()
    initial_loss, _, analytical = model.loss_and_gradients(inputs, targets)
    numerical = finite_difference_gradients(model, inputs, targets)

    errors = {
        name: maximum_relative_error(analytical[name], numerical[name])
        for name in analytical
    }
    _, cache = model.forward(inputs)
    minimum_relu_distance = float(
        np.min(np.abs(cache["hidden_pre_activation"]))
    )

    print(f"Fixed demonstration seed: {DEMO_SEED}")
    print("Gradient check (maximum relative error per parameter):")
    for name in ("w1", "b1", "w2", "b2"):
        print(f"  {name}: {errors[name]:.3e}")
    print(f"Largest relative error: {max(errors.values()):.3e}")
    print(
        "ReLU note: ReLU is not differentiable at zero. The analytical rule here "
        "uses derivative 0 at zero; a finite difference that crosses that boundary "
        "can disagree."
    )
    print(
        "Smallest absolute hidden pre-activation in this check: "
        f"{minimum_relu_distance:.6f} (safely away from zero)."
    )

    for _ in range(SGD_STEPS):
        _, _, gradients = model.loss_and_gradients(inputs, targets)
        model.sgd_step(gradients, SGD_LEARNING_RATE)
    final_loss = model.loss(inputs, targets)

    print(f"Loss before SGD: {initial_loss:.6f}")
    print(
        f"Loss after {SGD_STEPS} SGD updates "
        f"(learning rate {SGD_LEARNING_RATE}): {final_loss:.6f}"
    )


if __name__ == "__main__":
    main()
