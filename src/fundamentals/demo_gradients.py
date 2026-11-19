"""Check manual MLP gradients and train a deterministic tiny example."""

from __future__ import annotations

import numpy as np

try:
    from .numpy_mlp import TwoLayerMLP
except ImportError:  # Allows: python src/fundamentals/demo_gradients.py
    from numpy_mlp import TwoLayerMLP


DEMO_SEED = 7
FINITE_DIFFERENCE_EPSILON = 1e-5
FULL_BATCH_LEARNING_RATE = 0.2
FULL_BATCH_STEPS = 80
STOCHASTIC_LEARNING_RATE = 0.2
STOCHASTIC_UPDATES = 80
STOCHASTIC_SEED = 23


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


def train_full_batch_gradient_descent(
    model: TwoLayerMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int = FULL_BATCH_STEPS,
    learning_rate: float = FULL_BATCH_LEARNING_RATE,
) -> tuple[float, float]:
    """Train from gradients computed over all examples on every update.

    ``inputs`` has shape ``(5, 3)`` and ``targets`` has shape ``(5,)`` in the
    demonstration. Both returned losses are evaluated on all five examples.
    """
    initial_full_dataset_loss = model.loss(inputs, targets)
    for _ in range(steps):
        _, _, gradients = model.loss_and_gradients(inputs, targets)
        model.sgd_step(gradients, learning_rate)
    final_full_dataset_loss = model.loss(inputs, targets)
    return initial_full_dataset_loss, final_full_dataset_loss


def train_stochastic_sgd(
    model: TwoLayerMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    updates: int = STOCHASTIC_UPDATES,
    learning_rate: float = STOCHASTIC_LEARNING_RATE,
    seed: int = STOCHASTIC_SEED,
) -> tuple[float, float]:
    """Train using one randomly sampled example per parameter update.

    The reported initial and final losses use the full ``(5, 3)`` input matrix.
    Each update gradient, however, uses ``sample_inputs`` with shape ``(1, 3)``
    and ``sample_targets`` with shape ``(1,)``.
    """
    initial_full_dataset_loss = model.loss(inputs, targets)
    generator = np.random.default_rng(seed)
    for _ in range(updates):
        sample_index = int(generator.integers(inputs.shape[0]))
        sample_inputs = inputs[sample_index : sample_index + 1]  # (1, 3)
        sample_targets = targets[sample_index : sample_index + 1]  # (1,)
        _, _, gradients = model.loss_and_gradients(
            sample_inputs, sample_targets
        )
        model.sgd_step(gradients, learning_rate)
    final_full_dataset_loss = model.loss(inputs, targets)
    return initial_full_dataset_loss, final_full_dataset_loss


def main() -> None:
    inputs, targets, gradient_check_model = make_tiny_example()
    _, _, analytical = gradient_check_model.loss_and_gradients(inputs, targets)
    numerical = finite_difference_gradients(
        gradient_check_model, inputs, targets
    )

    errors = {
        name: maximum_relative_error(analytical[name], numerical[name])
        for name in analytical
    }
    _, cache = gradient_check_model.forward(inputs)
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

    _, _, full_batch_model = make_tiny_example()
    _, _, stochastic_model = make_tiny_example()
    identical_initial_weights = all(
        np.array_equal(
            full_batch_model.parameters()[name],
            stochastic_model.parameters()[name],
        )
        for name in full_batch_model.parameters()
    )
    if not identical_initial_weights:
        raise RuntimeError("training comparisons did not start from identical weights")

    full_initial, full_final = train_full_batch_gradient_descent(
        full_batch_model, inputs, targets
    )
    stochastic_initial, stochastic_final = train_stochastic_sgd(
        stochastic_model, inputs, targets
    )

    print(f"Training runs share identical initial weights: {identical_initial_weights}")
    print("Full-batch gradient descent:")
    print(f"  Initial loss on all five examples: {full_initial:.6f}")
    print(
        f"  Final loss on all five examples after {FULL_BATCH_STEPS} updates "
        f"(learning rate {FULL_BATCH_LEARNING_RATE}): {full_final:.6f}"
    )
    print(f"Stochastic SGD (one sampled example per update, seed {STOCHASTIC_SEED}):")
    print(f"  Initial loss on all five examples: {stochastic_initial:.6f}")
    print(
        f"  Final loss on all five examples after {STOCHASTIC_UPDATES} updates "
        f"(learning rate {STOCHASTIC_LEARNING_RATE}): {stochastic_final:.6f}"
    )


if __name__ == "__main__":
    main()
