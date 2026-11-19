"""Compare educational NumPy optimizers on the existing five-example dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .demo_gradients import (
        DEMO_SEED,
        STOCHASTIC_SEED,
        make_tiny_example,
        train_full_batch_gradient_descent,
        train_stochastic_sgd,
    )
    from .numpy_mlp import TwoLayerMLP
except ImportError:  # Allows: python src/fundamentals/compare_optimizers.py
    from demo_gradients import (
        DEMO_SEED,
        STOCHASTIC_SEED,
        make_tiny_example,
        train_full_batch_gradient_descent,
        train_stochastic_sgd,
    )
    from numpy_mlp import TwoLayerMLP


UPDATE_BUDGET = 80
GD_LEARNING_RATE = 0.2
STOCHASTIC_SGD_LEARNING_RATE = 0.2
MOMENTUM_LEARNING_RATE = 0.05
NESTEROV_LEARNING_RATE = 0.05
MOMENTUM_VALUE = 0.9


@dataclass(frozen=True)
class OptimizerResult:
    """Full-dataset loss measurements and declared optimizer settings."""

    method: str
    learning_rate: float
    momentum: float
    updates: int
    initial_loss: float
    final_loss: float


def _train_momentum(
    model: TwoLayerMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    """Run full-batch ordinary Momentum and report full-dataset losses."""
    initial_loss = model.loss(inputs, targets)
    velocities = model.zero_velocity_state()
    for _ in range(UPDATE_BUDGET):
        _, _, gradients = model.loss_and_gradients(inputs, targets)
        model.momentum_step(
            gradients,
            velocities,
            MOMENTUM_LEARNING_RATE,
            MOMENTUM_VALUE,
        )
    return initial_loss, model.loss(inputs, targets)


def _train_nesterov(
    model: TwoLayerMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    """Run full-batch Nesterov updates and report full-dataset losses."""
    initial_loss = model.loss(inputs, targets)
    velocities = model.zero_velocity_state()
    for _ in range(UPDATE_BUDGET):
        model.nesterov_step(
            inputs,
            targets,
            velocities,
            NESTEROV_LEARNING_RATE,
            MOMENTUM_VALUE,
        )
    return initial_loss, model.loss(inputs, targets)


def run_optimizer_comparison() -> list[OptimizerResult]:
    """Run four methods from identical seed-7 parameters for 80 updates each."""
    inputs, targets, reference_model = make_tiny_example()
    models: dict[str, TwoLayerMLP] = {}
    for method in ("Full-batch GD", "Stochastic SGD", "Momentum", "Nesterov"):
        _, _, models[method] = make_tiny_example()

    for model in models.values():
        for name, reference_parameter in reference_model.parameters().items():
            if not np.array_equal(model.parameters()[name], reference_parameter):
                raise RuntimeError("optimizer runs did not start from identical weights")

    gd_initial, gd_final = train_full_batch_gradient_descent(
        models["Full-batch GD"],
        inputs,
        targets,
        steps=UPDATE_BUDGET,
        learning_rate=GD_LEARNING_RATE,
    )
    stochastic_initial, stochastic_final = train_stochastic_sgd(
        models["Stochastic SGD"],
        inputs,
        targets,
        updates=UPDATE_BUDGET,
        learning_rate=STOCHASTIC_SGD_LEARNING_RATE,
        seed=STOCHASTIC_SEED,
    )
    momentum_initial, momentum_final = _train_momentum(
        models["Momentum"], inputs, targets
    )
    nesterov_initial, nesterov_final = _train_nesterov(
        models["Nesterov"], inputs, targets
    )

    return [
        OptimizerResult(
            "Full-batch GD",
            GD_LEARNING_RATE,
            0.0,
            UPDATE_BUDGET,
            gd_initial,
            gd_final,
        ),
        OptimizerResult(
            "Stochastic SGD",
            STOCHASTIC_SGD_LEARNING_RATE,
            0.0,
            UPDATE_BUDGET,
            stochastic_initial,
            stochastic_final,
        ),
        OptimizerResult(
            "Momentum",
            MOMENTUM_LEARNING_RATE,
            MOMENTUM_VALUE,
            UPDATE_BUDGET,
            momentum_initial,
            momentum_final,
        ),
        OptimizerResult(
            "Nesterov",
            NESTEROV_LEARNING_RATE,
            MOMENTUM_VALUE,
            UPDATE_BUDGET,
            nesterov_initial,
            nesterov_final,
        ),
    ]


def main() -> None:
    results = run_optimizer_comparison()
    print("NumPy optimizer comparison on the same five-example dataset")
    print(f"Initialization seed: {DEMO_SEED}; update budget per method: {UPDATE_BUDGET}")
    print("Losses are evaluated on all five examples before and after training.")
    print("Method          learning rate  momentum  initial loss  final loss")
    for result in results:
        print(
            f"{result.method:<15} {result.learning_rate:>13.3f}  "
            f"{result.momentum:>8.2f}  {result.initial_loss:>12.6f}  "
            f"{result.final_loss:>10.6f}"
        )
    print(
        "This tiny deterministic example illustrates update behavior only; it "
        "does not show that any optimizer is always better."
    )


if __name__ == "__main__":
    main()
