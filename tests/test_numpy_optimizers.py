from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FUNDAMENTALS = ROOT / "src" / "fundamentals"
sys.path.insert(0, str(FUNDAMENTALS))

from compare_optimizers import run_optimizer_comparison  # noqa: E402
from demo_gradients import make_tiny_example  # noqa: E402


class NumPyOptimizerTests(unittest.TestCase):
    def test_momentum_update_equation(self) -> None:
        _, _, model = make_tiny_example()
        original = {
            name: parameter.copy() for name, parameter in model.parameters().items()
        }
        velocities = {
            name: np.full_like(parameter, 0.04)
            for name, parameter in model.parameters().items()
        }
        original_velocities = {
            name: velocity.copy() for name, velocity in velocities.items()
        }
        gradients = {
            name: np.full_like(parameter, 0.3)
            for name, parameter in model.parameters().items()
        }
        learning_rate = 0.2
        momentum = 0.8

        model.momentum_step(gradients, velocities, learning_rate, momentum)

        for name, parameter in model.parameters().items():
            expected_velocity = (
                momentum * original_velocities[name]
                - learning_rate * gradients[name]
            )
            np.testing.assert_allclose(velocities[name], expected_velocity)
            np.testing.assert_allclose(parameter, original[name] + expected_velocity)

    def test_nesterov_uses_lookahead_gradient_and_update_equation(self) -> None:
        inputs, targets, model = make_tiny_example()
        original = {
            name: parameter.copy() for name, parameter in model.parameters().items()
        }
        velocities = {
            name: np.full_like(parameter, 0.05)
            for name, parameter in model.parameters().items()
        }
        original_velocities = {
            name: velocity.copy() for name, velocity in velocities.items()
        }
        controlled_gradients = {
            name: np.full_like(parameter, 0.25)
            for name, parameter in model.parameters().items()
        }
        observed_parameters: dict[str, np.ndarray] = {}
        learning_rate = 0.1
        momentum = 0.9

        def record_lookahead(
            batch_inputs: np.ndarray, batch_targets: np.ndarray
        ) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
            self.assertEqual(batch_inputs.shape, (5, 3))
            self.assertEqual(batch_targets.shape, (5,))
            observed_parameters.update(
                {
                    name: parameter.copy()
                    for name, parameter in model.parameters().items()
                }
            )
            probabilities = np.full((5, 3), 1.0 / 3.0)
            return 1.25, probabilities, controlled_gradients

        with patch.object(
            model,
            "loss_and_gradients",
            side_effect=record_lookahead,
        ) as gradient_call:
            lookahead_loss = model.nesterov_step(
                inputs,
                targets,
                velocities,
                learning_rate,
                momentum,
            )

        gradient_call.assert_called_once()
        self.assertEqual(lookahead_loss, 1.25)
        for name, parameter in model.parameters().items():
            expected_lookahead = original[name] + momentum * original_velocities[name]
            expected_velocity = (
                momentum * original_velocities[name]
                - learning_rate * controlled_gradients[name]
            )
            np.testing.assert_allclose(observed_parameters[name], expected_lookahead)
            np.testing.assert_allclose(velocities[name], expected_velocity)
            np.testing.assert_allclose(parameter, original[name] + expected_velocity)

    def test_comparison_losses_are_finite(self) -> None:
        results = run_optimizer_comparison()

        self.assertEqual(len(results), 4)
        initial_losses = np.array([result.initial_loss for result in results])
        final_losses = np.array([result.final_loss for result in results])
        self.assertTrue(np.all(np.isfinite(initial_losses)))
        self.assertTrue(np.all(np.isfinite(final_losses)))
        np.testing.assert_allclose(initial_losses, initial_losses[0], atol=0.0)


if __name__ == "__main__":
    unittest.main()
