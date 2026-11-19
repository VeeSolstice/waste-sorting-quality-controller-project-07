from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FUNDAMENTALS = ROOT / "src" / "fundamentals"
sys.path.insert(0, str(FUNDAMENTALS))

from demo_gradients import (  # noqa: E402
    FULL_BATCH_LEARNING_RATE,
    FULL_BATCH_STEPS,
    STOCHASTIC_LEARNING_RATE,
    STOCHASTIC_SEED,
    STOCHASTIC_UPDATES,
    finite_difference_gradients,
    make_tiny_example,
    maximum_relative_error,
    train_full_batch_gradient_descent,
    train_stochastic_sgd,
)


class NumPyMLPTests(unittest.TestCase):
    def test_forward_shapes_probabilities_and_finite_loss(self) -> None:
        inputs, targets, model = make_tiny_example()

        probabilities, cache = model.forward(inputs)
        loss = model.loss(inputs, targets)

        self.assertEqual(probabilities.shape, (5, 3))
        self.assertEqual(cache["hidden_pre_activation"].shape, (5, 4))
        self.assertEqual(cache["hidden_activation"].shape, (5, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.isfinite(loss))

    def test_manual_gradients_match_finite_differences(self) -> None:
        inputs, targets, model = make_tiny_example()
        _, _, analytical = model.loss_and_gradients(inputs, targets)
        numerical = finite_difference_gradients(model, inputs, targets)

        self.assertEqual(analytical.keys(), numerical.keys())
        for name in analytical:
            with self.subTest(parameter=name):
                self.assertEqual(analytical[name].shape, model.parameters()[name].shape)
                self.assertLess(
                    maximum_relative_error(analytical[name], numerical[name]),
                    1e-6,
                )

    def test_full_batch_gradient_descent_reduces_loss(self) -> None:
        inputs, targets, model = make_tiny_example()
        initial_loss, final_loss = train_full_batch_gradient_descent(
            model,
            inputs,
            targets,
            steps=FULL_BATCH_STEPS,
            learning_rate=FULL_BATCH_LEARNING_RATE,
        )

        self.assertLess(final_loss, initial_loss)

    def test_stochastic_sgd_uses_one_example_and_reduces_full_loss(self) -> None:
        inputs, targets, model = make_tiny_example()
        observed_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        original_loss_and_gradients = model.loss_and_gradients

        def record_batch_shapes(
            batch_inputs: np.ndarray, batch_targets: np.ndarray
        ) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
            observed_shapes.append((batch_inputs.shape, batch_targets.shape))
            return original_loss_and_gradients(batch_inputs, batch_targets)

        with patch.object(
            model,
            "loss_and_gradients",
            side_effect=record_batch_shapes,
        ):
            initial_loss, final_loss = train_stochastic_sgd(
                model,
                inputs,
                targets,
                updates=STOCHASTIC_UPDATES,
                learning_rate=STOCHASTIC_LEARNING_RATE,
                seed=STOCHASTIC_SEED,
            )

        self.assertEqual(len(observed_shapes), STOCHASTIC_UPDATES)
        self.assertTrue(
            all(shapes == ((1, 3), (1,)) for shapes in observed_shapes)
        )
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
