from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FUNDAMENTALS = ROOT / "src" / "fundamentals"
sys.path.insert(0, str(FUNDAMENTALS))

from demo_gradients import (  # noqa: E402
    SGD_LEARNING_RATE,
    SGD_STEPS,
    finite_difference_gradients,
    make_tiny_example,
    maximum_relative_error,
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

    def test_sgd_reduces_loss_on_tiny_example(self) -> None:
        inputs, targets, model = make_tiny_example()
        initial_loss = model.loss(inputs, targets)

        for _ in range(SGD_STEPS):
            _, _, gradients = model.loss_and_gradients(inputs, targets)
            model.sgd_step(gradients, SGD_LEARNING_RATE)

        final_loss = model.loss(inputs, targets)
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
