"""A small two-layer neural network with gradients written by hand.

This educational module uses only NumPy. It is intentionally separate from the
project's trained PyTorch image classifiers and does not use autograd.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def stable_softmax(logits: Array) -> Array:
    """Convert class logits ``(batch_size, class_count)`` into probabilities.

    Subtracting each row's maximum before exponentiation prevents overflow while
    preserving the softmax result.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, class_count)")

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)  # (batch_size, class_count)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def mean_cross_entropy(probabilities: Array, targets: Array) -> float:
    """Return mean cross-entropy for probabilities ``(N, C)`` and labels ``(N,)``."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch_size, class_count)")
    if targets.shape != (probabilities.shape[0],):
        raise ValueError("targets must have shape (batch_size,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("targets must contain integer class indices")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("target class index is out of range")

    # selected_probabilities has shape (batch_size,).
    selected_probabilities = probabilities[np.arange(targets.size), targets]
    safe_probabilities = np.clip(selected_probabilities, 1e-15, 1.0)
    return float(-np.mean(np.log(safe_probabilities)))


class TwoLayerMLP:
    """A fully connected ReLU network: input -> hidden -> class scores.

    Parameter shapes:
        w1: ``(input_size, hidden_size)``
        b1: ``(hidden_size,)``
        w2: ``(hidden_size, output_size)``
        b2: ``(output_size,)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        seed: int = 42,
        weight_scale: float = 0.1,
    ) -> None:
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("all layer sizes must be positive")
        if weight_scale <= 0:
            raise ValueError("weight_scale must be positive")

        generator = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w1 = generator.normal(
            0.0, weight_scale, size=(input_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = generator.normal(
            0.0, weight_scale, size=(hidden_size, output_size)
        )
        self.b2 = np.zeros(output_size, dtype=np.float64)

    def parameters(self) -> dict[str, Array]:
        """Return mutable parameter arrays with the shapes documented above."""
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def _validate_inputs(self, inputs: Array) -> Array:
        """Validate and return examples with shape ``(batch_size, input_size)``."""
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(
                f"inputs must have shape (batch_size, {self.input_size})"
            )
        return inputs

    def forward(self, inputs: Array) -> tuple[Array, dict[str, Array]]:
        """Run the network and return probabilities plus a backward-pass cache.

        Tensor shapes:
            inputs: ``(N, D)`` where N is batch size and D is input size
            hidden_pre_activation: ``(N, H)``
            hidden_activation: ``(N, H)``
            logits and probabilities: ``(N, C)``
        """
        inputs = self._validate_inputs(inputs)
        hidden_pre_activation = inputs @ self.w1 + self.b1  # (N, H)
        hidden_activation = np.maximum(0.0, hidden_pre_activation)  # (N, H)
        logits = hidden_activation @ self.w2 + self.b2  # (N, C)
        probabilities = stable_softmax(logits)  # (N, C)
        cache = {
            "inputs": inputs,
            "hidden_pre_activation": hidden_pre_activation,
            "hidden_activation": hidden_activation,
            "probabilities": probabilities,
        }
        return probabilities, cache

    def loss(self, inputs: Array, targets: Array) -> float:
        """Return mean loss for inputs ``(N, D)`` and integer targets ``(N,)``."""
        probabilities, _ = self.forward(inputs)
        return mean_cross_entropy(probabilities, targets)

    def loss_and_gradients(
        self, inputs: Array, targets: Array
    ) -> tuple[float, Array, dict[str, Array]]:
        """Return loss, probabilities, and manual parameter gradients.

        Gradient shapes match their parameters:
            gradients["w1"]: ``(D, H)``
            gradients["b1"]: ``(H,)``
            gradients["w2"]: ``(H, C)``
            gradients["b2"]: ``(C,)``
        """
        probabilities, cache = self.forward(inputs)
        targets = np.asarray(targets)
        loss = mean_cross_entropy(probabilities, targets)
        batch_size = probabilities.shape[0]

        # d_logits is (N, C): derivative of mean softmax cross-entropy.
        d_logits = probabilities.copy()
        d_logits[np.arange(batch_size), targets] -= 1.0
        d_logits /= batch_size

        hidden_activation = cache["hidden_activation"]  # (N, H)
        d_w2 = hidden_activation.T @ d_logits  # (H, C)
        d_b2 = np.sum(d_logits, axis=0)  # (C,)
        d_hidden = d_logits @ self.w2.T  # (N, H)

        hidden_pre_activation = cache["hidden_pre_activation"]  # (N, H)
        # ReLU's derivative is defined as zero at exactly zero in this example.
        d_hidden_pre_activation = d_hidden * (hidden_pre_activation > 0.0)  # (N, H)
        input_batch = cache["inputs"]  # (N, D)
        d_w1 = input_batch.T @ d_hidden_pre_activation  # (D, H)
        d_b1 = np.sum(d_hidden_pre_activation, axis=0)  # (H,)

        gradients = {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}
        return loss, probabilities, gradients

    def sgd_step(self, gradients: dict[str, Array], learning_rate: float) -> None:
        """Apply one SGD update to each parameter using same-shaped gradients."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        parameters = self.parameters()
        if gradients.keys() != parameters.keys():
            raise ValueError("gradients must contain w1, b1, w2, and b2")
        for name, parameter in parameters.items():
            gradient = np.asarray(gradients[name], dtype=np.float64)
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"gradient for {name} has shape {gradient.shape}; "
                    f"expected {parameter.shape}"
                )
            parameter -= learning_rate * gradient

    def zero_velocity_state(self) -> dict[str, Array]:
        """Return zero velocity arrays matching ``w1``, ``b1``, ``w2``, and ``b2``."""
        return {
            name: np.zeros_like(parameter)
            for name, parameter in self.parameters().items()
        }

    def _validate_momentum_inputs(
        self,
        gradients: dict[str, Array],
        velocities: dict[str, Array],
        learning_rate: float,
        momentum: float,
    ) -> None:
        """Validate same-shaped gradient and velocity tensors for an update."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the interval [0, 1)")

        parameters = self.parameters()
        if gradients.keys() != parameters.keys():
            raise ValueError("gradients must contain w1, b1, w2, and b2")
        if velocities.keys() != parameters.keys():
            raise ValueError("velocities must contain w1, b1, w2, and b2")

        for name, parameter in parameters.items():
            gradient = np.asarray(gradients[name])
            velocity = np.asarray(velocities[name])
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"gradient for {name} has shape {gradient.shape}; "
                    f"expected {parameter.shape}"
                )
            if velocity.shape != parameter.shape:
                raise ValueError(
                    f"velocity for {name} has shape {velocity.shape}; "
                    f"expected {parameter.shape}"
                )
            if not np.issubdtype(velocity.dtype, np.floating):
                raise ValueError(f"velocity for {name} must use a floating dtype")

    def momentum_step(
        self,
        gradients: dict[str, Array],
        velocities: dict[str, Array],
        learning_rate: float,
        momentum: float,
    ) -> None:
        """Apply ordinary Momentum using gradients at the current parameters.

        For every parameter tensor ``theta`` and same-shaped velocity ``v``:
        ``v = momentum * v - learning_rate * gradient`` and ``theta += v``.
        The supplied velocity arrays are updated in place.
        """
        self._validate_momentum_inputs(
            gradients, velocities, learning_rate, momentum
        )
        for name, parameter in self.parameters().items():
            gradient = np.asarray(gradients[name], dtype=np.float64)
            velocities[name][...] = (
                momentum * velocities[name] - learning_rate * gradient
            )
            parameter += velocities[name]

    def nesterov_step(
        self,
        inputs: Array,
        targets: Array,
        velocities: dict[str, Array],
        learning_rate: float,
        momentum: float,
    ) -> float:
        """Apply Nesterov momentum using a gradient at look-ahead parameters.

        The look-ahead point is ``theta + momentum * v``. The method evaluates
        ``loss_and_gradients`` there, restores the current parameters, then uses
        ``v = momentum * v - learning_rate * lookahead_gradient`` and
        ``theta += v``. It returns the finite loss observed at the look-ahead.
        """
        parameters = self.parameters()
        placeholder_gradients = {
            name: np.zeros_like(parameter)
            for name, parameter in parameters.items()
        }
        self._validate_momentum_inputs(
            placeholder_gradients, velocities, learning_rate, momentum
        )

        original_parameters = {
            name: parameter.copy() for name, parameter in parameters.items()
        }
        for name, parameter in parameters.items():
            parameter[...] = original_parameters[name] + momentum * velocities[name]

        try:
            lookahead_loss, _, lookahead_gradients = self.loss_and_gradients(
                inputs, targets
            )
        finally:
            for name, parameter in parameters.items():
                parameter[...] = original_parameters[name]

        self._validate_momentum_inputs(
            lookahead_gradients, velocities, learning_rate, momentum
        )
        for name, parameter in parameters.items():
            gradient = np.asarray(lookahead_gradients[name], dtype=np.float64)
            velocities[name][...] = (
                momentum * velocities[name] - learning_rate * gradient
            )
            parameter += velocities[name]
        return float(lookahead_loss)
