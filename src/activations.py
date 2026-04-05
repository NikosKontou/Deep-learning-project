import numpy as np


class ReLU:
    """Returns the input if positive, otherwise zero."""

    def forward(self, z: np.ndarray):
        # save which values were positive — reused in derivative() to avoid recalculating
        self._mask = z > 0
        return z * self._mask

    def derivative(self, z: np.ndarray):
        # _mask was set during forward(). 1.0 where neuron was active, 0.0 where it was not
        return self._mask.astype(float)


class Tanh:
    """Maps any value to the range (-1, 1)."""

    def forward(self, z: np.ndarray):
        # save output so derivative() can reuse it without recalculating tanh
        self._out = np.tanh(z)
        return self._out

    def derivative(self, z: np.ndarray):
        # _out was set during forward(). tanh derivative = 1 - tanh(z)^2
        return 1.0 - self._out ** 2


class Sigmoid:
    """Maps any value to the range (0, 1). Used as the output activation for classification."""

    def forward(self, z: np.ndarray):
        # save output so derivative() can reuse it without recalculating sigmoid
        self._out = 1.0 / (1.0 + np.exp(-z))
        return self._out

    def derivative(self, z: np.ndarray):
        # _out was set during forward(). sigmoid derivative = sigmoid(z) * (1 - sigmoid(z))
        return self._out * (1.0 - self._out)


class Linear:
    """Identity function — returns the input unchanged. Used as output activation for regression."""

    def forward(self, z: np.ndarray):
        return z

    def derivative(self, z: np.ndarray):
        # gradient of a straight line is always 1 — every element passes through unchanged
        return np.ones_like(z)


# maps the string name from the JSON config to the correct class
ACTIVATION_MAP: dict = {
    "relu": ReLU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "linear": Linear,
}

def get_activation(name: str):
    """Called by DenseLayer.__init__ to attach the correct activation to a layer."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATION_MAP)}")
    return ACTIVATION_MAP[name]()
