import numpy as np


class ReLU:
    """Rectified linear unit — used by dense layers during forward/backward passes."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._mask = z > 0
        return z * self._mask

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return self._mask.astype(float)


class Tanh:
    """Hyperbolic tangent — used by dense layers during forward/backward passes."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._out = np.tanh(z)
        return self._out

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - self._out ** 2


class Sigmoid:
    """Sigmoid — used by dense layers during forward/backward passes."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-z))
        return self._out

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return self._out * (1.0 - self._out)


class Linear:
    """Identity activation — used by dense layers during forward/backward passes."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


ACTIVATION_MAP: dict = {
    "relu": ReLU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "linear": Linear,
}


def get_activation(name: str):
    """Returns an activation instance by name; raises ValueError for unknown names."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATION_MAP)}")
    return ACTIVATION_MAP[name]()
