import numpy as np
from src.activations import get_activation


class DenseLayer:
    """
    Single fully-connected layer (no bias).
    Holds weights and delegates activation to an activation object.
    Interacts with Network: forward() is chained across layers;
    backward() receives upstream gradient and returns downstream gradient.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str) -> None:
        self.W = self._init_weights(in_dim, out_dim)
        self.activation = get_activation(activation)
        self._x: np.ndarray | None = None   # cached input for backward
        self._z: np.ndarray | None = None   # cached pre-activation for backward
        self.dW: np.ndarray | None = None   # gradient accumulated by backward

    @staticmethod
    def _init_weights(in_dim: int, out_dim: int) -> np.ndarray:
        # he initialisation — works well with relu and tanh
        scale = np.sqrt(2.0 / in_dim)
        return np.random.randn(in_dim, out_dim) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes a = activation(xW); caches x and z for backward."""
        self._x = x
        self._z = x @ self.W
        return self.activation.forward(self._z)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Receives dL/da from the next layer, stores dL/dW,
        and returns dL/dx to propagate to the previous layer.
        """
        da_dz = self.activation.derivative(self._z)
        delta = grad_out * da_dz      # dL/dz  (element-wise)
        self.dW = self._x.T @ delta   # dL/dW
        return delta @ self.W.T       # dL/dx
