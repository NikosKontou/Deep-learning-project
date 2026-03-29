import numpy as np
from src.activations import get_activation

class DenseLayer:
    """
    One fully-connected layer without  bias.
    Holds weights and delegates activation to an activation object.
    Interacts with Network: forward() is chained across layers
    backward() receives a gradient and returns another gradient.
    """
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        self.W = self._init_weights(in_dim, out_dim)
        self.activation = get_activation(activation)
        #for back the propagation
        # cached input
        self._x: np.ndarray | None = None
        # cached pre-activation
        self._z: np.ndarray | None = None
        # gradient accumulated by backward propagation
        self.dW: np.ndarray | None = None

    @staticmethod
    def _init_weights(in_dim: int, out_dim: int):
        # he initialisation — works well with relu and tanh
        scale = np.sqrt(2.0 / in_dim)
        return np.random.randn(in_dim, out_dim) * scale

    def forward(self, x: np.ndarray):
        """Computes a = activation(xW); caches x and z for backward."""
        self._x = x
        self._z = x @ self.W
        return self.activation.forward(self._z)

    def backward(self, grad_out: np.ndarray):
        """
        Receives dL/da from the next layer, stores dL/dW,
        and propagates to the previous layer.
        """
        da_dz = self.activation.derivative(self._z)
        # dL/dz
        delta = grad_out * da_dz
        # dL/dW
        self.dW = self._x.T @ delta
        # dL/dx
        return delta @ self.W.T
