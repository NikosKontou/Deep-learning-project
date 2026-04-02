import numpy as np
from src.activations import get_activation

class DenseLayer:
    """
    One fully-connected layer without bias.
    forward() computes the output of the layer.
    backward() computes the gradients and passes them to the previous layer.
    """
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        # W shape is (in_dim, out_dim) — each column is one output neuron
        self.W = self._init_weights(in_dim, out_dim)

        # activation object is set here. get_activation() is in activations.py
        self.activation = get_activation(activation)

        # these three are None at start and are filled during forward() and backward()
        # input to this layer
        self._x: np.ndarray | None = None
        # pre-activation output (xW), before activation
        self._z: np.ndarray | None = None
        # gradient of loss with respect to W, filled by backward()
        self.dW: np.ndarray | None = None

    @staticmethod
    def _init_weights(in_dim: int, out_dim: int):
        # he initialisation scales random weights by sqrt(2/in_dim)
        # this prevents activations from being too large or too small at the start
        scale = np.sqrt(2.0 / in_dim)
        return np.random.randn(in_dim, out_dim) * scale

    def forward(self, x: np.ndarray):
        """
        Computes a = activation(xW).
        x is the input from the previous layer (or the raw data for the first layer).
        Caches x and z so backward() can use them later.
        """
        # cache the input — backward() needs it to compute dL/dW = x.T @ delta
        self._x = x
        # matrix multiply: (batch, in_dim) @ (in_dim, out_dim) → (batch, out_dim)
        self._z = x @ self.W
        # pass z through the activation function (relu, sigmoid, etc.)
        return self.activation.forward(self._z)

    def backward(self, grad_out: np.ndarray):
        """
        Receives dL/da (gradient from the next layer).
        Computes and stores dL/dW in self.dW — the optimizer reads this in step().
        Returns dL/dx to pass to the previous layer.
        """
        # activation derivative reuses _z cached during forward()
        da_dz = self.activation.derivative(self._z)

        # element-wise multiply — applies chain rule through the activation
        # delta is dL/dz, shape: (batch, out_dim)
        delta = grad_out * da_dz

        # dL/dW — this is what the optimizer will use to update W
        # shape: (in_dim, batch) @ (batch, out_dim) → (in_dim, out_dim), same as W
        self.dW = self._x.T @ delta

        # dL/dx — passed to the previous layer as its grad_out
        # shape: (batch, out_dim) @ (out_dim, in_dim) → (batch, in_dim)
        return delta @ self.W.T
