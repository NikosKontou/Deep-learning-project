import numpy as np


class SGD:
    """
    Vanilla gradient descent
    """

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def init_state(self, layers):
        pass

    def step(self, layers):
        """Reads dW from each layer (set by backward()) and updates W directly."""
        for layer in layers:
            layer.W -= self.lr * layer.dW


class MomentumSGD:
    """
    Gradient descent with momentum.
    Keeps a running average of past gradients (velocity) to smooth updates.
    """

    def __init__(self, lr: float = 1e-2, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        # velocity buffers — one per layer, allocated in init_state()
        self._v: list[np.ndarray] = []

    def init_state(self, layers):
        """
        Creates one zero velocity buffer per layer, matching the shape of each W.
        Called by NeuralNetwork.build_from_config.
        """
        self._v = [np.zeros_like(layer.W) for layer in layers]

    def step(self, layers):
        """Updates velocity using the current gradient, then updates W using velocity."""
        for i, layer in enumerate(layers):
            # blend past velocity with current gradient
            # self._v[i] starts at zero (init_state) and grows each step
            self._v[i] = self.beta * self._v[i] + (1.0 - self.beta) * layer.dW
            layer.W -= self.lr * self._v[i]


class AdaBelief:
    """
    Adaptive optimizer that tracks how much the gradient deviates from its own prediction.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        # smoothing factor for the gradient mean (m)
        self.beta1 = beta1
        # smoothing factor for the gradient variance (s)
        self.beta2 = beta2
        # small number to prevent division by zero — does not affect direction
        self.eps = eps
        # first moment: running mean of gradients, one per layer
        self._m: list[np.ndarray] = []
        # second moment: running mean of (g - m)^2, one per layer
        self._s: list[np.ndarray] = []
        # time step counter — used for bias correction
        self._t: int = 0

    def init_state(self, layers):
        """
        Creates zero buffers for m and s, one per layer matching each W's shape.
        Called once by NeuralNetwork.build_from_config() after layers are built.
        """
        self._m = [np.zeros_like(layer.W) for layer in layers]
        self._s = [np.zeros_like(layer.W) for layer in layers]
        # _t starts at 0 and is incremented at the start of each step()

    def step(self, layers):
        """
        Updates m and s buffers, applies bias correction, then updates each layer's W.
        dW is read from each layer — it was stored there by DenseLayer.backward().
        """
        self._t += 1

        # bias correction terms. both start near 0 and grow toward 1 as _t increases so it corrects for m and s being initialized at zero
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for i, layer in enumerate(layers):
            g = layer.dW   # gradient for this layer, computed by backward()

            # self._m[i] was initialized to zero in init_state()
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g

            # self._s[i] was initialized to zero in init_state()
            self._s[i] = self.beta2 * self._s[i] + (1.0 - self.beta2) * (g - self._m[i]) ** 2

            # bias-corrected estimates — rescales m and s to compensate for zero initialization
            m_hat = self._m[i] / bc1
            s_hat = self._s[i] / bc2

            # eps prevents division by zero,
            layer.W -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


OPTIMIZER_MAP: dict = {
    "sgd": SGD,
    "momentum_sgd": MomentumSGD,
    "adabelief": AdaBelief,
}


def get_optimizer(name: str, lr: float):
    """Called by NeuralNetwork.build_from_config to attach the correct optimizer."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_MAP)}")
    return OPTIMIZER_MAP[name](lr=lr)
