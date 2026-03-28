import numpy as np


class SGD:
    """
    Vanilla stochastic gradient descent: θ_{t+1} = θ_t - η * g_t.
    Interacts with Network.train_step via init_state / step.
    """

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def init_state(self, layers):
        """No state required; exists so all optimisers are the same."""

    def step(self, layers):
        """Updates each layer's W in-place using its dW gradient."""
        for layer in layers:
            layer.W -= self.lr * layer.dW


class MomentumSGD:
    """
    SGD with momentum: v_t = β*v_{t-1} + (1-β)*g_t, θ_{t+1} = θ_t - η*v_t.
    Interacts with Network.train_step via init_state / step.
    """

    def __init__(self, lr: float = 1e-2, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self._v: list[np.ndarray] = []

    def init_state(self, layers):
        """Allocates velocity buffers; called once by Network after layer construction."""
        self._v = [np.zeros_like(layer.W) for layer in layers]

    def step(self, layers):
        """Updates velocity then W in-place for each layer."""
        for i, layer in enumerate(layers):
            self._v[i] = self.beta * self._v[i] + (1.0 - self.beta) * layer.dW
            layer.W -= self.lr * self._v[i]


class AdaBelief:
    """
    AdaBelief: tracks gradient prediction error (g - m)^2 instead of g^2.
    Update rule: θ_{t+1} = θ_t - η * m̂_t / (√s̃_t + ε).
    Interacts with Network.train_step via init_state / step.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: list[np.ndarray] = []   # first moment
        self._s: list[np.ndarray] = []   # second moment (belief)
        self._t: int = 0

    def init_state(self, layers):
        """Allocates moment buffers; called once by Network after layer construction."""
        self._m = [np.zeros_like(layer.W) for layer in layers]
        self._s = [np.zeros_like(layer.W) for layer in layers]

    def step(self, layers):
        """Updates m, s with bias correction then applies the AdaBelief update rule."""
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for i, layer in enumerate(layers):
            g = layer.dW
            # first moment: running mean of gradients
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            # second moment: running mean of squared gradient prediction error
            self._s[i] = self.beta2 * self._s[i] + (1.0 - self.beta2) * (g - self._m[i]) ** 2
            # bias-corrected estimates
            m_hat = self._m[i] / bc1
            s_hat = self._s[i] / bc2
            # ε is outside the sqrt — numerical stability only
            layer.W -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


OPTIMIZER_MAP: dict = {
    "sgd":          SGD,
    "momentum_sgd": MomentumSGD,
    "adabelief":    AdaBelief,
}


def get_optimizer(name: str, lr: float):
    """Returns an optimiser instance by name; raises ValueError for unknown names."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_MAP)}")
    return OPTIMIZER_MAP[name](lr=lr)
