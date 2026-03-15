import numpy as np


class AdaBelief:
    """
    AdaBelief optimiser.
    Interacts with Network.train_step: receives a list of DenseLayer objects
    and updates each layer's W in-place using the stored dW gradients.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-16) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: list[np.ndarray] = []   # first moment (mean of gradients)
        self._s: list[np.ndarray] = []   # second moment (variance of gradient)
        self._t: int = 0

    def init_state(self, layers) -> None:
        """Allocates moment buffers; called once by Network after layer construction."""
        self._m = [np.zeros_like(layer.W) for layer in layers]
        self._s = [np.zeros_like(layer.W) for layer in layers]

    def step(self, layers) -> None:
        """Updates each layer's W using its dW; increments internal time step."""
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for i, layer in enumerate(layers):
            g = layer.dW
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            grad_residual = g - self._m[i]
            self._s[i] = (self.beta2 * self._s[i]
                          + (1.0 - self.beta2) * grad_residual ** 2
                          + self.eps)
            m_hat = self._m[i] / bc1
            s_hat = self._s[i] / bc2
            layer.W -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


OPTIMIZER_MAP: dict = {
    "adabelief": AdaBelief,
}


def get_optimizer(name: str, lr: float):
    """Returns an optimiser instance by name; raises ValueError for unknown names."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_MAP)}")
    return OPTIMIZER_MAP[name](lr=lr)
