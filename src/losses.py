import numpy as np

_EPS = 1e-12  # numerical stability clip for BCE


class MSELoss:
    """Mean squared error — used by the network to compute loss and its gradient."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._diff = y_pred - y_true
        return float(np.mean(self._diff ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Returns dL/dy_pred; call after __call__ with the same inputs."""
        n = y_true.shape[0]
        return (2.0 / n) * self._diff


class BCELoss:
    """Binary cross-entropy — used by the network to compute loss and its gradient."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        self._p = p
        self._y = y_true
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Returns dL/dy_pred; call after __call__ with the same inputs."""
        n = y_true.shape[0]
        return (-(self._y / self._p) + (1.0 - self._y) / (1.0 - self._p)) / n


LOSS_MAP: dict = {
    "mse": MSELoss,
    "bce": BCELoss,
}


def get_loss(name: str):
    """Returns a loss instance by name; raises ValueError for unknown names."""
    name = name.lower()
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_MAP)}")
    return LOSS_MAP[name]()
