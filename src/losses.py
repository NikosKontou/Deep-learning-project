import numpy as np

# small number added to BCE to avoid computing log(0), which is undefined
_EPS = 1e-12


class MSELoss:
    """Mean Squared Error — used for regression (auto_mpg)."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        # cache the difference so gradient() can reuse it without recalculating
        self._diff = y_pred - y_true
        return float(np.mean(self._diff ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Returns dL/dy_pred — the starting gradient for backpropagation.
        Must be called after __call__() with the same inputs because it reads self._diff.
        """
        n = y_true.shape[0]
        # derivative of mean((y_pred - y_true)^2) with respect to y_pred
        return (2.0 / n) * self._diff


class BCELoss:
    """Binary Cross-Entropy — used for classification (breast_cancer)."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        # clip predictions away from 0 and 1 to prevent log(0) = -infinity
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        # cache p and y_true so gradient() can reuse them without recalculating
        self._p = p
        self._y = y_true
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Returns dL/dy_pred — the starting gradient for backpropagation.
        Must be called after __call__() with the same inputs because it reads self._p and self._y.
        """
        n = y_true.shape[0]
        # derivative of BCE with respect to y_pred
        return (-(self._y / self._p) + (1.0 - self._y) / (1.0 - self._p)) / n

LOSS_MAP: dict = {
    "mse": MSELoss,
    "bce": BCELoss,
}


def get_loss(name: str):
    """Returns a loss instance by name"""
    name = name.lower()
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_MAP)}")
    return LOSS_MAP[name]()
