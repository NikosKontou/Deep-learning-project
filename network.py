import json
import pickle
import numpy as np

from layers import DenseLayer
from losses import get_loss
from optimizers import get_optimizer


class NeuralNetwork:
    """
    Feed-forward neural network built from a JSON config.
    Orchestrates DenseLayer objects, a loss function, and an optimiser.
    Call build_from_config() to construct, then fit() to train.
    """

    def __init__(self) -> None:
        self.layers: list[DenseLayer] = []
        self._loss_fn = None
        self._optimizer = None

    # ------------------------------------------------------------------ build

    @classmethod
    def build_from_config(cls, config: dict | str) -> "NeuralNetwork":
        """
        Constructs a NeuralNetwork from a config dict or a path to a JSON file.
        Calls _build_layers, then inits loss and optimiser state.
        """
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)

        net = cls()
        net._build_layers(config)
        net._loss_fn = get_loss(config["loss"])
        net._optimizer = get_optimizer(config["optimizer"], config["learning_rate"])
        net._optimizer.init_state(net.layers)
        return net

    def _build_layers(self, config: dict) -> None:
        """Instantiates DenseLayer objects; infers in_dim from previous layer's units."""
        in_dim = config["input_dimension"]
        for spec in config["layers"]:
            if spec["type"] != "dense":
                raise ValueError(f"Unsupported layer type '{spec['type']}'")
            self.layers.append(DenseLayer(in_dim, spec["units"], spec["activation"]))
            in_dim = spec["units"]

    # ---------------------------------------------------------------- forward / backward

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Runs a full forward pass and returns the network output."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, grad: np.ndarray) -> None:
        """Propagates grad backwards through all layers, populating each layer's dW."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ------------------------------------------------------------------ training

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Runs one forward pass, computes loss, backpropagates, and calls optimiser.step.
        Returns the scalar loss value.
        """
        y_pred = self.predict(X)
        loss = self._loss_fn(y, y_pred)
        grad = self._loss_fn.gradient(y, y_pred)
        self._backward(grad)
        self._optimizer.step(self.layers)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, batch_size: int | None = None,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            verbose: bool = True) -> tuple[list[float], list[float]]:
        """
        Trains the network for a fixed number of epochs.
        Returns (train_history, val_history); val_history is empty if no val data given.
        Shuffles training data each epoch; uses full-batch if batch_size is None.
        """
        n = X.shape[0]
        batch_size = batch_size or n
        train_history: list[float] = []
        val_history:   list[float] = []

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
            batch_losses: list[float] = []

            for start in range(0, n, batch_size):
                Xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                batch_losses.append(self.train_step(Xb, yb))

            train_loss = float(np.mean(batch_losses))
            train_history.append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self._loss_fn(y_val, val_pred)
                val_history.append(float(val_loss))

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                val_str = f"  val={val_history[-1]:.6f}" if val_history else ""
                print(f"Epoch {epoch:>{len(str(epochs))}}/{epochs}"
                      f"  train={train_loss:.6f}{val_str}")

        return train_history, val_history

    # ------------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Serialises the network to disk via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "NeuralNetwork":
        """Deserialises a network saved with save()."""
        with open(path, "rb") as f:
            return pickle.load(f)