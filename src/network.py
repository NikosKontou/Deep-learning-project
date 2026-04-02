import json
import numpy as np
from src.layers import DenseLayer
from src.losses import get_loss
from src.optimizers import get_optimizer


class NeuralNetwork:
    """
    Feed-forward neural network built from a JSON config.
    Holds a list of DenseLayer objects, one loss function, and one optimizer.
    """

    def __init__(self):
        # layers are added by _build_layers(), called inside build_from_config()
        self.layers: list[DenseLayer] = []
        # loss and optimizer are set by build_from_config() using get_loss() / get_optimizer()
        self._loss_fn  = None
        self._optimizer = None

    # this is the same concept as a static factory method in Java.
    @classmethod
    def build_from_config(cls, config):
        """
        Creates a NeuralNetwork from a JSON file path or a Python dict.
        """
        # if config is a file path, read and parse it as JSON into a dict
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        else:
            IOError("cannot find config path, give a string")

        # cls() creates a new empty NeuralNetwork instance
        net = cls()
        net._build_layers(config)

        # get_loss() and get_optimizer() are in losses.py and optimizers.py
        # they read the string name from config
        net._loss_fn = get_loss(config["loss"])
        net._optimizer = get_optimizer(config["optimizer"], config["learning_rate"])

        # init_state() allocates the optimizer's internal buffers (velocity, moments etc.)
        # it must run after layers are built because buffer shapes match each layer's W shape
        net._optimizer.init_state(net.layers)
        return net

    def _build_layers(self, config: dict):
        """
        Reads the layers list from the config and creates DenseLayer objects.
        The output size of each layer becomes the input size of the next —
        this is why the weight matrices always have the correct shape automatically.
        """
        # first layer reads from the raw dataset features
        in_dim = config["input_dimension"]
        for spec in config["layers"]:
            if spec["type"] != "dense":
                raise ValueError(f"Unsupported layer type '{spec['type']}'")
            self.layers.append(DenseLayer(in_dim, spec["units"], spec["activation"]))
            # update in_dim so the next layer knows how many inputs it receives
            in_dim = spec["units"]

    # forward and backward

    def predict(self, X: np.ndarray):
        """
        Passes X through every layer in order and returns the final output.
        Used during training (inside train_step) and at evaluation time.
        """
        out = X
        for layer in self.layers:
            # each layer.forward() stores x and z internally for use in backward()
            out = layer.forward(out)
        return out

    def _backward(self, grad: np.ndarray):
        """
        Passes the loss gradient backwards through every layer in reverse order.
        Each layer.backward() reads the cached x and z from its own forward() call,
        computes dW (stored on the layer for the optimizer to read), and returns
        the gradient for the layer before it.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # training

    def train_step(self, X: np.ndarray, y: np.ndarray):
        """
        One complete learning step on a single mini-batch.
        Order: forward → loss → gradient → backward → optimizer update.
        Returns the scalar loss value for this batch.
        """
        y_pred = self.predict(X)

        # __call__ on the loss object computes and caches intermediate values
        loss = self._loss_fn(y, y_pred)

        # gradient() reads the cached values from the __call__ above
        grad = self._loss_fn.gradient(y, y_pred)

        # fills layer.dW on every layer — the optimizer reads these in step()
        self._backward(grad)

        # reads layer.dW and updates layer.W in-place for every layer
        self._optimizer.step(self.layers)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, batch_size=32,
            X_val=None, y_val=None,
            verbose: bool = True):
        """
        Trains the network for the given number of epochs.
        Returns two lists: train loss per epoch and validation loss per epoch.
        """
        n = X.shape[0]
        train_history: list[float] = []
        val_history:   list[float] = []

        for epoch in range(1, epochs + 1):
            # shuffle the training data at the start of every epoch
            # idx is a random ordering of row indices — both X and y are reordered the same way
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
            batch_losses: list[float] = []

            # slice the shuffled data into mini-batches and train on each
            for start in range(0, n, batch_size):
                Xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                batch_losses.append(self.train_step(Xb, yb))

            # average loss across all batches for this epoch
            train_loss = float(np.mean(batch_losses))
            train_history.append(train_loss)

            if X_val is not None and y_val is not None:
                # validation: forward pass only — no backward, no weight update
                val_pred = self.predict(X_val)
                val_loss = self._loss_fn(y_val, val_pred)
                val_history.append(float(val_loss))

            if epoch == 1 or epoch % 10 == 0:
                val_str = f"  val={val_history[-1]:.4f}" if val_history else ""
                print(f"Epoch {epoch}/{epochs}  train={train_loss:.4f}{val_str}")

        return train_history, val_history
