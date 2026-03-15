"""
Minimal usage example — regression with MSE and classification with BCE.
Run from the nn_framework/ directory: python example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork

rng = np.random.default_rng(42)


# ── regression example ─────────────────────────────────────────────────────
def regression_example():
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] + X[:, 1] ** 2).reshape(-1, 1)

    net = NeuralNetwork.build_from_config("config.json")
    history = net.fit(X, y, epochs=200, batch_size=32)

    plt.figure(figsize=(7, 3))
    plt.plot(history)
    plt.title("Regression — MSE loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("regression_loss.png", dpi=120)
    plt.close()
    print("Regression plot saved to regression_loss.png\n")


# ── classification example ──────────────────────────────────────────────────
def classification_example():
    X = rng.standard_normal((300, 10))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

    config = {
        "input_dimension": 10,
        "layers": [
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 16, "activation": "relu"},
            {"type": "dense", "units": 1,  "activation": "sigmoid"},
        ],
        "loss": "bce",
        "optimizer": "adabelief",
        "learning_rate": 0.001,
    }

    net = NeuralNetwork.build_from_config(config)
    history = net.fit(X, y, epochs=200, batch_size=32)

    preds = (net.predict(X) > 0.5).astype(float)
    accuracy = float(np.mean(preds == y))
    print(f"Classification accuracy: {accuracy:.3f}")

    plt.figure(figsize=(7, 3))
    plt.plot(history)
    plt.title("Classification — BCE loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.tight_layout()
    plt.savefig("classification_loss.png", dpi=120)
    plt.close()
    print("Classification plot saved to classification_loss.png")


if __name__ == "__main__":
    regression_example()
    classification_example()
