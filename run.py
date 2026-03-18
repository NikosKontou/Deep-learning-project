import sys
import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork
from datasets import get_dataset


def parse_args() -> tuple[str, str, str]:
    """Reads config path, dataset name, and CSV path from argv."""
    if len(sys.argv) != 4:
        sys.exit(
            "usage: python run.py <config.json> <dataset> <data.csv>\n"
            "datasets: auto_mpg, breast_cancer\n"
            "i.e.: python run.py ./config_regression.json auto_mpg ./data/auto-mpg.data"
        )
    return sys.argv[1], sys.argv[2], sys.argv[3]


def plot_loss(history: list[float], title: str, output_path: str) -> None:
    """Saves a loss-vs-epoch plot to output_path; called after network.fit() returns."""
    plt.figure(figsize=(7, 3))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Plot saved to {output_path}")


def evaluate(net: NeuralNetwork, X_test: np.ndarray,
             y_test: np.ndarray, dataset: str) -> None:
    """
    Prints a task-appropriate metric on the test split.
    Uses RMSE for auto_mpg and accuracy for breast_cancer.
    """
    y_pred = net.predict(X_test)
    if dataset == "auto_mpg":
        mse = float(np.mean((y_test - y_pred) ** 2))
        print(f"Test MSE:      {mse:.4f}")
        print(f"Test RMSE:     {mse ** 0.5:.4f}")
    else:
        preds = (y_pred > 0.5).astype(float)
        acc   = float(np.mean(preds == y_test))
        print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    config_path, dataset_name, csv_path = parse_args()

    print(f"Loading '{dataset_name}' from '{csv_path}' ...")
    X_train, X_test, y_train, y_test = get_dataset(dataset_name, csv_path)
    print(f"  train: {X_train.shape}  test: {X_test.shape}")

    net = NeuralNetwork.build_from_config(config_path)
    print(f"Network built from '{config_path}'\n")

    history = net.fit(X_train, y_train, epochs=200, batch_size=32)

    evaluate(net, X_test, y_test, dataset_name)
    plot_loss(history, f"Training loss — {dataset_name}", f"{dataset_name}_loss.png")
