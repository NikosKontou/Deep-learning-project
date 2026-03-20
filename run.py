import sys
import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork
from datasets import get_dataset

EPOCHS = 100


def parse_args() -> tuple[str, str, str]:
    """Reads config path, dataset name, and data path from argv; exits on wrong usage."""
    if len(sys.argv) != 4:
        sys.exit(
            "usage: python run.py <config.json> <dataset> <data.file>\n"
            "datasets: auto_mpg, breast_cancer"
        )
    return sys.argv[1], sys.argv[2], sys.argv[3]


def plot_loss(train_history: list[float], val_history: list[float],
              title: str, output_path: str) -> None:
    """Saves a train/val loss-vs-epoch plot; called after network.fit() returns."""
    plt.figure(figsize=(7, 3))
    plt.plot(train_history, label="train")
    plt.plot(val_history,   label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
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
    config_path, dataset_name, data_path = parse_args()

    print(f"Loading '{dataset_name}' from '{data_path}' ...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(dataset_name, data_path)
    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    net = NeuralNetwork.build_from_config(config_path)
    print(f"Network built from '{config_path}'\n")

    train_history, val_history = net.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=32,
        X_val=X_val, y_val=y_val,
    )

    evaluate(net, X_test, y_test, dataset_name)
    plot_loss(train_history, val_history,
              f"Training loss — {dataset_name}", f"{dataset_name}_loss.png")