import sys
import numpy as np
import matplotlib.pyplot as plt

from src.network import NeuralNetwork
from datasets import get_dataset

EPOCHS = 100


def parse_args():
    """config path, dataset name, data path are required"""
    if len(sys.argv) != 4:
        sys.exit(
            "usage: python -m src.train <config.json> <dataset> <data.file>\n"
            "datasets: auto_mpg, breast_cancer"
        )
    return sys.argv[1], sys.argv[2], sys.argv[3]

def evaluate(net: NeuralNetwork, X_test: np.ndarray,
             y_test: np.ndarray, dataset: str):
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
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(dataset_name, data_path)
    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    net = NeuralNetwork.build_from_config(config_path)

    train_history, val_history = net.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=32,
        X_val=X_val, y_val=y_val)

    evaluate(net, X_test, y_test, dataset_name)
