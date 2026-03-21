"""
Grid-search experiment over optimizers, learning rates, batch sizes, and architectures.

Usage (run from project root):
    python -m src.experiment <dataset> <datasets/data.file>

<dataset> must be one of: auto_mpg, breast_cancer
All outputs are saved to report/<dataset>/.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.network import NeuralNetwork
from datasets import get_dataset

# ── fixed parameters ─────────────────────────────────────────────────────────
EPOCHS = 100

# ── experiment grid ───────────────────────────────────────────────────────────
OPTIMIZERS     = ["sgd", "momentum_sgd", "adabelief"]
LEARNING_RATES = [0.1, 0.001]
BATCH_SIZES    = [16, 64]
ARCHITECTURES  = ["A1", "A2"]

# ── dataset metadata ──────────────────────────────────────────────────────────
_INPUT_DIM = {"auto_mpg": 7, "breast_cancer": 30}
_LOSS      = {"auto_mpg": "mse", "breast_cancer": "bce"}
_OUT_ACT   = {"auto_mpg": "linear", "breast_cancer": "sigmoid"}


def get_report_dir(dataset: str) -> str:
    """Returns the report subdirectory for dataset; creates it if it doesn't exist."""
    path = os.path.join("report", dataset)
    os.makedirs(path, exist_ok=True)
    return path


def build_config(arch: str, optimizer: str, lr: float, dataset: str) -> dict:
    """
    Constructs a network config dict from architecture name and dataset.
    Output activation and loss are inferred from dataset; called once per grid cell.
    """
    if arch == "A1":
        hidden = [{"type": "dense", "units": 32, "activation": "sigmoid"}]
    else:  # A2
        hidden = [{"type": "dense", "units": 32, "activation": "relu"}] * 3

    return {
        "input_dimension": _INPUT_DIM[dataset],
        "layers":          hidden + [{"type": "dense", "units": 1,
                                      "activation": _OUT_ACT[dataset]}],
        "loss":            _LOSS[dataset],
        "optimizer":       optimizer,
        "learning_rate":   lr,
    }


def compute_metric(net: NeuralNetwork, X: np.ndarray,
                   y: np.ndarray, dataset: str) -> float:
    """Returns RMSE for regression or accuracy for classification."""
    y_pred = net.predict(X)
    if dataset == "auto_mpg":
        return float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return float(np.mean((y_pred > 0.5).astype(float) == y))


def run_experiment(dataset: str, data_path: str) -> pd.DataFrame:
    """
    Iterates over the full grid; trains one network per combination.
    Returns a DataFrame with one row per run including loss histories.
    """
    print(f"Loading '{dataset}' from '{data_path}' ...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(dataset, data_path)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}\n")

    metric_name = "RMSE" if dataset == "auto_mpg" else "Accuracy"
    grid    = list(itertools.product(ARCHITECTURES, OPTIMIZERS, LEARNING_RATES, BATCH_SIZES))
    records = []

    for i, (arch, opt, lr, bs) in enumerate(grid, 1):
        print(f"[{i:>2}/{len(grid)}] arch={arch}  opt={opt:<12}  lr={lr}  bs={bs}")

        config = build_config(arch, opt, lr, dataset)
        net    = NeuralNetwork.build_from_config(config)

        train_hist, val_hist = net.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=bs,
            X_val=X_val, y_val=y_val,
            verbose=False,
        )

        train_m = compute_metric(net, X_train, y_train, dataset)
        val_m   = compute_metric(net, X_val,   y_val,   dataset)
        test_m  = compute_metric(net, X_test,  y_test,  dataset)

        print(f"         train {metric_name}={train_m:.4f}"
              f"  val {metric_name}={val_m:.4f}"
              f"  test {metric_name}={test_m:.4f}")

        records.append({
            "arch":          arch,
            "optimizer":     opt,
            "learning_rate": lr,
            "batch_size":    bs,
            f"train_{metric_name.lower()}": train_m,
            f"val_{metric_name.lower()}":   val_m,
            f"test_{metric_name.lower()}":  test_m,
            "train_history": train_hist,
            "val_history":   val_hist,
        })

    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, dataset: str, report_dir: str) -> None:
    """
    Saves two figures to report_dir:
      1. Loss curves grid — one subplot per run.
      2. Bar chart of final val metric across all runs.
    """
    metric_name = "rmse" if dataset == "auto_mpg" else "accuracy"
    n_runs = len(df)
    n_cols = 4
    n_rows = n_runs // n_cols + (1 if n_runs % n_cols else 0)

    # loss curves
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 2.8),
                             squeeze=False)
    for ax, (_, row) in zip(axes.flat, df.iterrows()):
        ax.plot(row["train_history"], label="train", linewidth=1.2)
        ax.plot(row["val_history"],   label="val",   linewidth=1.2, linestyle="--")
        ax.set_title(f"{row['arch']} | {row['optimizer']}\n"
                     f"lr={row['learning_rate']}  bs={row['batch_size']}", fontsize=7)
        ax.set_xlabel("Epoch", fontsize=6)
        ax.set_ylabel("Loss",  fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for ax in axes.flat[n_runs:]:
        ax.set_visible(False)

    fig.suptitle(f"{dataset} — training curves", fontsize=11, y=1.01)
    fig.tight_layout()
    curves_path = os.path.join(report_dir, "curves.png")
    fig.savefig(curves_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nLoss curves  → {curves_path}")

    # val metric bar chart
    val_col = f"val_{metric_name}"
    labels  = [f"{r['arch']}\n{r['optimizer']}\nlr={r['learning_rate']} bs={r['batch_size']}"
               for _, r in df.iterrows()]
    values  = df[val_col].tolist()
    colours = plt.cm.tab20(np.linspace(0, 1, len(values)))

    fig2, ax2 = plt.subplots(figsize=(max(10, n_runs * 0.9), 5))
    bars = ax2.bar(range(len(values)), values, color=colours, edgecolor="white", width=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax2.set_ylabel(f"Val {metric_name.upper()}", fontsize=9)
    ax2.set_title(f"{dataset} — validation {metric_name} by configuration", fontsize=11)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(values) * 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig2.tight_layout()
    bar_path = os.path.join(report_dir, f"val_{metric_name}.png")
    fig2.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Bar chart    → {bar_path}")


def save_results(df: pd.DataFrame, report_dir: str) -> None:
    """Writes results (without history columns) to report_dir; called after run_experiment."""
    path = os.path.join(report_dir, "results.csv")
    df.drop(columns=["train_history", "val_history"]).to_csv(path, index=False)
    print(f"Results CSV  → {path}")


def print_summary(df: pd.DataFrame, dataset: str) -> None:
    """Prints the top-5 configurations ranked by validation metric."""
    metric_name = "rmse" if dataset == "auto_mpg" else "accuracy"
    val_col     = f"val_{metric_name}"
    ascending   = dataset == "auto_mpg"

    top  = df.sort_values(val_col, ascending=ascending).head(5)
    cols = ["arch", "optimizer", "learning_rate", "batch_size", val_col]
    print(f"\n── Top 5 by val {metric_name.upper()} ──")
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "usage: python -m src.experiment <dataset> <datasets/data.file>\n"
            "datasets: auto_mpg, breast_cancer"
        )

    dataset_name = sys.argv[1]
    data_path    = sys.argv[2]

    report_dir = get_report_dir(dataset_name)

    df = run_experiment(dataset_name, data_path)
    print_summary(df, dataset_name)
    save_results(df, report_dir)
    plot_results(df, dataset_name, report_dir)