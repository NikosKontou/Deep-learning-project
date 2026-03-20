import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from network import NeuralNetwork
from datasets import get_dataset

# ── fixed parameters ────────────────────────────────────────────────────────
EPOCHS = 100

# ── experiment grid ──────────────────────────────────────────────────────────
OPTIMIZERS     = ["sgd", "momentum_sgd", "adabelief"]
LEARNING_RATES = [0.1, 0.001]
BATCH_SIZES    = [16, 64]
ARCHITECTURES  = ["A1", "A2"]


def build_config(arch: str, optimizer: str, lr: float, dataset: str) -> dict:
    """
    Constructs a network config dict from an architecture name and task type.
    Output layer activation and loss are determined by dataset (regression vs classification).
    """
    is_classification = dataset == "breast_cancer"
    output_activation = "sigmoid" if is_classification else "linear"
    loss = "bce" if is_classification else "mse"
    input_dim = 30 if is_classification else 7

    if arch == "A1":
        hidden_layers = [
            {"type": "dense", "units": 32, "activation": "sigmoid"},
        ]
    else:  # A2
        hidden_layers = [
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 32, "activation": "relu"},
        ]

    output_layer = {"type": "dense", "units": 1, "activation": output_activation}

    return {
        "input_dimension": input_dim,
        "layers":          hidden_layers + [output_layer],
        "loss":            loss,
        "optimizer":       optimizer,
        "learning_rate":   lr,
    }


def compute_metric(net: NeuralNetwork, X: np.ndarray,
                   y: np.ndarray, dataset: str) -> float:
    """Returns RMSE for regression or accuracy for classification."""
    y_pred = net.predict(X)
    if dataset == "auto_mpg":
        return float(np.sqrt(np.mean((y - y_pred) ** 2)))
    preds = (y_pred > 0.5).astype(float)
    return float(np.mean(preds == y))


def run_experiment(dataset: str, data_path: str) -> pd.DataFrame:
    """
    Iterates over the full grid; trains one network per combination.
    Returns a DataFrame with one row per run including train/val histories.
    """
    print(f"Loading '{dataset}' from '{data_path}' ...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(dataset, data_path)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}\n")

    metric_name = "RMSE" if dataset == "auto_mpg" else "Accuracy"
    grid = list(itertools.product(ARCHITECTURES, OPTIMIZERS, LEARNING_RATES, BATCH_SIZES))
    records = []

    for i, (arch, opt, lr, bs) in enumerate(grid, 1):
        label = f"[{i:>2}/{len(grid)}] arch={arch}  opt={opt:<12}  lr={lr}  bs={bs}"
        print(label)

        config = build_config(arch, opt, lr, dataset)
        net    = NeuralNetwork.build_from_config(config)

        train_hist, val_hist = net.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=bs,
            X_val=X_val, y_val=y_val,
            verbose=False,
        )

        train_metric = compute_metric(net, X_train, y_train, dataset)
        val_metric   = compute_metric(net, X_val,   y_val,   dataset)
        test_metric  = compute_metric(net, X_test,  y_test,  dataset)

        print(f"         train {metric_name}={train_metric:.4f}"
              f"  val {metric_name}={val_metric:.4f}"
              f"  test {metric_name}={test_metric:.4f}")

        records.append({
            "arch":          arch,
            "optimizer":     opt,
            "learning_rate": lr,
            "batch_size":    bs,
            f"train_{metric_name.lower()}": train_metric,
            f"val_{metric_name.lower()}":   val_metric,
            f"test_{metric_name.lower()}":  test_metric,
            "train_history": train_hist,
            "val_history":   val_hist,
        })

    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, dataset: str) -> None:
    """
    Saves two figures:
      1. Loss curves grid — one subplot per run, grouped by architecture.
      2. Bar chart comparing final val metric across all runs.
    """
    metric_name = "rmse" if dataset == "auto_mpg" else "accuracy"
    n_runs = len(df)

    # ── figure 1: loss curves ─────────────────────────────────────────────
    n_cols = 4   # one column per (optimizer x lr) combo per architecture
    n_rows = n_runs // n_cols + (1 if n_runs % n_cols else 0)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 2.8),
                             squeeze=False)

    for ax, (_, row) in zip(axes.flat, df.iterrows()):
        ax.plot(row["train_history"], label="train", linewidth=1.2)
        ax.plot(row["val_history"],   label="val",   linewidth=1.2, linestyle="--")
        title = (f"{row['arch']} | {row['optimizer']}\n"
                 f"lr={row['learning_rate']}  bs={row['batch_size']}")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Epoch", fontsize=6)
        ax.set_ylabel("Loss",  fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    # hide unused subplots
    for ax in axes.flat[n_runs:]:
        ax.set_visible(False)

    fig.suptitle(f"{dataset} — training curves", fontsize=11, y=1.01)
    fig.tight_layout()
    curves_path = f"{dataset}_curves.png"
    fig.savefig(curves_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nLoss curves saved to {curves_path}")

    # ── figure 2: val metric bar chart ────────────────────────────────────
    val_col = f"val_{metric_name}"
    labels  = [
        f"{r['arch']}\n{r['optimizer']}\nlr={r['learning_rate']} bs={r['batch_size']}"
        for _, r in df.iterrows()
    ]
    values = df[val_col].tolist()
    colours = plt.cm.tab20(np.linspace(0, 1, len(values)))

    fig2, ax2 = plt.subplots(figsize=(max(10, n_runs * 0.9), 5))
    bars = ax2.bar(range(len(values)), values, color=colours, edgecolor="white", width=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax2.set_ylabel(f"Val {metric_name.upper()}", fontsize=9)
    ax2.set_title(f"{dataset} — validation {metric_name} by configuration", fontsize=11)

    # annotate bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(values) * 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig2.tight_layout()
    bar_path = f"{dataset}_val_{metric_name}.png"
    fig2.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Bar chart saved to {bar_path}")


def save_results(df: pd.DataFrame, dataset: str) -> None:
    """Writes results (without history columns) to a CSV; called after run_experiment."""
    out = df.drop(columns=["train_history", "val_history"])
    path = f"{dataset}_results.csv"
    out.to_csv(path, index=False)
    print(f"Results saved to {path}")


def print_summary(df: pd.DataFrame, dataset: str) -> None:
    """Prints the top-5 configurations ranked by validation metric."""
    metric_name = "rmse" if dataset == "auto_mpg" else "accuracy"
    val_col     = f"val_{metric_name}"
    ascending   = dataset == "auto_mpg"   # lower RMSE is better; higher accuracy is better

    top = df.sort_values(val_col, ascending=ascending).head(5)
    print(f"\n── Top 5 by val {metric_name.upper()} ──")
    cols = ["arch", "optimizer", "learning_rate", "batch_size", val_col]
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "usage: python experiment.py <dataset> <data.file>\n"
            "datasets: auto_mpg, breast_cancer"
        )

    dataset_name = sys.argv[1]
    data_path    = sys.argv[2]

    df = run_experiment(dataset_name, data_path)
    print_summary(df, dataset_name)
    save_results(df, dataset_name)
    plot_results(df, dataset_name)
