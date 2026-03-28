"""
Neural Network Framework — Interactive CLI
ITC6230 Deep Learning

Usage (run from project root):
    python -m src.main
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.network import NeuralNetwork
from datasets import get_dataset
_DEPTH_CONFIGS = {
    "auto_mpg": [
        ("1 hidden layer", "configs/depth_1layer_regression.json"),
        ("3 hidden layers", "configs/depth_3layer_regression.json"),
    ],
    "breast_cancer": [
        ("1 hidden layer", "configs/depth_1layer_classification.json"),
        ("3 hidden layers", "configs/depth_3layer_classification.json"),
    ],
}
# ── fixed experiment parameters ───────────────────────────────────────────────
EPOCHS     = 100
OPTIMIZERS = ["sgd", "momentum_sgd", "adabelief"]
LR_OPTIONS = [0.1, 0.001]
BS_OPTIONS = [16, 64]
ARCHS      = ["A1", "A2"]
DATASETS   = ["auto_mpg", "breast_cancer"]

_LOSS    = {"auto_mpg": "mse",    "breast_cancer": "bce"}
_OUT_ACT = {"auto_mpg": "linear", "breast_cancer": "sigmoid"}

# ── auto mpg EDA constants ────────────────────────────────────────────────────
_MPG_COLUMNS  = ["mpg", "cylinders", "displacement", "horsepower",
                 "weight", "acceleration", "model_year", "origin", "car_name"]
_MPG_FEATURES = ["cylinders", "displacement", "horsepower",
                 "weight", "acceleration", "model_year", "origin"]
_MPG_TARGET   = "mpg"

# ── breast cancer EDA constants ───────────────────────────────────────────────
_BC_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
_BC_FEATURE_NAMES = (
    [f"{m}_mean"  for m in _BC_MEASUREMENTS] +
    [f"{m}_se"    for m in _BC_MEASUREMENTS] +
    [f"{m}_worst" for m in _BC_MEASUREMENTS]
)
_BC_TARGET = "diagnosis"


# ═════════════════════════════════════════════════════════════════════════════
# CLI helpers
# ═════════════════════════════════════════════════════════════════════════════

def _header(title: str) -> None:
    """Prints a section divider and title."""
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")


def _ask_choice(prompt: str, options: list) -> str:
    """Prints a numbered menu and returns the chosen option string."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("  > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Enter a number between 1 and {len(options)}.")


def _ask_path(prompt: str) -> str:
    """Prompts for a file path; keeps asking until the file exists."""
    while True:
        raw = input(f"\n{prompt}\n  > ").strip()
        if os.path.isfile(raw):
            return raw
        print(f"  File not found: '{raw}'. Please try again.")


def _report_dir(dataset: str) -> str:
    """Creates report/<dataset>/ if needed and returns the path."""
    path = os.path.join("report", dataset)
    os.makedirs(path, exist_ok=True)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Shared training utilities
# ═════════════════════════════════════════════════════════════════════════════
import os
import numpy as np
import matplotlib.pyplot as plt


def _plot_depth_curves(histories: list[tuple], title: str, out_path: str) -> None:
    # Auto-generate two filenames based on the provided out_path
    base, ext = os.path.splitext(out_path)
    out_path_full = f"{base}_full{ext}"
    out_path_zoom = f"{base}_zoomed{ext}"

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(histories)))
    zoom_n = 20

    total_epochs = max(len(h[1]) for h in histories)
    rect_x = total_epochs - zoom_n

    # ── 1. Full Epochs Plot ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for (label, train_h, val_h), color in zip(histories, colors):
        ax1.plot(train_h, label=f"{label} train", color=color, linewidth=1.5)
        ax1.plot(val_h, label=f"{label} val", color=color, linewidth=1.5, linestyle="--")

    ax1.axvspan(rect_x, total_epochs - 1, alpha=0.08, color="grey", label=f"zoomed region (last {zoom_n})")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8, loc="upper right")

    fig1.tight_layout()
    fig1.savefig(out_path_full, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── 2. Zoomed & Normalised Plot ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for (label, train_h, val_h), color in zip(histories, colors):
        t_tail = np.array(train_h[-zoom_n:])
        v_tail = np.array(val_h[-zoom_n:])

        # Normalise each curve independently
        for tail, ls, suffix in [(t_tail, "-", "train"), (v_tail, "--", "val")]:
            lo, hi = tail.min(), tail.max()
            norm = (tail - lo) / (hi - lo + 1e-12)
            ax2.plot(norm, color=color, linewidth=1.5, linestyle=ls, label=f"{label} {suffix}")

    ax2.set_title(f"{title} - Last {zoom_n} epochs (normalised per curve)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch offset")
    ax2.set_ylabel("Normalised loss")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc="upper right")

    fig2.tight_layout()
    fig2.savefig(out_path_zoom, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  Plots → {out_path_full}, {out_path_zoom}")

def _build_config(arch: str, optimizer: str, lr: float,
                  dataset: str, input_dim: int) -> dict:
    """
    Constructs a network config dict from architecture name and dataset.
    input_dim is read from X_train.shape[1]; output activation and loss
    are inferred from the dataset name.
    """
    if arch == "A1":
        hidden = [{"type": "dense", "units": 32, "activation": "sigmoid"}]
    else:  # A2
        hidden = [
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 32, "activation": "relu"},
        ]
    return {
        "input_dimension": input_dim,
        "layers":          hidden + [{"type": "dense", "units": 1,
                                      "activation": _OUT_ACT[dataset]}],
        "loss":            _LOSS[dataset],
        "optimizer":       optimizer,
        "learning_rate":   lr,
    }


def _compute_metrics(net: NeuralNetwork, X: np.ndarray,
                     y: np.ndarray, dataset: str) -> dict:
    """Returns MAE + RMSE for regression or accuracy for classification."""
    y_pred = net.predict(X)
    if dataset == "auto_mpg":
        mae  = float(np.mean(np.abs(y - y_pred)))
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        return {"MAE": mae, "RMSE": rmse}
    acc = float(np.mean((y_pred > 0.5).astype(float) == y))
    return {"Accuracy": acc}


def _load_dataset(dataset: str) -> tuple:
    """
    Asks user for the data file path and returns all six splits.
    Returns (dataset, X_train, X_val, X_test, y_train, y_val, y_test).
    """
    path = _ask_path(f"Enter path to '{dataset}' data file:")
    print(f"\n  Loading '{dataset}' ...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(dataset, path)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _plot_loss_curves(histories: list[tuple], title: str, out_path: str) -> None:
    """
    Plots overlaid train/val loss curves for multiple runs.
    histories is a list of (label, train_history, val_history).
    Saves to out_path; called by all experiment modes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(histories)))
    for (label, train_h, val_h), color in zip(histories, colors):
        ax.plot(train_h, label=f"{label} train",
                color=color, linewidth=1.5)
        ax.plot(val_h,   label=f"{label} val",
                color=color, linewidth=1.5, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path}")


def _print_results_table(records: list[dict], dataset: str,
                          metric_name: str) -> None:
    """Prints results in the required Problem|Optimizer|Batch|LR|Arch|Metric format."""
    col_w  = [16, 14, 7, 7, 6, 12]
    header = ["Problem", "Optimizer", "Batch", "LR", "Arch", metric_name]
    sep    = "  ".join("─" * w for w in col_w)
    print(f"\n  {sep}")
    print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(header, col_w)))
    print(f"  {sep}")
    for r in records:
        val = r.get(metric_name, "")
        row = [
            dataset,
            str(r.get("optimizer", "")),
            str(r.get("batch_size", "")),
            str(r.get("learning_rate", "")),
            str(r.get("arch", "")),
            f"{val:.4f}" if isinstance(val, float) else str(val),
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
    print(f"  {sep}")


# ═════════════════════════════════════════════════════════════════════════════
# Mode 1 — Train a single network
# ═════════════════════════════════════════════════════════════════════════════

def mode_train() -> None:
    """
    Trains one network from user-specified parameters or a JSON config file.
    Prints train/val/test metrics and saves a loss curve plot.
    """
    _header("TRAIN — Single Network")

    dataset    = _ask_choice("Select dataset:", DATASETS)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(dataset)
    report_dir = _report_dir(dataset)
    metric_name = "MAE" if dataset == "auto_mpg" else "Accuracy"

    src = _ask_choice("Configuration source:",
                      ["Architecture (A1 / A2)", "JSON config file"])

    if src == "JSON config file":
        config_path = _ask_path("Enter path to JSON config file:")
        config      = config_path
        run_label   = os.path.basename(config_path).replace(".json", "")
        bs          = int(_ask_choice("Batch size:", [str(x) for x in BS_OPTIONS]))
    else:
        arch   = _ask_choice("Architecture:",   ARCHS)
        opt    = _ask_choice("Optimizer:",       OPTIMIZERS)
        lr     = float(_ask_choice("Learning rate:", [str(x) for x in LR_OPTIONS]))
        bs     = int(_ask_choice("Batch size:",  [str(x) for x in BS_OPTIONS]))
        config = _build_config(arch, opt, lr, dataset, X_train.shape[1])
        run_label = f"{arch}_{opt}_lr{lr}_bs{bs}"

    net = NeuralNetwork.build_from_config(config)
    print(f"\n  Training for {EPOCHS} epochs ...")

    train_h, val_h = net.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=bs,
        X_val=X_val, y_val=y_val,
    )

    print()
    for split, X, y in [("train", X_train, y_train),
                         ("val",   X_val,   y_val),
                         ("test",  X_test,  y_test)]:
        m = _compute_metrics(net, X, y, dataset)
        print("  " + f"{split:5s}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    test_m = _compute_metrics(net, X_test, y_test, dataset)
    _print_results_table(
        [{"arch": run_label, "optimizer": run_label, "batch_size": bs,
          "learning_rate": "—", metric_name: list(test_m.values())[0]}],
        dataset, metric_name,
    )

    out = os.path.join(report_dir, f"train_{run_label}.png")
    _plot_loss_curves(
        [(run_label, train_h, val_h)],
        f"Training — {dataset} — {run_label}", out,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Mode 2 — Optimizer comparison
# ═════════════════════════════════════════════════════════════════════════════

def mode_optimizer_comparison() -> None:
    """
    Trains all three optimizers under a fixed arch, lr, and batch size.
    Produces one combined loss-curve plot per architecture and prints a results table.
    Satisfies Experiment 1 from the project spec.
    """
    _header("EXPERIMENT 1 — Optimizer Comparison")

    dataset    = _ask_choice("Select dataset:", DATASETS)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(dataset)
    arch       = _ask_choice("Architecture:",   ARCHS)
    lr         = float(_ask_choice("Learning rate:", [str(x) for x in LR_OPTIONS]))
    bs         = int(_ask_choice("Batch size:",  [str(x) for x in BS_OPTIONS]))
    report_dir = _report_dir(dataset)
    metric_name = "MAE" if dataset == "auto_mpg" else "Accuracy"

    histories = []
    results   = []

    for opt in OPTIMIZERS:
        print(f"\n  optimizer={opt:<12}", end="  ", flush=True)
        config = _build_config(arch, opt, lr, dataset, X_train.shape[1])
        net    = NeuralNetwork.build_from_config(config)

        train_h, val_h = net.fit(
            X_train, y_train, epochs=EPOCHS, batch_size=bs,
            X_val=X_val, y_val=y_val, verbose=False,
        )

        m = _compute_metrics(net, X_test, y_test, dataset)
        print("  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        histories.append((opt, train_h, val_h))
        results.append({
            "arch": arch, "optimizer": opt,
            "learning_rate": lr, "batch_size": bs,
            metric_name: list(m.values())[0],
        })

    out = os.path.join(report_dir,
                       f"exp1_optimizer_comparison_{arch}_lr{lr}_bs{bs}.png")
    _plot_loss_curves(
        histories,
        f"Optimizer Comparison — {dataset} — {arch}  lr={lr}  bs={bs}",
        out,
    )
    _print_results_table(results, dataset, metric_name)


# ═════════════════════════════════════════════════════════════════════════════
# Mode 3 — Network depth experiment
# ═════════════════════════════════════════════════════════════════════════════

def mode_depth_experiment() -> None:
    """
    Loads the two depth config files for the chosen dataset and trains them
    with identical optimizer, lr, and batch size to isolate the effect of depth.
    Both configs use relu hidden activations — only the number of layers differs.
    Produces one combined loss-curve plot and prints a results table.
    Satisfies Experiment 2 from the project spec.
    """
    _header("EXPERIMENT 2 — Network Depth")

    dataset = _ask_choice("Select dataset:", DATASETS)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(dataset)
    bs = int(_ask_choice("Batch size:", [str(x) for x in BS_OPTIONS]))
    report_dir = _report_dir(dataset)
    metric_name = "MAE" if dataset == "auto_mpg" else "Accuracy"

    configs = _DEPTH_CONFIGS[dataset]
    histories = []
    results = []

    print("\n  Using fixed configs (relu hidden layers, same optimizer/lr):")
    for label, cfg_path in configs:
        print(f"    {label:15s} → {cfg_path}")

    for label, cfg_path in configs:
        print(f"\n  {label}", end="  ", flush=True)

        if not os.path.isfile(cfg_path):
            print(f"\n  Config not found: '{cfg_path}' — skipping.")
            continue

        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        opt = cfg.get("optimizer", "?")
        lr = cfg.get("learning_rate", "?")

        net = NeuralNetwork.build_from_config(cfg_path)
        train_h, val_h = net.fit(
            X_train, y_train, epochs=EPOCHS, batch_size=bs,
            X_val=X_val, y_val=y_val, verbose=False,
        )

        m = _compute_metrics(net, X_test, y_test, dataset)
        print("  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        histories.append((label, train_h, val_h))
        results.append({
            "arch": label,
            "optimizer": opt,
            "learning_rate": lr,
            "batch_size": bs,
            metric_name: list(m.values())[0],
        })

    out = os.path.join(report_dir, f"exp2_depth_bs{bs}.png")
    _plot_depth_curves(
        histories,
        f"Network Depth — {dataset}  bs={bs}",
        out,
    )
    _print_results_table(results, dataset, metric_name)


# ═════════════════════════════════════════════════════════════════════════════
# Mode 4 — Learning rate sensitivity
# ═════════════════════════════════════════════════════════════════════════════

def mode_lr_sensitivity() -> None:
    """
    Trains the same arch and optimizer at both learning rates.
    Produces one combined loss-curve plot and prints a results table.
    Satisfies Experiment 3 from the project spec.
    """
    _header("EXPERIMENT 3 — Learning Rate Sensitivity")

    dataset    = _ask_choice("Select dataset:", DATASETS)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(dataset)
    arch       = _ask_choice("Architecture:",   ARCHS)
    opt        = _ask_choice("Optimizer:",       OPTIMIZERS)
    bs         = int(_ask_choice("Batch size:",  [str(x) for x in BS_OPTIONS]))
    report_dir = _report_dir(dataset)
    metric_name = "MAE" if dataset == "auto_mpg" else "Accuracy"

    histories = []
    results   = []

    for lr in LR_OPTIONS:
        print(f"\n  lr={lr}", end="  ", flush=True)
        config = _build_config(arch, opt, lr, dataset, X_train.shape[1])
        net    = NeuralNetwork.build_from_config(config)

        train_h, val_h = net.fit(
            X_train, y_train, epochs=EPOCHS, batch_size=bs,
            X_val=X_val, y_val=y_val, verbose=False,
        )

        m = _compute_metrics(net, X_test, y_test, dataset)
        print("  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        histories.append((f"lr={lr}", train_h, val_h))
        results.append({
            "arch": arch, "optimizer": opt,
            "learning_rate": lr, "batch_size": bs,
            metric_name: list(m.values())[0],
        })

    out = os.path.join(report_dir,
                       f"exp3_lr_sensitivity_{arch}_{opt}_bs{bs}.png")
    _plot_loss_curves(
        histories,
        f"Learning Rate Sensitivity — {dataset} — {arch}  {opt}  bs={bs}",
        out,
    )
    _print_results_table(results, dataset, metric_name)


# ═════════════════════════════════════════════════════════════════════════════
# Mode 5 — Full grid experiment (all 24 runs)
# ═════════════════════════════════════════════════════════════════════════════

def mode_full_grid() -> None:
    """
    Runs all 24 combinations of optimizer x lr x batch_size x architecture.
    Saves a results CSV, a 24-subplot loss-curve grid, and a val-metric bar chart.
    """
    _header("FULL GRID — 24 Configurations")

    dataset    = _ask_choice("Select dataset:", DATASETS)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(dataset)
    report_dir = _report_dir(dataset)
    metric_name = "MAE" if dataset == "auto_mpg" else "Accuracy"

    grid    = list(itertools.product(ARCHS, OPTIMIZERS, LR_OPTIONS, BS_OPTIONS))
    records = []

    for i, (arch, opt, lr, bs) in enumerate(grid, 1):
        print(f"  [{i:>2}/{len(grid)}]  arch={arch}  opt={opt:<12}  "
              f"lr={lr}  bs={bs}", end="  ", flush=True)

        config = _build_config(arch, opt, lr, dataset, X_train.shape[1])
        net    = NeuralNetwork.build_from_config(config)

        train_h, val_h = net.fit(
            X_train, y_train, epochs=EPOCHS, batch_size=bs,
            X_val=X_val, y_val=y_val, verbose=False,
        )

        train_m = list(_compute_metrics(net, X_train, y_train, dataset).values())[0]
        val_m   = list(_compute_metrics(net, X_val,   y_val,   dataset).values())[0]
        test_m  = list(_compute_metrics(net, X_test,  y_test,  dataset).values())[0]
        print(f"train={train_m:.4f}  val={val_m:.4f}  test={test_m:.4f}")

        records.append({
            "arch": arch, "optimizer": opt,
            "learning_rate": lr, "batch_size": bs,
            f"train_{metric_name}": train_m,
            f"val_{metric_name}":   val_m,
            f"test_{metric_name}":  test_m,
            "train_history": train_h,
            "val_history":   val_h,
        })

    df = pd.DataFrame(records)

    # save CSV
    csv_path = os.path.join(report_dir, "results.csv")
    df.drop(columns=["train_history", "val_history"]).to_csv(csv_path, index=False)
    print(f"\n  Results CSV → {csv_path}")

    # loss curves grid
    _plot_full_grid_curves(df, dataset, report_dir)

    # val metric bar chart
    _plot_full_grid_bar(df, dataset, metric_name, report_dir)

    # top 5 summary
    val_col   = f"val_{metric_name}"
    ascending = dataset == "auto_mpg"
    top5 = df.sort_values(val_col, ascending=ascending).head(5)
    top5_records = [
        {**row.to_dict(), metric_name: row[val_col]}
        for _, row in top5.iterrows()
    ]
    print(f"\n  Top 5 configurations by val {metric_name}:")
    _print_results_table(top5_records, dataset, metric_name)


def _plot_full_grid_curves(df: pd.DataFrame, dataset: str,
                            report_dir: str) -> None:
    """Saves a grid of 24 individual train/val loss-curve subplots."""
    n_runs = len(df)
    n_cols = 4
    n_rows = n_runs // n_cols + (1 if n_runs % n_cols else 0)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 8, n_rows * 5.6),
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

    fig.suptitle(f"{dataset} — all training curves", fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(report_dir, "curves.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curves → {path}")


def _plot_full_grid_bar(df: pd.DataFrame, dataset: str,
                         metric_name: str, report_dir: str) -> None:
    """Saves a bar chart ranking all 24 configurations by validation metric."""
    val_col = f"val_{metric_name}"
    labels  = [
        f"{r['arch']}\n{r['optimizer']}\nlr={r['learning_rate']} bs={r['batch_size']}"
        for _, r in df.iterrows()
    ]
    values  = df[val_col].tolist()
    colours = plt.cm.tab20(np.linspace(0, 1, len(values)))

    fig, ax = plt.subplots(figsize=(max(20, len(values) * 1.8), 10))
    bars = ax.bar(range(len(values)), values, color=colours,
                  edgecolor="white", width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel(f"Val {metric_name}", fontsize=9)
    ax.set_title(f"{dataset} — validation {metric_name} by configuration",
                 fontsize=11)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    path = os.path.join(report_dir, f"val_{metric_name.lower()}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart   → {path}")

# main menu
_MENU = [
    ("Train a single network",                    mode_train),
    ("Experiment 1 — Optimizer comparison",       mode_optimizer_comparison),
    ("Experiment 2 — Network depth",              mode_depth_experiment),
    ("Experiment 3 — Learning rate sensitivity",  mode_lr_sensitivity),
    ("Full grid experiment  (all 24 runs)",       mode_full_grid),
    ("Exit",                                      None),
]


def main() -> None:
    print("Neural Network Framework — ITC6230")

    while True:
        print("\n  Select an option:")
        for i, (label, _) in enumerate(_MENU, 1):
            print(f"    [{i}] {label}")

        raw = input("\n  > ").strip()
        if not raw.isdigit() or not (1 <= int(raw) <= len(_MENU)):
            print("  Invalid option.")
            continue

        label, fn = _MENU[int(raw) - 1]
        if fn is None:
            sys.exit(0)

        try:
            fn()
        except KeyboardInterrupt:
            print("\n\n back to menu.")
        except Exception as e:
            print(f"\n  Error: {e}")
            print("  Returning to menu.")


if __name__ == "__main__":
    main()