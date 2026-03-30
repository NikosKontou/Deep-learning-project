import numpy as np

RANDOM_SEED = 42


def train_val_test_split(
    X: np.ndarray, y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
):
    """
    randomizes and splits into 60/20/20 train/val/test using RANDOM_SEED.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X))
    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    te  = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    tr  = idx[n_test + n_val:]

    return X[tr], X[val], X[te], y[tr], y[val], y[te]


def standardise(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    """Z-scores X_train."""
    # use train data to calculate mean/std to X_val and X_test to prevent data leakage.
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    # return a tuple of 3 numpy arrays
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std
