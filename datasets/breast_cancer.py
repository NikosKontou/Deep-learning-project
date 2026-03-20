"""
Breast Cancer Wisconsin — binary classification dataset.
Raw file: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
M is encoded as 1, B as 0.
"""

import numpy as np
import pandas as pd
from .utils import standardise, train_val_test_split

_N_FEATURES = 30


def load_breast_cancer(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads Breast Cancer Wisconsin from data_path; drops the id column, encodes M=1/B=0.
    Returns X_train, X_test, y_train, y_test as float64 arrays.
    """
    df = pd.read_csv(data_path, header=None)
    # column 0 = id (drop), column 1 = diagnosis, columns 2..31 = features
    y = (df.iloc[:, 1] == "M").astype(float).to_numpy().reshape(-1, 1)
    X = df.iloc[:, 2:].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_val_test_split(X, y)
    X_train, X_test = standardise(X_train, X_test)
    return X_train, X_test, y_train, y_test