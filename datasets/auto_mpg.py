"""
Auto MPG — regression dataset (UCI id=9).
Raw file: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
Whitespace-delimited, no header, '?' marks missing horsepower values.
Column order: mpg cylinders displacement horsepower weight acceleration model_year origin car_name
"""

import numpy as np
import pandas as pd
from datasets.utils import standardise, train_val_test_split

_COLUMNS  = ["mpg", "cylinders", "displacement", "horsepower",
             "weight", "acceleration", "model_year", "origin", "car_name"]
_FEATURES = ["cylinders", "displacement", "horsepower",
             "weight", "acceleration", "model_year", "origin"]
_TARGET   = "mpg"


def load_auto_mpg(data_path: str) -> tuple[np.ndarray, ...]:
    """
    Loads Auto MPG from data_path; drops missing horsepower rows and standardises.
    Returns X_train, X_val, X_test, y_train, y_val, y_test as float64 arrays.
    """
    df = pd.read_csv(data_path, sep=r"\s+", header=None,
                     names=_COLUMNS, na_values="?")
    df = df.dropna(subset=["horsepower"])

    X = df[_FEATURES].to_numpy(dtype=float)
    y = df[_TARGET].to_numpy(dtype=float).reshape(-1, 1)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    X_train, X_val, X_test = standardise(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test
