import numpy as np
import pandas as pd
from datasets.utils import standardise, train_val_test_split


def load_breast_cancer(data_path: str):
    """
    loads the data_path drops the id column, encodes M=1/B=0
    """
    df = pd.read_csv(data_path, header=None)
    # column 0 = id (drop), column 1 = diagnosis, columns 2..31 = features
    y = (df.iloc[:, 1] == "M").astype(float).to_numpy().reshape(-1, 1)
    X = df.iloc[:, 2:].to_numpy(dtype=float)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    X_train, X_val, X_test = standardise(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test
