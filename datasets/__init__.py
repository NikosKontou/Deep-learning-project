from datasets.auto_mpg import load_auto_mpg
from datasets.breast_cancer import load_breast_cancer

DATASET_MAP = {
    "auto_mpg":      load_auto_mpg,
    "breast_cancer": load_breast_cancer,
}


def get_dataset(name: str, data_path: str) -> tuple:
    """Dispatches to the correct loader; returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    name = name.lower()
    if name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_MAP)}")
    return DATASET_MAP[name](data_path)
