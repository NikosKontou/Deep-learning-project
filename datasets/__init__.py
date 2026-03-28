from datasets.auto_mpg import load_auto_mpg
from datasets.breast_cancer import load_breast_cancer

DATASET_MAP = {
    "auto_mpg":      load_auto_mpg,
    "breast_cancer": load_breast_cancer,
}


def get_dataset(name: str, data_path: str):
    """chooses a loader and returns X,y train/test/val 60/20/20."""
    name = name.lower()
    if name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset")
    return DATASET_MAP[name](data_path)
