from typing import Tuple
import numpy as np
from sklearn.datasets import load_iris, make_classification


def load_dataset(name: str = "iris") -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset by name. Supported: 'iris', 'synthetic'."""
    if name == "iris":
        d = load_iris()
        return d.data, d.target
    elif name == "synthetic":
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {name}")
