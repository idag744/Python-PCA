"""Simple MNIST loader using sklearn.fetch_openml."""
from typing import Dict
import numpy as np


def load_mnist(cache: bool = True) -> Dict[str, np.ndarray]:
    """Return dict with keys 'data' and 'target'."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist['data'].astype(np.float32)
        y = mnist['target'].astype(int)
        return {'data': X, 'target': y}
    except Exception as e:
        raise RuntimeError('Failed to fetch MNIST dataset. Install scikit-learn or provide data.') from e
