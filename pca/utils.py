"""Utility helpers for visualization and evaluation."""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def mse(predict: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error per sample (averaged across dataset)."""
    predict = np.asarray(predict)
    actual = np.asarray(actual)
    return np.square(predict - actual).sum(axis=1).mean()


def draw_vector(v0: np.ndarray, v1: np.ndarray, ax=None, label=None):
    """Draw an arrow vector from v0 to v1 on a matplotlib axis."""
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, color='k')
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    if label:
        ax.text(*v1, f' {label}', fontsize=10)


def plot_mnist_reconstruction(originals: np.ndarray, reconstructions: np.ndarray, n: int = 10):
    """Plot original and reconstructed MNIST digit images side-by-side."""
    originals = np.asarray(originals)
    reconstructions = np.asarray(reconstructions)
    n = min(n, originals.shape[0])
    orig_imgs = originals[:n].reshape(-1, 28, 28)
    rec_imgs = reconstructions[:n].reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, 1, figsize=(n * 1.8, 4))
    axes[0].imshow(np.concatenate(orig_imgs, axis=1), cmap='gray')
    axes[0].set_title('Original images')
    axes[0].axis('off')
    axes[1].imshow(np.concatenate(rec_imgs, axis=1), cmap='gray')
    axes[1].set_title('Reconstructed images')
    axes[1].axis('off')
    plt.tight_layout()
    return fig
