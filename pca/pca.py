"""Core PCA implementation (from scratch).

Functions:
- normalize(X): center data (zero-mean)
- eig(S): eigendecomposition of symmetric matrix with sort (descending)
- projection_matrix(B): compute orthogonal projection matrix onto columns of B
- PCA(X, num_components): standard PCA via covariance (D x D eigendecomp)
- PCA_high_dim(X, num_components): PCA optimized for N << D
"""
from typing import Tuple
import numpy as np


def normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center X to zero mean along columns.

    Returns (X_centered, mean_vector).
    """
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    Xbar = X - mu
    return Xbar, mu


def eig(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of symmetric matrix S.

    Returns (eigvals_sorted_desc, eigvecs_sorted_desc)
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def projection_matrix(B: np.ndarray) -> np.ndarray:
    """Compute projection matrix P = B (B^T B)^{-1} B^T."""
    B = np.asarray(B, dtype=float)
    P = B @ np.linalg.pinv(B)
    return P


def PCA(X: np.ndarray, num_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform PCA using the covariance matrix approach.

    Parameters
    ----------
    X : (N, D) array
        Data matrix with N samples, D features.
    num_components : int
        Number of principal components to keep.

    Returns
    -------
    reconst : (N, D) array
        Reconstruction of X using `num_components` components.
    mean : (D,) array
        Mean vector subtracted from X.
    principal_vals : (num_components,) array
        Leading eigenvalues.
    principal_components : (D, num_components) array
        Principal axes (columns).
    """
    X = np.asarray(X, dtype=float)
    X_bar, mu = normalize(X)
    N, D = X.shape
    S = (X_bar.T @ X_bar) / N
    eig_vals, eig_vecs = eig(S)
    principal_vals = eig_vals[:num_components]
    principal_components = np.real(eig_vecs[:, :num_components])
    Z = X_bar @ principal_components
    reconst = Z @ principal_components.T + mu
    return reconst, mu, principal_vals, principal_components


def PCA_high_dim(X: np.ndarray, num_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Efficient PCA when D >> N. Uses eigendecomposition of (N x N) matrix."""
    X = np.asarray(X, dtype=float)
    X_bar, mu = normalize(X)
    N, D = X.shape
    M = (X_bar @ X_bar.T) / N
    eig_vals_small, eig_vecs_small = eig(M)
    U_all = X_bar.T @ eig_vecs_small
    norms = np.linalg.norm(U_all, axis=0)
    nonzero = norms > 0
    U_all[:, nonzero] = U_all[:, nonzero] / norms[nonzero]
    principal_vals = eig_vals_small[:num_components]
    principal_components = np.real(U_all[:, :num_components])
    Z = X_bar @ principal_components
    reconst = Z @ principal_components.T + mu
    return reconst, mu, principal_vals, principal_components
