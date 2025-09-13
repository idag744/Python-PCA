import numpy as np
from pca.pca import normalize, eig, projection_matrix, PCA, PCA_high_dim
from numpy.testing import assert_allclose


def test_normalize_zero_mean():
    X = np.array([[0., 0.], [1., 1.], [2., 2.]])
    Xbar, mu = normalize(X)
    assert_allclose(np.mean(Xbar, axis=0), np.zeros(2))
    assert_allclose(mu, np.array([1., 1.]))


def test_eig_basic():
    A = np.array([[3.0, 2.0], [2.0, 3.0]])
    eigvals, eigvecs = eig(A)
    assert_allclose(eigvals, np.array([5.0, 1.0]))


def test_projection_matrix():
    B = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
    P = projection_matrix(B)
    expected = np.array([[5, 2, -1], [2, 2, 2], [-1, 2, 5]]) / 6
    assert_allclose(P, expected)


def test_pca_reconstruction_small():
    X = np.array([[3., 6., 7.], [8., 9., 0.], [1., 5., 2.]])
    reconst, mean, pvals, pcs = PCA(X, 1)
    assert reconst.shape == X.shape
    assert mean.shape == (3,)
    assert pvals.shape == (1,)
    assert pcs.shape == (3, 1)


def test_pca_vs_pca_high_dim():
    rng = np.random.RandomState(0)
    X = rng.randn(5, 7)
    rec1, m1, p1, pc1 = PCA(X, 3)
    rec2, m2, p2, pc2 = PCA_high_dim(X, 3)
    assert_allclose(rec1, rec2)
    assert_allclose(m1, m2)
