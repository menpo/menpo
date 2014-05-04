from __future__ import division
import numpy as np
from scipy.linalg.blas import dgemm


def eigenvalue_decomposition(S, eps=10**-10):
    r"""

    Parameters
    ----------
    S : (N, N)  ndarray
        Covariance/Scatter matrix

    Returns
    -------
    pos_eigenvectors: (N, p) ndarray
    pos_eigenvalues: (p,) ndarray
    """
    # compute eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    # sort eigenvalues from largest to smallest
    index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    # set tolerance limit
    limit = np.max(np.abs(eigenvalues)) * eps

    # select positive eigenvalues
    pos_index = eigenvalues > 0.0
    pos_eigenvalues = eigenvalues[pos_index]
    pos_eigenvectors = eigenvectors[:, pos_index]
    # check they are within the expected tolerance
    index = pos_eigenvalues > limit
    pos_eigenvalues = pos_eigenvalues[index]
    pos_eigenvectors = pos_eigenvectors[:, index]

    return pos_eigenvectors, pos_eigenvalues


def principal_component_decomposition(X, whiten=False, center=True,
                                      bias=False):
    r"""
    Apply PCA on the data matrix X.

    Parameters
    ----------
    x : (n_samples, n_features) ndarray
        Training data

    Returns
    -------
    eigenvectors : (n_components, n_features) ndarray
    eigenvalues : (n_components,) ndarray
    """
    n_samples, n_features = X.shape

    if bias:
        N = n_samples
    else:
        N = n_samples - 1.0

    if center:
        # center data
        mean_vector = np.mean(X, axis=0)
    else:
        mean_vector = np.zeros(n_features)

    X = X - mean_vector

    if n_features < n_samples:
        # compute covariance matrix
        # S:  n_features  x  n_features
        S = dgemm(alpha=1.0, a=X.T, b=X.T, trans_b=True) / N
        # S should be perfectly symmetrical, but numerical error can creep
        # in. Enforce symmetry here to avoid creating complex
        # eigenvectors from eigendecomposition
        S = (S + S.T) / 2.0

        # perform eigenvalue decomposition
        # eigenvectors:  n_features x  n_features
        # eigenvalues:   n_features
        eigenvectors, eigenvalues = eigenvalue_decomposition(S)

        if whiten:
            # whiten eigenvectors
            eigenvectors *= np.sqrt(1.0 / eigenvalues)

    else:
        # n_features > n_samples
        # compute covariance matrix
        # S:  n_samples  x  n_samples
        S = dgemm(alpha=1.0, a=X.T, b=X.T, trans_a=True) / N
        # S should be perfectly symmetrical, but numerical error can creep
        # in. Enforce symmetry here to avoid creating complex
        # eigenvectors from eigendecomposition
        S = (S + S.T) / 2.0

        # perform eigenvalue decomposition
        # eigenvectors:  n_samples  x  n_samples
        # eigenvalues:   n_samples
        eigenvectors, eigenvalues = eigenvalue_decomposition(S)

        # compute final eigenvectors
        # eigenvectors:  n_samples  x  n_features
        if whiten:
            w = (N * eigenvalues) ** -1
        else:
            w = np.sqrt(1.0 / (N * eigenvalues))
        eigenvectors = w * dgemm(alpha=1.0, a=X.T, b=eigenvectors.T,
                                 trans_b=True)

    # transpose eigenvectors
    # eigenvectors:  n_samples  x  n_features
    eigenvectors = eigenvectors.T

    return eigenvectors, eigenvalues, mean_vector
