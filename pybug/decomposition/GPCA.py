from __future__ import division
import numpy as np
from scipy.linalg.blas import dgemm


class GPCA(object):
    r"""
    Generalized Principal Component Analysis (GPCA) by Eigenvalue
    Decomposition of the data's scatter matrix.

    This implementation uses the scipy.linalg implementation of the
    eigenvalue decomposition. Similar to the Scikit-Learn PCA implementation
    it only works for dense arrays, however, this one should scale better to
    large dimensional data.

    The class interface matches the one of the Scikit-Learn PCA class

    Parameters
    -----------
    n_components : int or None
        Number of components to be retained.
        if n_components is not set all components

    center : bool, optional
        When True GPCA is performed after mean centering the data

    whiten : bool, optional
        When True (False by default) transforms the `components_` to ensure
        that they covariance is the identity matrix.
    """

    def __init__(self, n_components=None, center=True, whiten=False):
        self.n_components = n_components
        self.center = center
        self.whiten = whiten

    def fit(self, X):
        r"""
        Apply GPCA on the data matrix X

        Parameters
        ----------
        x : (n_samples, n_features) ndarray
            Training data

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def _fit(self, X):
        r"""
        Apply GPCA on the data matrix X

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

        if self.center:
            # center data
            self.mean_ = np.mean(X, axis=0)
            X -= self.mean_

        if n_features < n_samples:
            # compute covariance matrix
            S = dgemm(alpha=1.0, a=X.T, b=X.T, trans_b=True)
            # perform eigenvalue decomposition
            eigenvectors, eigenvalues = _eigenvalue_decomposition(S)

            if self.whiten:
                # whiten the eigenvectors
                eigenvectors *= eigenvalues ** -0.5

        else:
            # n_features > n_samples
            # compute covariance matrix
            S = dgemm(alpha=1.0, a=X.T, b=X.T, trans_a=True)
            # perform eigenvalue decomposition
            eigenvectors, eigenvalues = _eigenvalue_decomposition(S)

            aux = 2
            if self.whiten:
                # will cause the eigenvectors to be whiten
                aux = 1

            # compute the final eigenvectors
            w = eigenvalues ** (-1 / aux)
            eigenvectors = w * dgemm(alpha=1.0, a=X.T, b=eigenvectors.T,
                                     trans_b=True)

        # transpose eigenvectors so that it is n_samples x n_features
        eigenvectors = eigenvectors.T

        if self.n_components is None:
            # set # of components to number of recovered eigenvalues
            self.n_components = eigenvalues.shape[0]
        self.components_ = eigenvectors[:self.n_components, :]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (self.explained_variance_ /
                                          self.explained_variance_.sum())

        return eigenvectors, eigenvalues

    def transform(self, X):
        r"""
        Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : (n_samples, n_features) ndarray

        Returns
        -------
        Z: (n_samples, n_components) ndarray

        """
        if self.center:
            X = X - self.mean_
        Z = dgemm(alpha=1.0, a=X.T, b=self.components_.T, trans_a=True)
        # Z = np.dot(X, self.components_.T)
        return Z

    def inverse_transform(self, Z):
        r"""
        Parameters
        ----------
        Z : (n_samples, n_components) ndarray

        Returns
        -------
        X: (n_samples, n_features) ndarray

        Notes
        -----
        If whitening is used, inverse_transform does not perform the
        exact inverse operation of transform.
        """
        X = dgemm(alpha=1.0, a=Z.T, b=self.components_.T,
                  trans_a=True, trans_b=True)
        #X = np.dot(Z, self.components_)
        if self.center:
            X = X + self.mean_

        return X


def _eigenvalue_decomposition(S, eps=10**-10):
    r"""

    Parameters
    ----------
    S : (N, N)  ndarray
        Scatter matrix

    Returns
    -------
    pos_eigenvectors: (N, p) ndarray
    pos_eigenvalues: (p,) ndarray
    neg_eigenvectors: (N, n) ndarray
    neg_eigenvalues: (n,) ndarray
    """
    # compute eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(S)
    # sort eigenvalues from largest to smallest
    index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    # set tolerance limit
    limit = np.max(np.abs(eigenvalues)) * eps

    # select positive eigenvalues
    pos_index = eigenvalues > 0
    pos_eigenvalues = eigenvalues[pos_index]
    pos_eigenvectors = eigenvectors[:, pos_index]
    # check they are within the expected tolerance
    index = pos_eigenvalues > limit
    pos_eigenvalues = pos_eigenvalues[index]
    pos_eigenvectors = pos_eigenvectors[:, index]

    # # select negative eigenvalues
    # neg_index = eigenvalues < 0
    # neg_eigenvalues = eigenvalues[neg_index]
    # neg_eigenvectors = eigenvectors[:, neg_index]
    # # check they are within the expected tolerance
    # index = np.abs(neg_eigenvalues) > limit
    # neg_eigenvalues = neg_eigenvalues[index]
    # neg_eigenvectors = neg_eigenvectors[:, index]
    # # resort them
    # index = np.argsort(np.abs(neg_eigenvalues))[::-1]
    # neg_eigenvalues = neg_eigenvalues[index]
    # neg_eigenvectors = neg_eigenvectors[:, index]

    return pos_eigenvectors, pos_eigenvalues
