from __future__ import division
import numpy as np
from scipy.linalg.blas import dgemm


class GPCA(object):

    def __init__(self, n_components=None, center=True, whiten=False):
        """
        :param n_components:
        :param center:
        :param whiten:
        :return:
        """
        self.n_components = n_components
        self.center = center
        self.whiten = whiten

    def fit(self, X):
        """
        :param X:
        :return:
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """
        :param X:
        :return:
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
        """
        :param X:
        :return:
        """
        if self.center:
            X = X - self.mean_
        Z = dgemm(alpha=1.0, a=X.T, b=self.components_.T, trans_a=True)
        # Z = np.dot(X, self.components_.T)
        return Z

    def inverse_transform(self, Z):
        """
        :param Z:
        :return:
        """
        X = dgemm(alpha=1.0, a=Z.T, b=self.components_.T,
                  trans_a=True, trans_b=True)
        #X = np.dot(Z, self.components_)
        if self.center:
            X = X + self.mean_

        return X


def _eigenvalue_decomposition(S, eps=10**-10):
    """
    :param S:
    :return:
    """
    # compute eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(S)

    # sort eigenvalues from largest to smallest
    index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    # set tolerance limit
    limit = np.max(np.abs(eigenvalues)) * eps

    # positive eigenvalues
    pos_index = eigenvalues > 0
    pos_eigenvalues = eigenvalues[pos_index]
    pos_eigenvectors = eigenvectors[:, pos_index]

    index = pos_eigenvalues > limit
    pos_eigenvalues = pos_eigenvalues[index]
    pos_eigenvectors = pos_eigenvectors[:, index]

    # negative eigenvalues
    # neg_index = eigenvalues < 0
    # neg_eigenvalues = eigenvalues[neg_index]
    # neg_eigenvectors = eigenvectors[:, neg_index]
    #
    # index = np.abs(neg_eigenvalues) > limit
    # neg_eigenvalues = neg_eigenvalues[index]
    # neg_eigenvectors = neg_eigenvectors[:, index]

    return pos_eigenvectors, pos_eigenvalues
