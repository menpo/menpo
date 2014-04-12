from __future__ import division
import numpy as np


#TODO: document me
def regression(X, T, regression_type, **kwargs):
    r"""
    """
    if hasattr(regression_type, '__call__'):
        regression_closure = regression_type(X, T, **kwargs)

        # note that, self is required as this closure is assigned to an object
        def regression_object_method(x):
            return regression_closure(x)

        return regression_closure
    else:
        raise ValueError("regression_type can only be: a closure defining "
                         "a particular regression technique. Several examples"
                         " of such closures can be found in "
                         "`menpo.sdm.functions` (mlr, pcr, pls, ccr, ...).")


#TODO: document me
def mlr(X, T):
    r"""
    Multivariate Linear Regression
    """
    XX = np.dot(X.T, X)
    XX = (XX + XX.T) / 2
    XT = np.dot(X.T, T)
    R = np.linalg.solve(XX, XT)

    def mlr_fitting(x):
        return np.dot(x, R)

    return mlr_fitting


#TODO: document me
def mlr_svd(X, T, variance=None):
    r"""
    Multivariate Linear Regression using SVD decomposition
    """
    R, _, _, _ = _svd_regression(X, T, variance=variance)

    def mlr_svd_fitting(x):
        return np.dot(x, R)

    return mlr_svd_fitting


#TODO: document me
def mlr_pca(X, T, variance=None):
    r"""
    Multivariate Linear Regression using PCA reconstructions
    """
    R, _, _, V = _svd_regression(X, T, variance=variance)

    def mlr_svd_fitting(x):
        x = np.dot(np.dot(x, V.T), V)
        return np.dot(x, R)

    return mlr_svd_fitting


#TODO: document me
def mlr_pca_weights(X, T, variance=None):
    r"""
    Multivariate Linear Regression using PCA weights
    """
    R, _, _, V = _svd_regression(X, T, variance=variance)
    W = np.dot(X, V.T)
    R, _, _, _ = _svd_regression(W, T)

    def mlr_svd_fitting(x):
        w = np.dot(x, V.T)
        return np.dot(w, R)

    return mlr_svd_fitting


def _svd_regression(X, T, variance=None):
    if variance is not None and not (0 < variance <= 1):
        raise ValueError("variance must be set to a number between 0 and 1.")

    U, s, V = np.linalg.svd(X)
    if variance:
        total = sum(s)
        acc = 0
        for j, y in enumerate(s):
            acc += y
            if acc / total >= variance:
                r = j+1
                break
    else:
        tol = np.max(X.shape) * np.spacing(np.max(s))
        r = np.sum(s > tol)
    U = U[:, :r]
    s = 1 / s[:r]
    V = V[:r, :]
    R = np.dot(np.dot(V.T * s, U.T), T)

    return R, U, s, V
