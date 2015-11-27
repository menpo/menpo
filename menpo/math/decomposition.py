from __future__ import division
import numpy as np
from scipy.sparse import issparse
from .linalg import dot_inplace_right


def eigenvalue_decomposition(C, is_inverse=False, eps=1e-10):
    r"""
    Eigenvalue decomposition of a given covariance (or scatter) matrix.

    Parameters
    ----------
    C : ``(N, N)`` `ndarray` or `scipy.sparse`
        The Covariance/Scatter matrix. If it is a `numpy.array`, then
        `numpy.linalg.eigh` is used. If it is an instance of `scipy.sparse`,
        then `scipy.sparse.linalg.eigsh` is used. If it is a precision matrix
        (inverse covariance), then set `is_inverse=True`.
    is_inverse : `bool`, optional
        It ``True``, then it is assumed that `C` is a precision matrix (
        inverse covariance). Thus, the eigenvalues will be inverted. If
        ``False``, then it is assumed that `C` is a covariance matrix.
    eps : `float`, optional
        Tolerance value for positive eigenvalue. Those eigenvalues smaller
        than the specified eps value, together with their corresponding
        eigenvectors, will be automatically discarded. The final
        limit is computed as ::

            limit = np.max(np.abs(eigenvalues)) * eps

    Returns
    -------
    pos_eigenvectors : ``(N, p)`` `ndarray`
        The matrix with the eigenvectors corresponding to positive eigenvalues.
    pos_eigenvalues : ``(p,)`` `ndarray`
        The array of positive eigenvalues.
    """
    # compute eigenvalue decomposition
    if issparse(C):
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(C, k=C.shape[0] - 1)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(C)

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

    # if C was a precision matrix (inverse covariance), then invert and re-sort
    # the eigenvalues
    if is_inverse:
        pos_eigenvalues = pos_eigenvalues[::-1] ** -1
        pos_eigenvectors = pos_eigenvectors[:, ::-1]

    return pos_eigenvectors, pos_eigenvalues


def pca(X, centre=True, inplace=False, eps=1e-10):
    r"""
    Apply Principal Component Analysis (PCA) on the data matrix `X`. In the case
    where the data matrix is very large, it is advisable to set
    ``inplace = True``. However, note this destructively edits the data matrix
    by subtracting the mean inplace.

    Parameters
    ----------
    X : ``(n_samples, n_dims)`` `ndarray`
        Data matrix.
    centre : `bool`, optional
        Whether to centre the data matrix. If `False`, zero will be subtracted.
    inplace : `bool`, optional
        Whether to do the mean subtracting inplace or not. This is crucial if
        the data matrix is greater than half the available memory size.
    eps : `float`, optional
        Tolerance value for positive eigenvalue. Those eigenvalues smaller
        than the specified eps value, together with their corresponding
        eigenvectors, will be automatically discarded.

    Returns
    -------
    U (eigenvectors) : ``(``(n_components, n_dims)``)`` `ndarray`
        Eigenvectors of the data matrix.
    l (eigenvalues) : ``(n_components,)`` `ndarray`
        Positive eigenvalues of the data matrix.
    m (mean vector) : ``(n_dimensions,)`` `ndarray`
        Mean that was subtracted from the data matrix.
    """
    n, d = X.shape

    if centre:
        # centre data
        # m (mean vector): d
        m = np.mean(X, axis=0)
    else:
        m = np.zeros(d, dtype=X.dtype)

    # This is required if the data matrix is very large!
    if inplace:
        X -= m
    else:
        X = X - m

    if d < n:
        # compute covariance matrix
        # C (covariance): d x d
        C = np.dot(X.conj().T, X) / (n - 1)
        # C should be perfectly symmetrical, but numerical error can creep
        # in. Enforce symmetry here to avoid creating complex eigenvectors
        C = (C + C.conj().T) / 2.0

        # perform eigenvalue decomposition
        # U (eigenvectors): d x n
        # s (eigenvalues):  n
        U, l = eigenvalue_decomposition(C, is_inverse=False, eps=eps)

        # transpose U
        # U: n x d
        U = U.T

    else:
        # d > n
        # compute small covariance matrix
        # C (covariance): n x n
        C = np.dot(X, X.conj().T) / (n - 1)
        # C should be perfectly symmetrical, but numerical error can creep
        # in. Enforce symmetry here to avoid creating complex eigenvectors
        C = (C + C.conj().T) / 2.0

        # perform eigenvalue decomposition
        # V (eigenvectors): n x n
        # s (eigenvalues):  n
        V, l = eigenvalue_decomposition(C, is_inverse=False, eps=eps)

        # compute final eigenvectors
        # U: n x d
        w = np.sqrt(1.0 / ((n - 1) * l))
        dot = dot_inplace_right if inplace else np.dot
        U = dot(V.conj().T, X)
        U *= w[:, None]

    return U, l, m


# The default value of eps tolerance is set to 1e-5 (instead of 1e-10 that used
# to be). This is done in order for pcacov to work for inverse single precision C
# i.e. is_inverse=True and dtype=np.float32. 1e-10 works perfectly when the
# covariance matrix has double precision (np.float64). However, if C has single
# precision (np.float32) and is inverse, then the first two eigenvectors end up
# having noise.
def pcacov(C, is_inverse=False, eps=1e-5):
    r"""
    Apply Principal Component Analysis (PCA) given a covariance/scatter matrix
    `C`. In the case where the data matrix is very large, it is advisable to set
    ``inplace = True``. However, note this destructively edits the data matrix
    by subtracting the mean inplace.

    Parameters
    ----------
    C : ``(N, N)`` `ndarray` or `scipy.sparse`
        The Covariance/Scatter matrix. If it is a precision matrix (inverse
        covariance), then set `is_inverse=True`.
    is_inverse : `bool`, optional
        It ``True``, then it is assumed that `C` is a precision matrix (
        inverse covariance). Thus, the eigenvalues will be inverted. If
        ``False``, then it is assumed that `C` is a covariance matrix.
    eps : `float`, optional
        Tolerance value for positive eigenvalue. Those eigenvalues smaller
        than the specified eps value, together with their corresponding
        eigenvectors, will be automatically discarded.

    Returns
    -------
    U (eigenvectors) : ``(n_components, n_dims)`` `ndarray`
        Eigenvectors of the data matrix.
    l (eigenvalues) : ``(n_components,)`` `ndarray`
        Positive eigenvalues of the data matrix.
    """
    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be square.')

    # C should be perfectly symmetrical, but numerical error can creep in.
    # Enforce symmetry here to avoid creating complex eigenvectors
    C = (C + C.conj().T) / 2.0

    # C (covariance): d x d
    # perform eigenvalue decomposition
    # U (eigenvectors): d x n
    # s (eigenvalues):  n
    U, l = eigenvalue_decomposition(C, is_inverse=is_inverse, eps=eps)

    # transpose U
    # U: n x d
    U = U.conj().T

    return U, l


def ipca(B, U_a, l_a, n_a, m_a=None, f=1.0, eps=1e-10):
    r"""
    Perform Incremental PCA on the eigenvectors ``U_a``, eigenvalues ``l_a`` and
    mean vector ``m_a`` (if present) given a new data matrix ``B``.

    Parameters
    ----------
    B : ``(n_samples, n_dims)`` `ndarray`
        New data matrix.
    U_a : ``(n_components, n_dims)`` `ndarray`
        Eigenvectors to be updated.
    l_a : (n_components) `ndarray`
        Eigenvalues to be updated.
    n_a : `int`
        Total number of samples used to produce U_a, s_a and m_a.
    m_a : ``(n_dims,)`` `ndarray`, optional
        Mean to be updated. If ``None`` or ``(n_dims,)`` `ndarray` filled
        with 0s the data matrix will not be centred.
    f : ``[0, 1]`` `float`, optional
        Forgetting factor that weights the relative contribution of new
        samples vs old samples. If 1.0, all samples are weighted equally
        and, hence, the results is the exact same as performing batch
        PCA on the concatenated list of old and new simples. If <1.0,
        more emphasis is put on the new samples. See [1] for details.
    eps : `float`, optional
        Tolerance value for positive eigenvalue. Those eigenvalues smaller
        than the specified eps value, together with their corresponding
        eigenvectors, will be automatically discarded.

    Returns
    -------
    U (eigenvectors) : ``(n_components, n_dims)`` `ndarray`
        Updated eigenvectors.
    s (eigenvalues) : ``(n_components,)`` `ndarray`
        Updated positive eigenvalues.
    m (mean vector) : ``(n_dims,)`` `ndarray`
        Updated mean.

    References
    ----------
    .. [1] David Ross, Jongwoo Lim, Ruei-Sung Lin, Ming-Hsuan Yang.
       "Incremental Learning for Robust Visual Tracking". IJCV, 2007.
    """
    # multiply current eigenvalues by total number of samples and square
    # root them to obtain singular values of the original data.
    s_a = np.sqrt((n_a - 1) * l_a)

    # obtain number of dimensions and number of samples of new data.
    n_b, d = B.shape
    # multiply the number of samples of the original data by the forgetting
    # factor
    n_a *= f
    # total number of samples
    n = n_a + n_b

    if m_a is not None and not np.all(m_a == 0):
        # centred ipca; compute mean of new data
        m_b = np.mean(B, axis=0)
        # compute new mean
        m = (n_a / n) * m_a + (n_b / n) * m_b
        # centre new data
        B = B - m_b
        # augment centred data with extra sample
        B = np.vstack((B, np.sqrt((n_a * n_b) / n) * (m_b - m_a)))
    else:
        m = np.zeros(d, dtype=B.dtype)

    # project out current eigenspace out of data matrix
    PB = B - B.dot(U_a.T).dot(U_a)
    # orthogonalise the previous projection using QR
    B_tilde = np.linalg.qr(PB.T)[0].T

    # form R matrix
    S_a = np.diag(s_a)
    R = np.hstack((np.vstack((f * S_a, B.dot(U_a.T))),
                   np.vstack((np.zeros((S_a.shape[0], B_tilde.shape[0]),
                                       dtype=B.dtype),
                              PB.dot(B_tilde.T)))))

    # compute SVD of R
    U_tilde, s_tilde, Vt_tilde = np.linalg.svd(R)

    # compute new eigenvalues
    l = s_tilde ** 2 / (n - 1)
    # keep only positive eigenvalues within tolerance
    l = l[l > eps]

    U = Vt_tilde.dot(np.vstack((U_a, B_tilde)))[:len(l), :]

    return U, l, m
