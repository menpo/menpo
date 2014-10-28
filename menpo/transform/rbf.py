import numpy as np
from scipy.spatial.distance import cdist
from .base import Transform


class RadialBasisFunction(Transform):
    r"""
    Radial Basis Functions are a class of transform that is used by
    TPS. They have to be able to take their own radial derivative for TPS to
    be able to take it's own total derivative.

    Parameters
    ----------
    c : (n_centres, n_dims) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    def __init__(self, c):
        self.c = c

    @property
    def n_centres(self):
        return self.c.shape[0]

    @property
    def n_dims(self):
        r"""
        The RBF can only be applied on points with the same dimensionality as
        the centres.
        """
        return self.c.shape[1]

    @property
    def n_dims_output(self):
        r"""
        The result of the transform has a dimension (weight) for every centre
        """
        return self.n_centres


class R2LogR2RBF(RadialBasisFunction):
    r"""
    The :math:`r^2 \log{r^2}` basis function.

    The derivative of this function is :math:`2 r (\log{r^2} + 1)`.

    .. note::

        :math:`r = \lVert x - c \rVert`

    Parameters
    ----------
    c : (n_centres, n_dims) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    def __init__(self, c):
        super(R2LogR2RBF, self).__init__(c)

    def _apply(self, x, **kwargs):
        """
        Apply the basis function.

        .. note::

            :math:`r^2 \log{r^2} === r^2 2 \log{r}`

        Parameters
        ----------
        x : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        u : (n_points, n_centres) ndarray
            The basis function applied to each distance,
            :math:`\lVert x - c \rVert`.
        """
        euclidean_distance = cdist(x, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (euclidean_distance ** 2 *
                 (2 * np.log(euclidean_distance)))
        # reset singularities to 0
        u[mask] = 0
        return u


class R2LogRRBF(RadialBasisFunction):
    r"""
    Calculates the :math:`r^2 \log{r}` basis function.

    The derivative of this function is :math:`r (1 + 2 \log{r})`.

    .. note::

        :math:`r = \lVert x - c \rVert`

    Parameters
    ----------
    c : (n_centres, n_dims) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    def __init__(self, c):
        super(R2LogRRBF, self).__init__(c)

    def _apply(self, points, **kwargs):
        """
        Apply the basis function :math:`r^2 \log{r}`.

        Parameters
        ----------
        points : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        u : (n_points, n_centres) ndarray
            The basis function applied to each distance,
            :math:`\lVert points - c \rVert`.
        """
        euclidean_distance = cdist(points, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            u = euclidean_distance ** 2 * np.log(euclidean_distance)
        # reset singularities to 0
        u[mask] = 0
        return u
