import numpy as np
from scipy.spatial.distance import cdist

from menpo.base import DL

from .base import Transform


class RadialBasisFunction(Transform, DL):
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

    def d_dl(self, points):
        """
        Apply the derivative of the basis function wrt the centres and the
        points given by `points`.

        .. note::

            Let `points` be `x`, then

            ..math::

                2 (x - c)^T (\log{r^2_{x, l}} + 1) ===
                2 (x - c)^T (2 \log{r_{x, l}} + 1)

            where:

            :math:`r_{x, l} = \lVert x - c \rVert`


        Parameters
        ----------
        x : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        d_dl : (n_points, n_centres, n_dims) ndarray
            The jacobian tensor representing the first order derivative
            of the radius from each centre wrt the centre's position, evaluated
            at each point.
        """
        euclidean_distance = cdist(points, self.c)
        component_distances = points[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        d_dl = (2 * component_distances *
                (2 * np.log(euclidean_distance[..., None]) + 1))
        return d_dl


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

    def d_dl(self, points):
        """
        The derivative of the basis function wrt the coordinate system
        evaluated at `points`.

        :math:`(x - c)^T (1 + 2 \log{r_{x, l}})`.

        .. note::

            :math:`r_{x, l} = \lVert x - c \rVert`

        Parameters
        ----------
        points : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        d_dl : (n_points, n_centres, n_dims) ndarray
            The jacobian tensor representing the first order partial derivative
            of each points wrt the centres
        """
        euclidean_distance = cdist(points, self.c)
        component_distances = points[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        d_dl = (component_distances *
                (1 + 2 * np.log(euclidean_distance[..., None])))
        return d_dl
