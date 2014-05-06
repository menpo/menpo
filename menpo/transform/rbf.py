import abc
import numpy as np
from scipy.spatial.distance import cdist

from menpo.base import DX

from .base import Transform

# TODO RBF is really implementing d_dl (with dim=1 for function)
class RadialBasisFunction(Transform, DX):
    r"""
    Radial Basis Functions are a class of transform that is used by
    TPS. They have to be able to take their own spatial derivative for TPS to
    be able to take it's own total derivative.

    Parameters
    ----------
    c : (L, D) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    def __init__(self, c):
        self.c = c


class R2LogR2RBF(RadialBasisFunction):
    r"""
    The :math:`r^2 \log{r^2}` basis function.

    The derivative of this function is :math:`2 r (\log{r^2} + 1)`.

    .. note::

        :math:`r = \lVert x - c \rVert`

    Parameters
    ----------
    c : (L, D) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    def __init__(self, c):
        super(R2LogR2RBF, self).__init__(c)

    def _apply(self, x):
        """
        Apply the basis function.

        .. note::

            :math:`r^2 \log{r^2} === r^2 2 \log{r}`

        Parameters
        ----------
        x : (N, D) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        u : (N, L) ndarray
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

    def d_dx(self, x):
        """
        Apply the derivative of the basis function wrt the coordinate system.
        This is applied over each dimension of the input vector, `x`.

        .. note::

            ..math::

                2 (x - c)^T (\log{r^2_{x, l}} + 1) ===
                2 (x - c)^T (2 \log{r_{x, l}} + 1)

            where:

            :math:`r_{x, l} = \lVert x - c \rVert``


        Parameters
        ----------
        x : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        d_dx : (n_points, n_landmarks, n_dims) ndarray
            The jacobian tensor representing the first order partial derivative
            of each point wrt the coordinate system
        """
        euclidean_distance = cdist(x, self.c)
        component_distances = x[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        dudx = (2 * component_distances *
                (2 * np.log(euclidean_distance[..., None]) + 1))
        return dudx


class R2LogRRBF(RadialBasisFunction):
    r"""
    Calculates the :math:`r^2 \log{r}` basis function.

    The derivative of this function is :math:`r (1 + 2 \log{r})`.

    .. note::

        :math:`r = \lVert x - c \rVert`

    Parameters
    ----------
    c : (L, D) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    def __init__(self, c):
        super(R2LogRRBF, self).__init__(c)

    def _apply(self, x):
        """
        Apply the basis function :math:`r^2 \log{r}`.

        Parameters
        ----------
        x : (N, D) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        u : (N, L) ndarray
            The basis function applied to each distance,
            :math:`\lVert x - c \rVert`.
        """
        euclidean_distance = cdist(x, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            u = euclidean_distance ** 2 * np.log(euclidean_distance)
        # reset singularities to 0
        u[mask] = 0
        return u

    def d_dx(self, x):
        """
        The derivative of the basis function wrt the coordinate system
        evaluated at `x`.

        :math:`(x - c)^T (1 + 2 \log{r_{x, l}})`.

        .. note::

            :math:`r_{x, l} = \lVert x - c \rVert``

        Parameters
        ----------
        x : (N, D) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        dudx : (N, L, D) ndarray
            The jacobian tensor representing the first order partial derivative
            of each points wrt the coordinate system
        """
        euclidean_distance = cdist(x, self.c)
        component_distances = x[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        dudx = (component_distances *
                (1 + 2 * np.log(euclidean_distance[..., None])))
        return dudx
