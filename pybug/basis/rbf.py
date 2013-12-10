import abc
import numpy as np
from scipy.spatial.distance import cdist


class BasisFunction(object):
    r"""
    An abstract base class for Basis functions. In the case, radial basis
    functions. They provide two methods, :meth:`apply`, which calculates the
    basis itself, and :meth:`jacobian_points`, which calculates the derivative
    of the basis wrt the coordinate system.

    Parameters
    ----------
    c : (L, D) ndarray
        The set of centers that make the basis. Usually represents a set of
        source landmarks.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, c):
        self.c = c

    @abc.abstractmethod
    def apply(self, x):
        r"""
        Calculate the basis function on the given residuals. The input is the
        set of points the basis should be calculated for. The euclidean
        distance between ``x`` and the centers, ``c``, will be used as the
        residual.

        .. note::

            Divisions by zero are avoided and any zero residuals remain zero.

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
        pass

    @abc.abstractmethod
    def jacobian_points(self, x):
        r"""
        Calculate the derivative of the basis function wrt the
        coordinate system.

        .. note::

            Divisions by zero are avoided and any zero residuals remain zero.

        Parameters
        ----------
        x : (N, D) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        dudx : (N, L, D) ndarray
            Tensor representing the first order partial derivative
            of each points with respect to the centers, over each dimension.
        """
        pass


class R2LogR2(BasisFunction):
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
        super(R2LogR2, self).__init__(c)

    def apply(self, x):
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

    def jacobian_points(self, x):
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
        x : (N, D) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        dudx : (N, L, D) ndarray
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


class R2LogR(BasisFunction):
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
        super(R2LogR, self).__init__(c)

    def apply(self, x):
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

    def jacobian_points(self, x):
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
