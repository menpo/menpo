import abc
import numpy as np
from scipy.spatial.distance import cdist


class BasisFunction(object):
    r"""
    An abstract base class for Basis functions. In the case, radial basis
    functions. They provide two methods, :meth:`phi`, which calculates the
    basis itself, and :meth:`derivative`, which calculates the derivative
    of the basis.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, c):
        self.c = c

    @abc.abstractmethod
    def apply(self, x):
        r"""
        Calculate the basis function on the given residuals. These are expected
        to be a square distance matrix representing the euclidean distance
        between the points in the space.

        .. note::

            Divisions by zero are avoided and any zero residuals remain zero.

        Parameters
        ----------
        r : (N, N) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        U : (N, N) ndarray
            The basis function applied to each distance.
        """
        pass

    @abc.abstractmethod
    def jacobian(self, x):
        r"""
        Calculate the derivative of the basis function on the given residuals.
        These are expected to be a square distance matrix representing the
        euclidean distance between the points in the space.

        .. note::

            Divisions by zero are avoided and any zero residuals remain zero.

        Parameters
        ----------
        r : (N, N) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        dUdr : (N, N) ndarray
            The derivative of the basis function applied to each distance.
        """
        pass


class R2LogR2(BasisFunction):
    r"""
    Calculates the :math:`r^2 \log{r^2}` basis function.

    The derivative of this function is :math:`2 r (\log{r^2} + 1)`.
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
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        U : (N, N) ndarray
            The basis function applied to each distance.
        """
        euclidean_distance = cdist(x, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            U = (euclidean_distance ** 2 *
                 (2 * np.log(euclidean_distance)))
            # reset singularities to 0
        U[mask] = 0
        return U

    def jacobian(self, x):
        """
        Apply the derivative of the basis function.

        .. note::

            :math:`2 r (\log{r^2} + 1) === 2 r (2 \log{r} + 1)`


        Parameters
        ----------
        x : (N, D) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        dUdr : (N, N) ndarray
            The derivative of the basis function applied to each distance.
        """
        euclidean_distance = cdist(x, self.c)
        component_distances = x[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        dUdr = (2 * component_distances *
                (2 * np.log(euclidean_distance[..., None]) + 1))
        return dUdr


class R2LogR(BasisFunction):
    r"""
    Calculates the :math:`r^2 \log{r}` basis function.

    The derivative of this function is :math:`r (1 + 2 \log{r})`.
    """

    def __init__(self, c):
        super(R2LogR, self).__init__(c)

    def apply(self, x):
        """
        Apply the basis function :math:`r^2 \log{r}`.

        Parameters
        ----------
        r : (N, N) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        U : (N, N) ndarray
            The basis function applied to each distance.
        """
        euclidean_distance = cdist(x, self.c)
        mask = euclidean_distance == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            U = euclidean_distance ** 2 * np.log(euclidean_distance)
        # reset singularities to 0
        U[mask] = 0
        return U

    def jacobian(self, x):
        """
        Apply the derivative of the basis function :math:`r (1 + 2 \log{r})`.

        Parameters
        ----------
        r : (N, N) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        dUdr : (N, N) ndarray
            The derivative of the basis function applied to each distance.
        """
        euclidean_distance = cdist(x, self.c)
        component_distances = x[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        dUdr = (component_distances *
                (1 + 2 * np.log(euclidean_distance[..., None])))
        return dUdr