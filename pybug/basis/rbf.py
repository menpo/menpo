import abc
import numpy as np


class BasisFunction(object):
    r"""
    An abstract base class for Basis functions. In the case, radial basis
    functions. They provide two methods, :meth:`phi`, which calculates the
    basis itself, and :meth:`derivative`, which calculates the derivative
    of the basis.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def phi(self, r):
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
    def derivative(self, r):
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

    def __init__(self):
        super(R2LogR2, self).__init__()

    def phi(self, r):
        """
        Apply the basis function.

        .. note::

            :math:`r^2 \log{r^2} === r^2 2 \log{r}`

        Parameters
        ----------
        r : (N, N) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        U : (N, N) ndarray
            The basis function applied to each distance.
        """
        mask = r == 0
        r[mask] = 1
        U = r ** 2 * (2 * np.log(r))
        # reset singularities to 0
        U[mask] = 0
        return U

    def derivative(self, r):
        """
        Apply the derivative of the basis function.

        .. note::

            :math:`2 r (\log{r^2} + 1) === 2 r (2 \log{r} + 1)`


        Parameters
        ----------
        r : (N, N) ndarray
            Square distance matrix of pairwise euclidean distances.

        Returns
        -------
        dUdr : (N, N) ndarray
            The derivative of the basis function applied to each distance.
        """
        mask = r == 0
        r[mask] = 1
        dUdr = 2 * r * (2 * np.log(r) + 1)
        # reset singularities to 0
        dUdr[mask] = 0
        return dUdr


class R2LogR(BasisFunction):
    r"""
    Calculates the :math:`r^2 \log{r}` basis function.

    The derivative of this function is :math:`r (1 + 2 \log{r})`.
    """

    def __init__(self):
        super(R2LogR, self).__init__()

    def phi(self, r):
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
        mask = r == 0
        r[mask] = 1
        U = r ** 2 * np.log(r)
        # reset singularities to 0
        U[mask] = 0
        return U

    def derivative(self, r):
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
        mask = r == 0
        r[mask] = 1
        dUdr = r * (1 + 2 * np.log(r))
        # reset singularities to 0
        dUdr[mask] = 0
        return dUdr