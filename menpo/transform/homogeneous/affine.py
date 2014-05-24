import abc
import copy
import numpy as np

from menpo.base import DX, DP
from .base import Homogeneous, HomogFamilyAlignment


class Affine(Homogeneous, DP, DX):
    r"""Base class for all n-dimensional affine transformations. Provides
    methods to break the transform down into it's constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transform with other affine transformations.

    Parameters
    ----------
    h_matrix : ``(n_dims + 1, n_dims + 1)`` `ndarray`
        The homogeneous matrix of the affine transformation.

    copy : `bool`, optional
        If ``False`` avoid copying ``h_matrix`` for performance.

    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """
    def __init__(self, h_matrix, copy=True, skip_checks=False):
        Homogeneous.__init__(self, h_matrix, copy=copy,
                             skip_checks=skip_checks)

    @classmethod
    def identity(cls, n_dims):
        return Affine(np.eye(n_dims + 1))

    @property
    def h_matrix(self):
        return self._h_matrix

    def _set_h_matrix(self, value, copy=True, skip_checks=False):
        r"""Updates the h_matrix, performing sanity checks.

        Parameters
        ----------
        value : ndarray
            The new homogeneous matrix to set

        copy : `bool`, optional
            If False do not copy the h_matrix. Useful for performance.

        skip_checks : `bool`, optional
            If True skip sanity checks on the matrix. Useful for performance.
        """
        if not skip_checks:
            shape = value.shape
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("You need to provide a square homogeneous "
                                 "matrix")
            if self.h_matrix is not None:
                # already have a matrix set! The update better be the same size
                if self.n_dims != shape[0] - 1:
                    raise ValueError("Trying to update the homogeneous "
                                     "matrix to a different dimension")
            if shape[0] - 1 not in [2, 3]:
                raise ValueError("Affine Transforms can only be 2D or 3D")
            if not (np.allclose(value[-1, :-1], 0) and
                    np.allclose(value[-1, -1], 1)):
                raise ValueError("Bottom row must be [0 0 0 1] or [0, 0, 1]")
        if copy:
            value = value.copy()
        self._h_matrix = value

    @property
    def linear_component(self):
        r"""The linear component of this affine transform.

        :type: ``(n_dims, n_dims)`` `ndarray`
        """
        return self.h_matrix[:-1, :-1]

    @property
    def translation_component(self):
        r"""The translation component of this affine transform.

        :type: ``(n_dims,)`` `ndarray`
        """
        return self.h_matrix[:-1, -1]

    def decompose(self):
        r"""Decompose this transform into discrete Affine Transforms.

        Useful for understanding the effect of a complex composite transform.

        Returns
        -------
        transforms : list of :map:`DiscreteAffine`
            Equivalent to this affine transform, such that::

                reduce(lambda x,y: x.chain(y), self.decompose()) == self
        """
        from .rotation import Rotation
        from .translation import Translation
        from .scale import Scale
        U, S, V = np.linalg.svd(self.linear_component)
        rotation_2 = Rotation(U)
        rotation_1 = Rotation(V)
        scale = Scale(S)
        translation = Translation(self.translation_component)
        return [rotation_1, scale, rotation_2, translation]

    def _transform_str(self):
        r"""
        A string representation explaining what this affine transform does.
        Has to be implemented by base classes.

        Returns
        -------
        str : string
            String representation of transform.
        """
        header = 'Affine decomposing into:'
        list_str = [t._transform_str() for t in self.decompose()]
        return header + reduce(lambda x, y: x + '\n' + '  ' + y, list_str, '  ')

    def _apply(self, x, **kwargs):
        r"""
        Applies this transform to a new set of vectors.

        Parameters
        ----------
        x : (N, D) ndarray
            Array to apply this transform to.


        Returns
        -------
        transformed_x : (N, D) ndarray
            The transformed array.
        """
        return np.dot(x, self.linear_component.T) + self.translation_component

    @property
    def n_parameters(self):
        r"""
        `n_dims * (n_dims + 1)` parameters - every element of the matrix bar
        the homogeneous part.

        :type: int

        Examples
        --------
        2D Affine: 6 parameters::

            [p1, p3, p5]
            [p2, p4, p6]

        3D Affine: 12 parameters::

            [p1, p4, p7, p10]
            [p2, p5, p8, p11]
            [p3, p6, p9, p12]
        """
        return self.n_dims * (self.n_dims + 1)

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. This does not
        include the homogeneous part of the warp. Note that it flattens using
        Fortran ordering, to stay consistent with Matlab.

        **2D**

        ========= ===========================================
        parameter definition
        ========= ===========================================
        p1        Affine parameter
        p2        Affine parameter
        p3        Affine parameter
        p4        Affine parameter
        p5        Translation in `x`
        p6        Translation in `y`
        ========= ===========================================

        3D and higher transformations follow a similar format to the 2D case.

        Returns
        -------
        params : ``(n_parameters,)`` `ndarray`
            The values that parametrise the transform.
        """
        params = self.h_matrix - np.eye(self.n_dims + 1)
        return params[:self.n_dims, :].ravel(order='F')

    def from_vector_inplace(self, p):
        r"""
        Updates this Affine in-place from the new parameters. See
        from_vector for details of the parameter format
        """
        h_matrix = None
        if p.shape[0] == 6:  # 2D affine
            h_matrix = np.eye(3)
            h_matrix[:2, :] += p.reshape((2, 3), order='F')
        elif p.shape[0] == 12:  # 3D affine
            h_matrix = np.eye(4)
            h_matrix[:3, :] += p.reshape((3, 4), order='F')
        else:
            ValueError("Only 2D (6 parameters) or 3D (12 parameters) "
                       "homogeneous matrices are supported.")
        self.set_h_matrix(h_matrix, copy=False, skip_checks=True)

    @property
    def composes_inplace_with(self):
        return Affine

    def _build_pseudoinverse(self):
        # Skip the checks as we know inverse of a homogeneous is a homogeneous
        return Affine(np.linalg.inv(self.h_matrix), copy=False,
                      skip_checks=True)

    def d_dp(self, points):
        r"""The first order derivative of this Affine transform wrt parameter
        changes evaluated at points.

        The Jacobian generated (for 2D) is of the form::

            x 0 y 0 1 0
            0 x 0 y 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + p1  p3      p5] [x]
                   [p2      1 + p4  p6] [y]
                                        [1]

        Parameters
        ----------
        points : (n_points, n_dims) ndarray
            The set of points to calculate the jacobian for.


        Returns
        -------
        (n_points, n_params, n_dims) ndarray
            The jacobian wrt parametrization

        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dims:
            raise ValueError(
                "Trying to sample jacobian in incorrect dimensions "
                "(transform is {0}D, sampling at {1}D)".format(
                    self.n_dims, points_n_dim))
        # prealloc the jacobian
        jac = np.zeros((n_points, self.n_parameters, self.n_dims))
        # a mask that we can apply at each iteration
        dim_mask = np.eye(self.n_dims, dtype=np.bool)

        for i, s in enumerate(
                range(0, self.n_dims * self.n_dims, self.n_dims)):
            # i is current axis
            # s is slicing offset
            # make a mask for a single points jacobian
            full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=bool)
            # fill the mask in for the ith axis
            full_mask[slice(s, s + self.n_dims)] = dim_mask
            # assign the ith axis points to this mask, broadcasting over all
            # points
            jac[:, full_mask] = points[:, i][..., None]
            # finally, just repeat the same but for the ones at the end
        full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=bool)
        full_mask[slice(s + self.n_dims, s + 2 * self.n_dims)] = dim_mask
        jac[:, full_mask] = 1
        return jac

    def d_dx(self, points):
        r"""
        The first order derivative of this Affine transform wrt spatial changes
        evaluated at points.

        The Jacobian for a given point (for 2D) is of the form::

            Jx = [(1 + a),     -b  ]
            Jy = [   b,     (1 + a)]
            J =  [Jx, Jy] = [[(1 + a), -b], [b, (1 + a)]]

        where a and b come from:

            W(x;p) = [1 + a   -b      tx] [x]
                     [b       1 + a   ty] [y]
                                          [1]
        Hence it is simply the linear component of the transform.

        Parameters
        ----------

        points: ndarray shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------

        d_dx: (1, n_dims, n_dims) ndarray
            The jacobian wrt spatial changes.

            d_dx[0, j, k] is the scalar differential change that the
            j'th dimension of the i'th point experiences due to a first order
            change in the k'th dimension.

            Note that because the jacobian is constant across space the first
            axis is length 1 to allow for broadcasting.

        """
        return self.linear_component[None, ...]


class AlignmentAffine(HomogFamilyAlignment, Affine):
    r"""
    Constructs an Affine by finding the optimal affine transform to align
    source to target.

    Parameters
    ----------
    source : :map:`PointCloud`
        The source pointcloud instance used in the alignment

    target : :map:`PointCloud`
        The target pointcloud instance used in the alignment

    Notes
    -----
    We want to find the optimal transform M which satisfies

        M a = b

    where `a` and `b` are the source and target homogeneous vectors
    respectively.

       (M a)' = b'
       a' M' = b'
       a a' M' = a b'

       `a a'` is of shape `(n_dim + 1, n_dim + 1)` and so can be inverted
       to solve for M.

       This approach is the analytical linear least squares solution to
       the problem at hand. It will have a solution as long as `(a a')`
       is non-singular, which generally means at least 2 corresponding
       points are required.
    """
    def __init__(self, source, target):
        # first, initialize the alignment
        HomogFamilyAlignment.__init__(self, source, target)
        # now, the Affine
        optimal_h = self._build_alignment_h_matrix(source, target)
        Affine.__init__(self, optimal_h, copy=False, skip_checks=True)

    @staticmethod
    def _build_alignment_h_matrix(source, target):
        r"""
        Returns the optimal alignment of source to target.
        """
        a = source.h_points
        b = target.h_points
        return np.linalg.solve(np.dot(a, a.T), np.dot(a, b.T)).T

    def set_h_matrix(self, value, copy=True, skip_checks=False):
        r"""
        Updates ``h_matrix``, optionally performing sanity checks.

        .. note::

            Updating the ``h_matrix`` on an :map:`AlignmentAffine`
            triggers a sync of the target.

        Note that it won't always be possible to manually specify the
        ``h_matrix`` through this method, specifically if changing the
        ``h_matrix`` could change the nature of the transform. See
        :attr:`h_matrix_is_mutable` for how you can discover if the
        ``h_matrix`` is allowed to be set for a given class.

        Parameters
        ----------
        value : ndarray
            The new homogeneous matrix to set
        copy : `bool`, optional
            If False do not copy the h_matrix. Useful for performance.
        skip_checks : `bool`, optional
            If True skip checking. Useful for performance.

        Raises
        ------
        NotImplementedError
            If :attr:`h_matrix_is_mutable` returns ``False``.


        Parameters
        ----------

        value : ndarray
            The new homogeneous matrix to set

        copy : bool, optional
            If False do not copy the h_matrix. Useful for performance.

        skip_checks : bool, optional
            If True skip verification for performance.
        """
        Affine.set_h_matrix(self, value, copy=copy, skip_checks=skip_checks)
        # now update the state
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        optimal_h = self._build_alignment_h_matrix(self.source, self.target)
        # Use the pure Affine setter (so we don't get syncing)
        # We know the resulting affine is correct so skip the checks
        Affine.set_h_matrix(self, optimal_h, copy=False, skip_checks=True)

    def as_non_alignment(self):
        r"""Returns a copy of this affine without it's alignment nature.

        Returns
        -------
        transform : :map:`Affine`
            A version of this affine with the same transform behavior but
            without the alignment logic.
        """
        return Affine(self.h_matrix, skip_checks=True)


class DiscreteAffine(object):
    r"""
    A discrete Affine transform operation (such as a :meth:`Scale`,
    :class:`Translation` or :meth:`Rotation`). Has to be able to invertable.
    Make sure you inherit from :class:`DiscreteAffine` first,
    for optimal `decompose()` behavior.
    """

    __metaclass__ = abc.ABCMeta

    def decompose(self):
        r"""
        A :class:`DiscreteAffine` is already maximally decomposed -
        return a copy of self in a list.

        Returns
        -------
        transform : :class:`DiscreteAffine`
            Deep copy of `self`.
        """
        return [self.copy()]
