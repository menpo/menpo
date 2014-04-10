import abc
import copy
import numpy as np

from menpo.exception import DimensionalityError

from .base import Homogeneous, HomogFamilyAlignment


class Affine(Homogeneous):
    r"""
    The base class for all n-dimensional affine transformations. Provides
    methods to break the transform down into it's constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transform with other affine transformations.

    Parameters
    ----------
    h_matrix : (n_dims + 1, n_dims + 1) ndarray
        The homogeneous matrix of the affine transformation.
    """
    def __init__(self, h_matrix):
        Homogeneous.__init__(self, h_matrix)
        # Affine is a little more constrained (only 2D or 3D supported)
        # so run our verification
        Affine.set_h_matrix(self, h_matrix)

    @classmethod
    def identity(cls, n_dims):
        return Affine(np.eye(n_dims + 1))

    @property
    def h_matrix(self):
        return self._h_matrix

    def set_h_matrix(self, value):
        r"""
        Updates the h_matrix, performing sanity checks.

        The Affine h_matrix is limited in what values are allowed. Account
        for them here.
        """
        shape = value.shape
        if len(shape) != 2 and shape[0] != shape[1]:
            raise ValueError("You need to provide a square homogeneous matrix")
        if self.h_matrix is not None:
            # already have a matrix set! The update better be the same size
            if self.n_dims != shape[0] - 1:
                raise DimensionalityError("Trying to update the homogeneous "
                                          "matrix to a different dimension")
        if shape[0] - 1 not in [2, 3]:
            raise DimensionalityError("Affine Transforms can only be 2D or 3D")
        if not (np.allclose(value[-1, :-1], 0) and
                np.allclose(value[-1, -1], 1)):
            raise ValueError("Bottom row must be [0 0 0 1] or [0, 0, 1]")
        self._h_matrix = value.copy()

    @property
    def linear_component(self):
        r"""
        Returns just the linear transform component of this affine
        transform.

        :type: (D, D) ndarray
        """
        return self.h_matrix[:-1, :-1]

    @property
    def translation_component(self):
        r"""
        Returns just the translation component.

        :type: (D,) ndarray
        """
        return self.h_matrix[:-1, -1]

    def decompose(self):
        r"""
        Uses an SVD to decompose this transform into discrete Affine
        Transforms.

        Returns
        -------
        transforms: list of :class`DiscreteAffine` that
            Equivalent to this affine transform, such that:

            ``reduce(lambda x,y: x.chain(y), self.decompose()) == self``
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

    def __eq__(self, other):
        return np.allclose(self.h_matrix, other.h_matrix)

    def __str__(self):
        rep = repr(self) + '\n'
        rep += str(self.h_matrix) + '\n'
        rep += self._transform_str()
        return rep

    def _transform_str(self):
        r"""
        A string representation explaining what this affine transform does.
        Has to be implemented by base classes.

        Returns
        -------
        str : string
            String representation of transform.
        """
        list_str = [t._transform_str() for t in self.decompose()]
        return reduce(lambda x, y: x + '\n' + y, list_str)

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
        ``n_dims * (n_dims + 1)`` parameters - every element of the matrix bar
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

    def as_vector(self):
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
        p5        Translation in ``x``
        p6        Translation in ``y``
        ========= ===========================================

        3D and higher transformations follow a similar format to the 2D case.

        Returns
        -------
        params : (P,) ndarray
            The values that paramaterise the transform.
        """
        params = self.h_matrix - np.eye(self.n_dims + 1)
        return params[:self.n_dims, :].flatten(order='F')

    def from_vector_inplace(self, p):
        r"""
        Updates this Affine in-place from the new parameters. See
        from_vector for details of the parameter format
        """
        h_matrix = None
        if p.shape[0] is 6:  # 2D affine
            h_matrix = np.eye(3)
            h_matrix[:2, :] += p.reshape((2, 3), order='F')
        elif p.shape[0] is 12:  # 3D affine
            h_matrix = np.eye(4)
            h_matrix[:3, :] += p.reshape((3, 4), order='F')
        else:
            ValueError("Only 2D (6 parameters) or 3D (12 parameters) "
                       "homogeneous matrices are supported.")
        self.set_h_matrix(h_matrix)

    @property
    def composes_inplace_with(self):
        return Affine

    def _build_pseudoinverse(self):
        return Affine(np.linalg.inv(self.h_matrix))

    def jacobian(self, points):
        r"""
        Computes the Jacobian of the transform w.r.t the parameters. This is
        constant for affine transforms.

        The Jacobian generated (for 2D) is of the form::

            x 0 y 0 1 0
            0 x 0 y 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + p1  p3      p5] [x]
                   [p2      1 + p4  p6] [y]
                                        [1]

        Parameters
        ----------
        points : (N, D) ndarray
            The set of points to calculate the jacobian for.

        Returns
        -------
        dW_dp : (N, P, D) ndarray
            A (``n_points``, ``n_params``, ``n_dims``) array representing
            the Jacobian of the transform.
        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dims:
            raise DimensionalityError(
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

    def jacobian_points(self, points):
        r"""
        Computes the Jacobian of the transform wrt the points to which
        the transform is applied to. This is constant for affine transforms.

        The Jacobian for a given point (for 2D) is of the form::

            Jx = [(1 + a),     -b  ]
            Jy = [   b,     (1 + a)]
            J =  [Jx, Jy] = [[(1 + a), -b], [b, (1 + a)]]

        where a and b come from:

            W(x;p) = [1 + a   -b      tx] [x]
                     [b       1 + a   ty] [y]
                                          [1]

        Returns
        -------
        dW/dx: dW/dx: (N, D, D) ndarray
            The Jacobian of the transform wrt the points to which the
            transform is applied to.
        """
        return self.linear_component[None, ...]


class AlignmentAffine(Affine, HomogFamilyAlignment):
    r"""
    Constructs an Affine by finding the optimal affine transform to align
    source to target.

    Parameters
    ----------

    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment

    target: :class:`menpo.shape.PointCloud`
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
        Affine.__init__(self, optimal_h)

    @staticmethod
    def _build_alignment_h_matrix(source, target):
        r"""
        Returns the optimal alignment of source to target.
        """
        a = source.h_points
        b = target.h_points
        return np.linalg.solve(np.dot(a, a.T), np.dot(a, b.T)).T

    def set_h_matrix(self, value):
        r"""
        Upon updating the h_matrix we must resync the target.
        """
        Affine.set_h_matrix(self, value)
        # now update the state
        self._sync_target_from_state()

    def from_vector_inplace(self, p):
        r"""
        Updates this Affine in-place from the new parameters. See
        from_vector for details of the parameter format.
        """
        Affine.from_vector_inplace(self, p)
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        optimal_h = self._build_alignment_h_matrix(self.source, self.target)
        # Use the pure Affine setter (so we don't get syncing)
        Affine.set_h_matrix(self, optimal_h)

    def copy_without_alignment(self):
        return Affine(self.h_matrix.copy())


class DiscreteAffine(object):
    r"""
    A discrete Affine transform operation (such as a :meth:`Scale`,
    :class:`Translation` or :meth:`Rotation`). Has to be able to invertable.
    Make sure you inherit from :class:`DiscreteAffine` first,
    for optimal ``decompose()`` behavior.
    """

    __metaclass__ = abc.ABCMeta

    def decompose(self):
        r"""
        A :class:`DiscreteAffine` is already maximally decomposed -
        return a copy of self in a list.

        Returns
        -------
        transform : :class:`DiscreteAffine`
            Deep copy of ``self``.
        """
        return [copy.deepcopy(self)]
