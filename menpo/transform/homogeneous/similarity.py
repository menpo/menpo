import numpy as np

from .base import HomogFamilyAlignment
from .affine import Affine


class Similarity(Affine):
    r"""
    Specialist version of an :map:`Affine` that is guaranteed to be
    a Similarity transform.

    Parameters
    ----------
    h_matrix : (D + 1, D + 1) ndarray
        The homogeneous matrix of the similarity transform.

    """

    def __init__(self, h_matrix, copy=True, skip_checks=False):
        Affine.__init__(self, h_matrix, copy=copy, skip_checks=skip_checks)

    def _transform_str(self):
        r"""
        A string representation explaining what this similarity transform does.

        Returns
        -------
        str : string
            String representation of transform.

        """
        header = 'Similarity decomposing into:'
        list_str = [t._transform_str() for t in self.decompose()]
        return header + reduce(lambda x, y: x + '\n' + '  ' + y, list_str, '  ')

    @property
    def h_matrix_is_mutable(self):
        return False

    @classmethod
    def identity(cls, n_dims):
        return Similarity(np.eye(n_dims + 1))

    @property
    def n_parameters(self):
        r"""
        2D Similarity: 4 parameters::

            [(1 + a), -b,      tx]
            [b,       (1 + a), ty]

        3D Similarity: Currently not supported

        Returns
        -------
        int

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.

        """
        if self.n_dims == 2:
            return 4
        elif self.n_dims == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise ValueError("Only 2D and 3D Similarity transforms "
                             "are currently supported.")

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order `[a, b, tx, ty]`, given that
        `a = k cos(theta) - 1` and `b = k sin(theta)` where `k` is a
        uniform scale and `theta` is a clockwise rotation in radians.

        **2D**

        ========= ===========================================
        parameter definition
        ========= ===========================================
        a         `a = k cos(theta) - 1`
        b         `b = k sin(theta)`
        tx        Translation in `x`
        ty        Translation in `y`
        ========= ===========================================

        .. note::

            Only 2D transforms are currently supported.

        Returns
        -------
        params : (P,) ndarray
            The values that parameterise the transform.

        Raises
        ------
        DimensionalityError, NotImplementedError
            If the transform is not 2D

        """
        n_dims = self.n_dims
        if n_dims == 2:
            params = self.h_matrix - np.eye(n_dims + 1)
            # Pick off a, b, tx, ty
            params = params[:n_dims, :].ravel(order='F')
            # Pick out a, b, tx, ty
            return params[[0, 1, 4, 5]]
        elif n_dims == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise ValueError("Only 2D and 3D Similarity transforms "
                             "are currently supported.")

    def from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Similarity: 4 parameters::

            [a, b, tx, ty]

        Parameters
        ----------
        p : (P,) ndarray
            The array of parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.

        """
        if p.shape[0] == 4:
            homog = np.eye(3)
            homog[0, 0] += p[0]
            homog[1, 1] += p[0]
            homog[0, 1] = -p[1]
            homog[1, 0] = p[1]
            homog[:2, 2] = p[2:]
            self._set_h_matrix(homog, skip_checks=True, copy=False)
        elif p.shape[0] == 7:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise ValueError("Only 2D and 3D Similarity transforms "
                             "are currently supported.")

    def _build_pseudoinverse(self):
        return Similarity(np.linalg.inv(self.h_matrix), copy=False,
                          skip_checks=True)

    def d_dp(self, points):
        r"""
        Computes the Jacobian of the transform w.r.t the parameters.

        The Jacobian generated (for 2D) is of the form::

            x -y 1 0
            y  x 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + a  -b   ] [x] + tx
                   [b      1 + a] [y] + ty

        Parameters
        ----------
        points : (n_points, n_dims) ndarray
            The set of points to calculate the jacobian for.

        Returns
        -------
        (n_points, n_params, n_dims) ndarray
            The jacobian wrt parametrization

        Raises
        ------
        DimensionalityError
            `points.n_dims != self.n_dims` or transform is not 2D

        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dims:
            raise ValueError('Trying to sample jacobian in incorrect '
                             'dimensions (transform is {0}D, sampling '
                             'at {1}D)'.format(self.n_dims, points_n_dim))
        elif self.n_dims != 2:
            # TODO: implement 3D Jacobian
            raise ValueError("Only the Jacobian of a 2D similarity "
                             "transform is currently supported.")

        # prealloc the jacobian
        jac = np.zeros((n_points, self.n_parameters, self.n_dims))
        ones = np.ones_like(points)

        # Build a mask and apply it to the points to build the jacobian
        # Do this for each parameter - [a, b, tx, ty] respectively
        self._apply_jacobian_mask(jac, np.array([1, 1]), 0, points)
        self._apply_jacobian_mask(jac, np.array([-1, 1]), 1, points[:, ::-1])
        self._apply_jacobian_mask(jac, np.array([1, 0]), 2, ones)
        self._apply_jacobian_mask(jac, np.array([0, 1]), 3, ones)

        return jac

    def _apply_jacobian_mask(self, jac, param_mask, row_index, points):
        # make a mask for a single points jacobian
        full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=np.bool)
        # fill the mask in for the ith axis
        full_mask[row_index] = [True, True]
        # assign the ith axis points to this mask, broadcasting over all
        # points
        jac[:, full_mask] = points * param_mask


class AlignmentSimilarity(HomogFamilyAlignment, Similarity):
    """
    Infers the similarity transform relating two vectors with the same
    dimensionality. This is simply the procrustes alignment of the
    source to the target.

    Parameters
    ----------

    source : :map:`PointCloud`
        The source pointcloud instance used in the alignment

    target : :map:`PointCloud`
        The target pointcloud instance used in the alignment

    rotation: boolean, optional
        If False, the rotation component of the similarity transform is not
        inferred.

        Default: True

    """
    def __init__(self, source, target, rotation=True):
        HomogFamilyAlignment.__init__(self, source, target)
        x = self._procrustes_alignment(source, target, rotation=rotation)
        Similarity.__init__(self, x.h_matrix, copy=False, skip_checks=True)

    @staticmethod
    def _procrustes_alignment(source, target, rotation=True):
        r"""
        Returns the similarity transform that aligns the source to the target.

        """
        from .rotation import Rotation, optimal_rotation_matrix
        from .translation import Translation
        from .scale import UniformScale
        target_translation = Translation(-target.centre)
        centred_target = target_translation.apply(target)
        # now translate the source to the origin
        translation = Translation(-source.centre)
        # apply the translation to the source
        aligned_source = translation.apply(source)
        scale = UniformScale(target.norm() / source.norm(), source.n_dims)
        scaled_aligned_source = scale.apply(aligned_source)
        # compute the target's inverse translation
        inv_target_translation = target_translation.pseudoinverse
        if rotation:
            rotation = Rotation(optimal_rotation_matrix(scaled_aligned_source,
                                                        centred_target))
            return translation.compose_before(scale).compose_before(
                rotation).compose_before(inv_target_translation)
        else:
            return translation.compose_before(scale).compose_before(
                inv_target_translation)

    def _sync_state_from_target(self):
        similarity = self._procrustes_alignment(self.source, self.target)
        self._set_h_matrix(similarity.h_matrix, copy=False, skip_checks=True)

    def as_non_alignment(self):
        r"""Returns a copy of this similarity without it's alignment nature.

        Returns
        -------
        transform : :map:`Similarity`
            A version of this similarity with the same transform behavior but
            without the alignment logic.
        """
        return Similarity(self.h_matrix, skip_checks=True)

    def from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Similarity: 4 parameters::

            [a, b, tx, ty]

        Parameters
        ----------
        p : (P,) ndarray
            The array of parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.

        """
        Similarity.from_vector_inplace(self, p)
        self._sync_target_from_state()
