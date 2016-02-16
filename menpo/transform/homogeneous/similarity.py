import numpy as np

from .base import HomogFamilyAlignment
from .affine import Affine
from functools import reduce


class Similarity(Affine):
    r"""
    Specialist version of an :map:`Affine` that is guaranteed to be a
    Similarity transform.

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
        Affine.__init__(self, h_matrix, copy=copy, skip_checks=skip_checks)

    @classmethod
    def init_identity(cls, n_dims):
        r"""
        Creates an identity transform.

        Parameters
        ----------
        n_dims : `int`
            The number of dimensions.

        Returns
        -------
        identity : :class:`Similarity`
            The identity matrix transform.
        """
        return cls(np.eye(n_dims + 1), copy=False, skip_checks=True)

    def _transform_str(self):
        r"""
        A string representation explaining what this similarity transform does.

        Returns
        -------
        string : `str`
            String representation of transform.
        """
        header = 'Similarity decomposing into:'
        list_str = [t._transform_str() for t in self.decompose()]
        return header + reduce(lambda x, y: x + '\n' + '  ' + y, list_str, '  ')

    @property
    def n_parameters(self):
        r"""
        2D Similarity: 4 parameters::

            [(1 + a), -b,      tx]
            [b,       (1 + a), ty]

        3D Similarity: Currently not supported

        Returns
        -------
        n_parameters : `int`
            The transform parameters

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
        are output in the order ``[a, b, tx, ty]``, given that
        ``a = k cos(theta) - 1`` and ``b = k sin(theta)`` where ``k`` is a
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
        params : ``(P,)`` `ndarray`
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

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Similarity: 4 parameters ::

            [a, b, tx, ty]

        Parameters
        ----------
        p : ``(P,)`` `ndarray`
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


class AlignmentSimilarity(HomogFamilyAlignment, Similarity):
    """
    Infers the similarity transform relating two vectors with the same
    dimensionality. This is simply the procrustes alignment of the
    `source` to the `target`.

    Parameters
    ----------
    source : :map:`PointCloud`
        The source pointcloud instance used in the alignment
    target : :map:`PointCloud`
        The target pointcloud instance used in the alignment
    rotation: `bool`, optional
        If ``False``, the rotation component of the similarity transform is not
        inferred.
    allow_mirror : `bool`, optional
        If ``True``, the Kabsch algorithm check is not performed, and mirroring
        of the Rotation matrix is permitted.
    """
    def __init__(self, source, target, rotation=True, allow_mirror=False):
        HomogFamilyAlignment.__init__(self, source, target)
        x = procrustes_alignment(source, target, rotation=rotation,
                                 allow_mirror=allow_mirror)
        Similarity.__init__(self, x.h_matrix, copy=False, skip_checks=True)
        self.allow_mirror = allow_mirror

    def _sync_state_from_target(self):
        similarity = procrustes_alignment(self.source, self.target,
                                          allow_mirror=self.allow_mirror)
        self._set_h_matrix(similarity.h_matrix, copy=False, skip_checks=True)

    def as_non_alignment(self):
        r"""
        Returns a copy of this similarity without it's alignment nature.

        Returns
        -------
        transform : :map:`Similarity`
            A version of this similarity with the same transform behavior but
            without the alignment logic.
        """
        return Similarity(self.h_matrix, skip_checks=True)

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Similarity: 4 parameters ::

            [a, b, tx, ty]

        Parameters
        ----------
        p : ``(P,)`` `ndarray`
            The array of parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.
        """
        Similarity._from_vector_inplace(self, p)
        self._sync_target_from_state()


def procrustes_alignment(source, target, rotation=True, allow_mirror=False):
    r"""
    Returns the similarity transform that aligns the `source` to the `target`.

    Parameters
    ----------
    source : :map:`PointCloud`
        The source pointcloud
    target : :map:`PointCloud`
        The target pointcloud
    rotation : `bool`, optional
        If ``True``, rotation is allowed in the Procrustes calculation. If
        ``False``, only scale and translation effects are allowed in the
        returned transform.
    allow_mirror : `bool`, optional
        If ``True``, the Kabsch algorithm check is not performed, and mirroring
        of the Rotation matrix is permitted.

    Returns
    -------
    transform : :map:`Similarity`
        A :map:`Similarity` transform that optimally aligns the `source` to
        `target`.
    """
    from .rotation import Rotation, optimal_rotation_matrix
    from .translation import Translation
    from .scale import UniformScale
    # Compute the transforms we need - centering translations
    tgt_t = Translation(-target.centre(), skip_checks=True)
    src_t = Translation(-source.centre(), skip_checks=True)
    # and a scale that matches the norm of the source to the norm of the target
    src_s = UniformScale(target.norm() / source.norm(), source.n_dims,
                         skip_checks=True)

    # start building the Procrustes Alignment - src translation followed by
    # scale
    p = Similarity.init_identity(source.n_dims)
    p.compose_before_inplace(src_t)
    p.compose_before_inplace(src_s)

    if rotation:
        # to calculate optimal rotation we need the source and target in the
        # centre and of the correct size. Use the current p to do this
        aligned_src = p.apply(source)
        aligned_tgt = tgt_t.apply(target)
        r = Rotation(optimal_rotation_matrix(aligned_src, aligned_tgt,
                                             allow_mirror=allow_mirror),
                     skip_checks=True)
        p.compose_before_inplace(r)
    # finally, translate the target back
    p.compose_before_inplace(tgt_t.pseudoinverse())
    return p
