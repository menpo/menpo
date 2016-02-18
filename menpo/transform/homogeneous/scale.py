import numpy as np

from .base import HomogFamilyAlignment
from .affine import DiscreteAffine, Affine
from .similarity import Similarity


def Scale(scale_factor, n_dims=None):
    r"""
    Factory function for producing Scale transforms. Zero scale factors are not
    permitted.

    A :class:`UniformScale` will be produced if:

        - A `float` ``scale_factor`` and a ``n_dims`` `kwarg` are provided
        - A `ndarray` ``scale_factor`` with shape ``(n_dims,)`` is provided
          with all elements being the same

    A :class:`NonUniformScale` will be provided if:

        - A `ndarray` ``scale_factor`` with shape ``(n_dims,)`` is provided with
          at least two differing scale factors.

    Parameters
    ----------
    scale_factor : `float` or ``(n_dims,)`` `ndarray`
        Scale for each axis.
    n_dims : `int`, optional
        The dimensionality of the output transform.

    Returns
    -------
    scale : :class:`UniformScale` or :class:`NonUniformScale`
        The correct type of scale

    Raises
    ------
    ValueError
        If any of the scale factors is zero
    """
    from numbers import Number
    if not isinstance(scale_factor, Number):
        # some array like thing - make it a numpy array for sure
        scale_factor = np.asarray(scale_factor)
    if not np.all(scale_factor):
        raise ValueError('Having a zero in one of the scales is invalid')

    if n_dims is None:
        # scale_factor better be a numpy array then
        if np.allclose(scale_factor, scale_factor[0]):
            return UniformScale(scale_factor[0], scale_factor.shape[0])
        else:
            return NonUniformScale(scale_factor)
    else:
        # interpret as a scalar then
        return UniformScale(scale_factor, n_dims)


class NonUniformScale(DiscreteAffine, Affine):
    r"""
    An ``n_dims`` scale transform, with a scale component for each dimension.

    Parameters
    ----------
    scale : ``(n_dims,)`` `ndarray`
        A scale for each axis.
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """

    def __init__(self, scale, skip_checks=False):
        scale = np.asarray(scale)
        if not skip_checks:
            if scale.size > 3 or scale.size < 2:
                raise ValueError("NonUniformScale can only be 2D or 3D"
                                 ", not {}".format(scale.size))
        h_matrix = np.eye(scale.size + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        Affine.__init__(self, h_matrix, skip_checks=True, copy=False)

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
        identity : :class:`NonUniformScale`
            The identity matrix transform.
        """
        return NonUniformScale(np.ones(n_dims))

    @property
    def scale(self):
        r"""
        The scale vector.

        :type: ``(n_dims,)`` `ndarray`
        """
        # Copy the vector as Numpy 1.10 will return a writeable view
        return self.h_matrix.diagonal()[:-1].copy()

    def _transform_str(self):
        message = 'NonUniformScale by {}'.format(self.scale)
        return message

    @property
    def n_parameters(self):
        """
        The number of parameters: ``n_dims``. They have the form
        ``[scale_x, scale_y, ....]`` representing the scale across each axis.

        :type: `list` of `int`
        """
        return self.scale.size

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order ``[s0, s1, ...]``.

        +----------+--------------------------------------------+
        |parameter | definition                                 |
        +==========+============================================+
        |s0        | The scale across the first axis            |
        +----------+--------------------------------------------+
        |s1        | The scale across the second axis           |
        +----------+--------------------------------------------+
        |...       | ...                                        |
        +----------+--------------------------------------------+
        |sn        | The scale across the nth axis              |
        +----------+--------------------------------------------+

        Returns
        -------
        s : ``(n_dims,)`` `ndarray`
            The scale across each axis.
        """
        return self.scale

    def _from_vector_inplace(self, vector):
        r"""
        Updates the :class:`NonUniformScale` inplace.

        Parameters
        ----------
        vector : ``(n_dims,)`` `ndarray`
            The array of parameters.
        """
        np.fill_diagonal(self.h_matrix, vector)
        self.h_matrix[-1, -1] = 1

    @property
    def composes_inplace_with(self):
        r"""
        :class:`NonUniformScale` can swallow composition with any other
        :class:`NonUniformScale` and :class:`UniformScale`.
        """
        return NonUniformScale, UniformScale

    def pseudoinverse(self):
        """
        The inverse scale matrix.

        :type: :class:`NonUniformScale`
        """
        return NonUniformScale(1.0 / self.scale, skip_checks=True)


class UniformScale(DiscreteAffine, Similarity):
    r"""
    An abstract similarity scale transform, with a single scale component
    applied to all dimensions. This is abstracted out to remove unnecessary
    code duplication.

    Parameters
    ----------
    scale : ``(n_dims,)`` `ndarray`
        A scale for each axis.
    n_dims : `int`
        The number of dimensions
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``h_matrix`` for performance.
    """

    def __init__(self, scale, n_dims, skip_checks=False):
        if not skip_checks:
            if n_dims > 3 or n_dims < 2:
                raise ValueError("UniformScale can only be 2D or 3D"
                                 ", not {}".format(n_dims))
        h_matrix = np.eye(n_dims + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        Similarity.__init__(self, h_matrix, copy=False,
                            skip_checks=True)

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
        identity : :class:`UniformScale`
            The identity matrix transform.
        """
        return UniformScale(1, n_dims)

    @property
    def scale(self):
        r"""
        The single scale value.

        :type: `float`
        """
        return self.h_matrix[0, 0]

    def _transform_str(self):
        message = 'UniformScale by {}'.format(self.scale)
        return message

    @property
    def n_parameters(self):
        r"""
        The number of parameters: 1

        :type: `int`
        """
        return 1

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order ``[s]``.

        +----------+--------------------------------+
        |parameter | definition                     |
        +==========+================================+
        |s         | The scale across each axis     |
        +----------+--------------------------------+

        Returns
        -------
        s : `float`
            The scale across each axis.
        """
        return np.asarray(self.scale)

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Parameters
        ----------
        p : `float`
            The parameter
        """
        np.fill_diagonal(self.h_matrix, p)
        self.h_matrix[-1, -1] = 1

    @property
    def composes_inplace_with(self):
        r"""
        :class:`UniformScale` can swallow composition with any other
        :class:`UniformScale`.
        """
        return UniformScale

    def pseudoinverse(self):
        r"""
        The inverse scale.

        :type: :class:`UniformScale`
        """
        return UniformScale(1.0 / self.scale, self.n_dims, skip_checks=True)


class AlignmentUniformScale(HomogFamilyAlignment, UniformScale):
    r"""
    Constructs a :class:`UniformScale` by finding the optimal scale transform to
    align `source` to `target`.

    Parameters
    ----------
    source : :map:`PointCloud`
        The source pointcloud instance used in the alignment
    target : :map:`PointCloud`
        The target pointcloud instance used in the alignment
    """

    def __init__(self, source, target):
        HomogFamilyAlignment.__init__(self, source, target)
        UniformScale.__init__(self, target.norm() / source.norm(),
                              source.n_dims)

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Parameters
        ----------
        p : `float`
            The parameter
        """
        UniformScale._from_vector_inplace(self, p)
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        new_scale = self.target.norm() / self.source.norm()
        np.fill_diagonal(self.h_matrix, new_scale)
        self.h_matrix[-1, -1] = 1

    def as_non_alignment(self):
        r"""Returns a copy of this uniform scale without it's alignment nature.

        Returns
        -------
        transform : :map:`UniformScale`
            A version of this scale with the same transform behavior but
            without the alignment logic.
        """
        return UniformScale(self.scale, self.n_dims)
