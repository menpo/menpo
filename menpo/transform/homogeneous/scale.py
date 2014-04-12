import numpy as np

from .base import HomogFamilyAlignment
from .affine import DiscreteAffine, Affine
from .similarity import Similarity


def Scale(scale_factor, n_dims=None):
    r"""
    Factory function for producing Scale transforms. Zero scale factors are not
    permitted.

    A :class:`UniformScale` will be produced if:

        - A float ``scale_factor`` and a ``n_dims`` kwarg are provided
        - A ndarray scale_factor with shape (``n_dims``, ) is provided with all
          elements being the same

    A :class:`NonUniformScale` will be provided if:

        - A ndarray ``scale_factor`` with shape (``n_dims``, ) is provided with
          at least two differing scale factors.

    Parameters
    ----------
    scale_factor: double or (D,) ndarray
        Scale for each axis.
    n_dims: int
        The dimensionality of the output transform.

    Returns
    -------
    scale : :class:`UniformScale` or :class:`NonUniformScale`
        The correct type of scale

    Raises
    -------
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
    scale : (D,) ndarray
        A scale for each axis.
    """

    def __init__(self, scale):
        scale = np.asarray(scale)
        h_matrix = np.eye(scale.size + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        Affine.__init__(self, h_matrix)

    @classmethod
    def identity(cls, n_dims):
        return NonUniformScale(np.ones(n_dims))

    def set_h_matrix(self, value):
        raise NotImplementedError("The h_matrix cannot "
                                  "be set on a NonUniformScale.")

    @property
    def scale(self):
        r"""
        The scale vector.

        :type: (D,) ndarray
        """
        return self.h_matrix.diagonal()[:-1]

    def _transform_str(self):
        message = 'NonUniformScale by %s ' % self.scale
        return message

    @property
    def n_parameters(self):
        """
        The number of parameters: ``n_dims``.

        :type: int

        ``n_dims`` parameters - ``[scale_x, scale_y, ....]`` - The scalar values
        representing the scale across each axis.
        """
        return self.scale.size

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [s0, s1, ...].

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
        s : (D,) ndarray
            The scale across each axis.
        """
        return self.scale

    def from_vector_inplace(self, vector):
        r"""
        Updates the NonUniformScale inplace.

        Parameters
        ----------
        vector : (D,) ndarray
            The array of parameters.

        """
        np.fill_diagonal(self.h_matrix, vector)
        self.h_matrix[-1, -1] = 1

    @property
    def composes_inplace_with(self):
        return (NonUniformScale, UniformScale)

    def _build_pseudoinverse(self):
        """
        The inverse scale.

        :type: :class:`NonUniformScale`
        """
        return NonUniformScale(1.0 / self.scale)


class UniformScale(DiscreteAffine, Similarity):
    r"""
    An abstract similarity scale transform, with a single scale component
    applied to all dimensions. This is abstracted out to remove unnecessary
    code duplication.
    """

    def __init__(self, scale, n_dims):
        h_matrix = np.eye(n_dims + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        Similarity.__init__(self, h_matrix)

    @classmethod
    def identity(cls, n_dims):
        return UniformScale(1, n_dims)

    @property
    def scale(self):
        r"""
        The single scale value.

        :type: double
        """
        return self.h_matrix[0, 0]

    def _transform_str(self):
        message = 'UniformScale by %f ' % self.scale
        return message

    @property
    def n_parameters(self):
        r"""
        The number of parameters: 1

        :type: int
        """
        return 1

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [s].

        +----------+--------------------------------+
        |parameter | definition                     |
        +==========+================================+
        |s         | The scale across each axis     |
        +----------+--------------------------------+

        Returns
        -------
        s : double
            The scale across each axis.
        """
        return self.scale

    def from_vector_inplace(self, p):
        np.fill_diagonal(self.h_matrix, p)
        self.h_matrix[-1, -1] = 1

    @property
    def composes_inplace_with(self):
        return UniformScale

    def _build_pseudoinverse(self):
        r"""
        The inverse scale.

        :type: type(self)
        """
        return type(self)(1.0 / self.scale, self.n_dims)


class AlignmentUniformScale(HomogFamilyAlignment, UniformScale):

    def __init__(self, source, target):
        HomogFamilyAlignment.__init__(self, source, target)
        UniformScale.__init__(self, target.norm() / source.norm(),
                              source.n_dims)

    def from_vector_inplace(self, p):
        UniformScale.from_vector_inplace(self, p)
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        new_scale = self.target.norm() / self.source.norm()
        np.fill_diagonal(self.h_matrix, new_scale)
        self.h_matrix[-1, -1] = 1

    def copy_without_alignment(self):
        return UniformScale(self.scale, self.n_dims)
