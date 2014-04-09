import numpy as np

from .base import HomogFamilyAlignment
from .affine import DiscreteAffine
from .similarity import Similarity


class Translation(DiscreteAffine, Similarity):
    r"""
    An N-dimensional translation transform.

    Parameters
    ----------
    translation : (D,) ndarray
        The translation in each axis.
    """

    def __init__(self, translation):
        translation = np.asarray(translation)
        h_matrix = np.eye(translation.shape[0] + 1)
        h_matrix[:-1, -1] = translation
        Similarity.__init__(self, h_matrix)

    @property
    def n_parameters(self):
        r"""
        The number of parameters: ``n_dims``

        :type: int
        """
        return self.n_dims

    def _build_pseudoinverse(self):
        r"""
        The inverse translation (negated).

        :return: :class:`Translation`
        """
        return Translation(-self.translation_component)

    def _transform_str(self):
        message = 'Translate by %s ' % self.translation_component
        return message

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [t0, t1, ...].

        +-----------+--------------------------------------------+
        |parameter | definition                                  |
        +==========+=============================================+
        |t0        | The translation in the first axis           |
        |t1        | The translation in the second axis          |
        |...       | ...                                         |
        |tn        | The translation in the nth axis             |
        +----------+---------------------------------------------+

        Returns
        -------
        ts : (D,) ndarray
            The translation in each axis.
        """
        return self.h_matrix[:-1, -1]

    def from_vector_inplace(self, p):
        self.h_matrix[:-1, -1] = p

    @classmethod
    def identity(cls, n_dims):
        return Translation(np.zeros(n_dims))


class AlignmentTranslation(HomogFamilyAlignment, Translation):

    def __init__(self, source, target):
        HomogFamilyAlignment.__init__(self, source, target)
        Translation.__init__(self, target.centre - source.centre)

    def from_vector_inplace(self, p):
        Translation.from_vector_inplace(self, p)
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        translation = self.target.centre - self.source.centre
        self.h_matrix[:-1, -1] = translation

    def copy_without_alignment(self):
        return Translation(self.translation_component)
