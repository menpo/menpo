from pybug.align.base import Alignment, ParallelAlignment


class RigidAlignment(Alignment):
    """ Abstract class specializing in rigid alignments. As all such alignments
      are affine, a 'h_transforms' list is present, storing a homogeneous
      transform matrix for each source specifying the transform it has
      undergone to match the target.
  """

    def __init__(self, source, target):
        Alignment.__init__(self, source, target)

    @property
    def scalerotation_matrices(self):
        """ Returns a list of 2 dimensional numpy arrays, each of shape
        (n_dimensions,n_dimensions) which can be applied to each source frame
        to rigidly align to the target frame. Needs to be used in conjunction
        with translation_vectors. (source*scalerotatation + translation ->
        target)
    """
        return [matrix[:-1, :-1] for matrix in self.h_transforms]

    @property
    def translation_vectors(self):
        """ Returns a list of translation vectors, each of shape
        (n_dimensions,) which can be applied to each source frame to rigidly
         align to the target frame (source*scalerotatation + translation ->
         target)
    """
        return [matrix[-1, :-1] for matrix in self.h_transforms]
        pass

    def h_transform_for_source(self, source):
        i = self._lookup[_numpy_hash(source)]
        return self.h_transforms[i]

    def scalerotation_translation_for_source(self, source):
        i = self._lookup[_numpy_hash(source)]
        return self.scalerotation_matrices[i], self.translation_vectors[i]

    def _normalise_transformation_matrices(self):
        """ Ensures that the transform matrix has a unit z scale,
        and is affine (e.g. bottom row = [0,0,0,1] for dim = 3)
    """
        pass


class ParallelRigidAlignment(ParallelAlignment):
    """ Abstract class specializing in rigid alignments. As all such alignments
      are affine, a 'h_transforms' list is present, storing a homogeneous
      transform matrix for each source specifying the transform it has
      undergone to match the target.
  """

    def __init__(self, sources, **kwargs):
        super(ParallelRigidAlignment, self).__init__(sources, **kwargs)
