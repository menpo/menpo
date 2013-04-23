from docutils.nodes import target
import numpy as np
from . import _numpy_hash
from pybug.alignment import Alignment


class LinearTransformation(object):
    def __init__(self, n_dim):
        self.n_dim = n_dim


class Rotation(LinearTransformation):
    pass


class Scale(LinearTransformation):
    pass


class Translation(LinearTransformation):
    pass


def h_translation_matrix(translation_vector):
    dim = translation_vector.size
    matrix = np.eye(dim + 1)
    matrix[-1, :-1] = translation_vector
    return matrix


def h_scale_matrix(scale_vector, dim=None):
    if dim is None:
        dim = scale_vector.size
    matrix = np.eye(dim + 1)
    np.fill_diagonal(matrix, scale_vector)
    # set the corner value back to 1
    matrix[-1, -1] = 1
    return matrix


def h_rotation_matrix(rotation_matrix):
    dim = rotation_matrix.shape[0]
    matrix = np.eye(dim + 1)
    matrix[:-1, :-1] = rotation_matrix
    return matrix


class RigidAlignment(Alignment):
    """ Abstract class specializing in rigid alignments. As all such alignments
      are affine, a 'h_transforms' list is present, storing a homogeneous
      transformation matrix for each source specifying the transform it has
      undergone to match the target.
  """

    def __init__(self, source, target):
        Alignment.__init__(self, source, target)

    @property
    def scalerotation_matrices(self):
        """ Returns a list of 2 dimensional numpy arrays, each of shape
        (n_dimensions,n_dimensions) which can be applied to each source frame
        to rigidly align to the target frame. Needs to be used in conjunction
        with translation_vectors. (source*scalerotatation + translation -> target)
    """
        return [matrix[:-1, :-1] for matrix in self.h_transforms]

    @property
    def translation_vectors(self):
        """ Returns a list of translation vectors, each of shape (n_dimensions,)
        which can be applied to each source frame to rigidly align to the
        target frame (source*scalerotatation + translation -> target)
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
        """ Ensures that the transformation matrix has a unit z scale,
        and is affine (e.g. bottom row = [0,0,0,1] for dim = 3)
    """
        pass


class ParallelRigidAlignment(RigidAlignment):
    """ Abstract class specializing in rigid alignments. As all such alignments
      are affine, a 'h_transforms' list is present, storing a homogeneous
      transformation matrix for each source specifying the transform it has
      undergone to match the target.
  """

    def __init__(self, sources, **kwargs):
        Alignment.__init__(self, sources, **kwargs)
        self.h_transforms = []

    @property
    def scalerotation_matrices(self):
        """ Returns a list of 2 dimensional numpy arrays, each of shape
        (n_dimensions,n_dimensions) which can be applied to each source frame
        to rigidly align to the target frame. Needs to be used in conjunction
        with translation_vectors. (source*scalerotatation + translation -> target)
    """
        return [matrix[:-1, :-1] for matrix in self.h_transforms]

    @property
    def translation_vectors(self):
        """ Returns a list of translation vectors, each of shape (n_dimensions,)
        which can be applied to each source frame to rigidly align to the
        target frame (source*scalerotatation + translation -> target)
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
        """ Ensures that the transformation matrix has a unit z scale,
        and is affine (e.g. bottom row = [0,0,0,1] for dim = 3)
    """
        pass


class Procrustes(RigidAlignment):
    """Procrustes Alignment of a set of source landmarks to a target.
    """
    def __init__(self, source, target):
        RigidAlignment.__init__(self, source, target)
        self._procrustes_step()

    @property
    def error(self):
        return np.sum((self.target - self.aligned_source) ** 2)

    def _procrustes_step(self):
        # calculate the translation required to align the sources' centre of
        # mass to the the target centre of mass
        self.translation = (self.target.mean(axis=0) - self.source.mean(
            axis=0))
        # apply the translation to the source
        print self.aligned_source
        np.add(self.source, self.translation, out=self.aligned_source)
        print self.aligned_source
        scale_source = np.linalg.norm(self.aligned_source)
        scale_target = np.linalg.norm(self.target)
        self.scale = scale_target / scale_source
        np.multiply(self.aligned_source, self.scale, out=self.aligned_source)
        print self.aligned_source
        # calculate the correlation along each dimension
        correlation = np.dot(self.aligned_source.T, self.target)
        U, D, Vt = np.linalg.svd(correlation)
        # find the optimal rotation to minimise rotational differences
        self.rotation = np.dot(U, Vt)
        # apply the rotation
        np.dot(self.aligned_source, self.rotation.T, out=self.aligned_source)
        print self.aligned_source


class ParallelProcrustes(ParallelRigidAlignment):
    def __init__(self, sources, **kwargs):
        RigidAlignment.__init__(self, sources, **kwargs)

    def general_alignment(self):
        # stores the items used in each procrustes step
        self.operations = []
        error = 999999999
        while error > 0.0001:
            self._procrustes_step()
            old_target = self.target
            self.target = self.aligned_sources.mean(axis=-1)[..., np.newaxis]
            error = np.sum((self.target - old_target) ** 2)
            print 'error is ' + `error`
        self.h_transforms = []
        for i in range(self.n_sources):
            self.h_transforms.append(np.eye(self.n_dimensions + 1))
            for ops in self.operations:
                t = h_translation_matrix(ops['translate'][..., i].flatten())
                s = h_scale_matrix(ops['rescale'][..., i].flatten(),
                                   dim=self.n_dimensions)
                r = h_rotation_matrix(ops['rotation'][i])
                self.h_transforms[i] = np.dot(self.h_transforms[i],
                                              np.dot(t,
                                                     np.dot(s,
                                                            r)))

    def _procrustes_step(self):
        print 'taking Procrustes step'
        ops = {}
        # calculate the translation required for each source to align the
        # sources' centre of mass to the the target centre of mass
        translation = (self.target.mean(axis=0) -
                       self.aligned_sources.mean(axis=0))[np.newaxis, ...]
        # apply the translation to each source respectively
        self.aligned_sources += translation
        ops['translate'] = translation
        # calcuate the frobenious norm of each shape as our metric
        scale_sources = np.sqrt(np.apply_over_axes(np.sum,
                                                   (
                                                       self.aligned_sources - self.aligned_sources.mean(
                                                           axis=0)) ** 2,
                                                   [0, 1]))
        scale_target = np.sqrt(np.sum((self.target -
                                       self.target.mean(axis=0)) ** 2))
        rescale = scale_target / scale_sources
        self.aligned_sources = self.aligned_sources * rescale
        ops['rescale'] = rescale
        rotations = []
        #for each source
        for i in range(self.n_sources):
            # calculate the correlation along each dimension
            correlation = np.dot(self.aligned_sources[..., i].T,
                                 self.target[..., 0])
            U, D, Vt = np.linalg.svd(correlation)
            # find the optimal rotation to minimise rotational differences
            rotation = np.dot(U, Vt)
            rotations.append(rotation)
            # apply the rotation
            self.aligned_sources[..., i] = np.dot(self.aligned_sources[..., i],
                                                  rotation)
        ops['rotation'] = rotations
        self.operations.append(ops)


