import numpy as np
from . import Alignment
from scipy import linalg

class RigidAlignment(Alignment):
  """ Abstract class specializing in rigid alignments. As all such alignments
      are affine, a 'transformationMatix' list is present, storing a homogeneous
      transformation matrix for each source specifying the transform it has
      undergone to match target.
  """
  def __init__(self, sources, **kwargs):
    Alignment.__init__(self, sources, **kwargs)
    self.transformation_matrices = []

  def h_translation_matrix(self, translation_v):
    return h_translation_matrix(translation_v ,dim=self.n_dimensions)

  def h_scale_matrix(self, scaleVector):
    return h_scale_matrix(scaleVector,dim=self.n_dimensions)

  def h_rotation_matrix(self,rotationMatrix):
    return h_rotation_matrix(rotationMatrix,dim=self.n_dimensions)

  @property
  def scale_rotation_matrices(self):
    """ Returns a list of scaleRotationMatrices, each of shape
        (n_dimensions,n_dimensions) which can be applied to each source frame
        to rigidly align to the target frame. Needs to be used in conjunction
        with translationVectors. (scaleRotatation*source + translation -> target)
    """
    return [matrix[:-1,:-1] for matrix in self.transformation_matrices]

  @property
  def translation_vectors(self):
    """ Returns a list of translation vectors, each of shape (n_dimensions,)
        which can be applied to each source frame to rigidly align to the
        target frame (scaleRotatation*source + translation -> target)
    """
    return [matrix[:-1,-1] for matrix in self.transformation_matrices]
    pass

  def _normalise_transformation_matrices(self):
    """ Ensures that the transformation matrix has a unit z scale,
        and is affine (e.g. bottom row = [0,0,0,1] for dim = 3)
    """
    pass


class Procrustes(RigidAlignment):

  def __init__(self, sources, **kwargs):
    RigidAlignment.__init__(self, sources, **kwargs)
    self.operations = []

  def general_alignment(self):
    error = 999999999
    while error > 0.0001:
      self._procrustes_step()
      old_target = self.target
      self.target = self.sources.mean(axis=-1)[...,np.newaxis]
      # compare the oldSource to the new - if the difference is sufficiently
      # small, stop. Else, call again.
      error = np.sum((self.target - old_target)**2)
      print 'error is ' + `error`
    self.transformation_matrices = []
    for i in range(self.n_sources):
      self.transformation_matrices.append(np.eye(self.n_dimensions+1))
    for ops in self.operations:
      for i in range(self.n_sources):
        t = self.h_translation_matrix(ops['translate'][..., i].flatten())
        s = self.h_scale_matrix(ops['rescale'][..., i].flatten())
        r = self.h_rotation_matrix(ops['rotation'][i])
        self.transformation_matrices[i] = np.dot(t,
                                          np.dot(s,
                                          np.dot(r,
                                                 self.transformation_matrices[i]
                                                )))
    self._normalise_transformation_matrices()

  def _procrustes_step(self):
    print 'taking Procrustes step'
    ops = {}
    # calculate the translation required for each source to align the sources'
    # centre of mass to the the target centre of mass
    translation = (self.target.mean(axis=0) - 
        self.sources.mean(axis=0))[np.newaxis, ...]
    # apply the translation to each source respectively
    self.sources += translation
    ops['translate'] = translation
    # calcuate the frobenious norm of each shape as our metric
    scale_sources = np.sqrt(np.apply_over_axes(np.sum,
      (self.sources - self.sources.mean(axis=0))**2,
                           [0,1]))
    scale_target = np.sqrt(np.sum((self.target -
                                  self.target.mean(axis=0))**2))
    rescale = scale_target/scale_sources
    self.sources = self.sources*rescale
    ops['rescale'] = rescale
    rotations = []
    #for each source
    for i in range(self.n_sources):
      # calculate the correlation along each dimension
      correlation = np.dot(self.sources[...,i].T, self.target[...,0])
      U,D,Vt = np.linalg.svd(correlation)
      # find the optimal rotation to minimise rotational differences
      rotation = np.dot(U, Vt)
      rotations.append(rotation)
      # apply the rotation
      self.sources[...,i] = np.dot(self.sources[...,i], rotation)
    ops['rotation'] = rotations
    self.operations.append(ops)

def h_translation_matrix(translation_vector, dim=3):
  matrix = np.eye(dim + 1)
  matrix[:-1, -1] = translation_vector
  return matrix

def h_scale_matrix(scale_vector, dim=3):
  matrix = np.eye(dim + 1)
  np.fill_diagonal(matrix,scale_vector)
  # set the corner value back to 1
  matrix[-1, -1] = 1
  return matrix

def h_rotation_matrix(rotation_matrix, dim=3):
  matrix = np.eye(dim + 1)
  matrix[:-1, :-1] = rotation_matrix
  return matrix
