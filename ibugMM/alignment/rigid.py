import numpy as np
from scipy import linalg
__all__ = ["Procrustes"]

class Alignment(object):

  def __init__(self, sources, **kwargs):
    """ sources - a list of numpy arrays of landmarks which will be aligned
                    e.g. [landmarks1, landmakrks2,...landmarksn] where 
                    landmarksi is an ndarray of dimension [dim x n_landmarks]

        target  - a single numpy array (of the same dimension of sources) which
                  every instance of source will be aligned to
    """
    target = kwargs.get('target',None)
    n_sources    = len(sources)
    n_landmarks  = sources[0].shape[1]
    n_dimensions = sources[0].shape[0]
    self.sources = np.zeros([n_dimensions,n_landmarks,n_sources])
    for i, source in enumerate(sources):
      assert n_dimensions,n_landmarks == source.shape
      self.sources[:,:,i] = source
    if target is None:
      # set the target to the mean source position
      self.target = self.sources.mean(2)[...,np.newaxis]
    else:
      assert n_dimensions,n_landmarks == target.shape
      self.target = target[...,np.newaxis]

  @property
  def n_dimensions(self):
    return self.sources.shape[0]

  @property
  def n_landmarks(self):
    return self.sources.shape[1]

  @property
  def n_sources(self):
    return self.sources.shape[2]
  
class RigidAlignment(Alignment):
  def __init__(self, sources, **kwargs):
    Alignment.__init__(self,sources, **kwargs)

  def buildHomoTranslationMatrix(self,translationVector):
    return buildHomoTranslationMatrix(translationVector,dim=self.n_dimensions) 
  
  def buildHomoScaleMatrix(self,scaleVector):
    return buildHomoScaleMatrix(scaleVector,dim=self.n_dimensions)
  
  def buildHomoRotationMatrix(self,rotationMatrix):
    return buildHomoRotationMatrix(rotationMatrix,dim=self.n_dimensions)

class Procrustes(RigidAlignment):

  def __init__(self, sources, **kwargs):
    RigidAlignment.__init__(self,sources,**kwargs)
    self.operations = []


  def generalProcrusesAlignment(self):
    error = 999999999
    while error > 0.0001:
      self.procrustesStep()
      oldTarget = self.target
      self.target = self.sources.mean(-1)[...,np.newaxis]
      error = np.sum((self.target - oldTarget)**2)
    self.transform = []
    for i in range(self.n_sources):
      self.transform.append(np.eye(self.n_dimensions+1))
    for ops in reversed(self.operations):
      for i in range(self.n_sources):
        t = self.buildHomoTranslationMatrix(ops['translate'][:,:,i].flatten())
        s = self.buildHomoScaleMatrix(ops['rescale'][:,:,i].flatten())
        r = self.buildHomoRotationMatrix(ops['rotation'][i])
        self.transform[i] = np.dot(r,np.dot(s,np.dot(t,self.transform[i])))

  def procrustesAlignment(self):
    self.procrustesStep()

  def procrustesStep(self):
    print 'taking ProcrustesStep'
    ops = {}
    # calculate the translation required for each source to align the sources'
    # centre of mass to the the target centre of mass
    translation = (self.target.mean(1) - self.sources.mean(1))[:,np.newaxis,:]
    # apply the translation to each source respectively
    self.sources += translation
    ops['translate'] = translation
    # calcuate the frobenious norm of each shape as our metric
    scaleSources = np.sqrt(np.apply_over_axes(np.sum,
      (self.sources - self.sources.mean(1)[:,np.newaxis,:])**2,
                           [0,1]))
    scaleTarget = np.sqrt(np.sum((self.target - 
                                  self.target.mean(1)[:,np.newaxis,:])**2))
    rescale = scaleTarget/scaleSources
    self.sources = self.sources*rescale
    ops['rescale'] = rescale
    rotations = []
    #for each source
    for i in range(self.n_sources):
      # calculate the correlation along each dimension
      correlation = np.dot(self.sources[...,i],self.target[...,0].T)
      U,D,Vt = np.linalg.svd(correlation)
      # find the optimal rotation to minimise rotational differences
      rotation = np.dot(Vt.T,U.T)
      rotations.append(rotation)
      # apply the rotation
      self.sources[...,i] = np.dot(rotation,self.sources[...,i])
    ops['rotation'] = rotations
    self.operations.append(ops)
    # compare the oldSource to the new - if the difference is sufficiently 
    # small, stop. Else, call again.


  def buildOperation(self):
    for operation in operations:
      pass

def buildHomoTranslationMatrix(translationVector, **kwargs):
  dim = kwargs.get('dim',3)
  matrix = np.eye(dim+1)
  matrix[:-1,-1] = translationVector
  return matrix

def buildHomoScaleMatrix(scaleVector, **kwargs):
  dim = kwargs.get('dim',3)
  matrix = np.eye(dim+1)
  np.fill_diagonal(matrix,scaleVector)
  # set the corner value back to 1
  matrix[-1,-1] = 1
  return matrix

def buildHomoRotationMatrix(rotationMatrix, **kwargs):
  dim = kwargs.get('dim',3)
  matrix = np.eye(dim+1)
  matrix[:-1,:-1] = rotationMatrix
  return matrix
