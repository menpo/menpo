import numpy as np

class Alignment(object):

  def __init__(self, sources, **kwargs):
    """ sources - a list of numpy arrays of landmarks which will be aligned
                    e.g. [landmarks1, landmakrks2,...landmarksn] where 
                    landmarksi is an ndarray of dimension [dim x n_landmarks]
      KWARGS
        target  - a single numpy array (of the same dimension of sources) which
                  every instance of source will be aligned to. If not present,
                  target is set to the mean source position
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
