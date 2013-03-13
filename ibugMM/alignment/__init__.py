import numpy as np

class Alignment(object):

  def __init__(self, sources, target=None):
    """ sources - a list of numpy arrays of landmarks which will be aligned
                    e.g. [landmarks1, landmakrks2,...landmarksn] where 
                    landmarksi is an ndarray of dimension [n_landmarks x n_dim]
      KWARGS
        target  - a single numpy array (of the same dimension of sources) which
                  every instance of source will be aligned to. If not present,
                  target is set to the mean source position
    """
    if type(sources) == np.ndarray:
      # only a single landmark passed in
      sources = [sources]
    n_sources    = len(sources)
    n_landmarks  = sources[0].shape[0]
    n_dim = sources[0].shape[1]
    self.sources = np.zeros([n_landmarks, n_dim, n_sources])
    for i, source in enumerate(sources):
      assert n_dim, n_landmarks == source.shape
      self.sources[:,:,i] = source
    if target is None:
      # set the target to the mean source position
      self.target = self.sources.mean(2)[...,np.newaxis]
    else:
      assert n_dim, n_landmarks == target.shape
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
