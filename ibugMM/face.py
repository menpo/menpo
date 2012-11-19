import numpy as np
from mayavi import mlab

class Face(object):

  def __init__(self, **kwargs):
    #self._coords            = np.array(_homoVector(kwargs['coords']))
    self.coords             = np.array(kwargs['coords'])
    self.textureCoords      = np.array(kwargs['textureCoords'])
    self.coordsIndex        = np.array(kwargs['coordsIndex']) 
    self.textureCoordsIndex = np.array(kwargs['textureCoordsIndex'])
    self.texture            = kwargs['texture']

    # some file types include normals - if they are not none, import them
    if kwargs['normals']:
      self.normals            = np.array(kwargs['normals'])
      self.normalsIndex       = np.array(kwargs['normalsIndex'])
    else:
      self.normals, self.normalsIndex = None, None

  @property
  def n_coords(self):
    return len(self.coords)

  @property
  def n_triangles(self):
    return len(self.coordsIndex)

  @property
  def n_landmarks(self):
    return len(self.landmarkIndex)

  def view(self):
    s = mlab.triangular_mesh(self.coords[:,0],self.coords[:,1],
                             self.coords[:,2],self.coordsIndex) 
    mlab.show()

