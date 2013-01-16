import numpy as np
# expensive to import mlab - leave it for now
from mayavi import mlab
from cppmesh import CppMesh

class Face(CppMesh):

  def __init__(self, **kwargs):
    CppMesh.__init__(self)
    #self.textureCoords      = kwargs.get('textureCoords')
    #self.textureCoordsIndex = kwargs.get('textureCoordsIndex')
    #self.texture            = kwargs['texture']

    # some file types include normals - if they are not none, import them
    #if kwargs['normals']:
    #  self.normals            = np.array(kwargs['normals'])
    #  self.normalsIndex       = np.array(kwargs['normalsIndex'])
    #else:
    #  self.normals, self.normalsIndex = None, None

  def view(self):
    #s = mlab.triangular_mesh(self.coords[:,0],self.coords[:,1],
    #                         self.coords[:,2],self.coordsIndex) 
    #mlab.show()
    pass

