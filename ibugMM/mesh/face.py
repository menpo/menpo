import numpy as np
# expensive to import mlab - leave it for now
from mayavi import mlab
from cppmesh import CppMesh
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import linalg 

class Face(CppMesh):

  def __init__(self, **kwargs):
    CppMesh.__init__(self)
    self.textureCoords      = kwargs.get('textureCoords')
    self.textureCoordsIndex = kwargs.get('textureCoordsIndex')
    self.texture            = kwargs.get('texture')
    self.L_c = None
    self.A = None
    # some file types include normals - if they are not none, import them
    #if kwargs['normals']:
    #self.normals            = np.array(kwargs.get(['normals']))
    #self.normalsIndex       = np.array(kwargs.get)['normalsIndex']))
    #else:
    #  self.normals, self.normalsIndex = None, None

  def view(self):
    s = mlab.triangular_mesh(self.coords[:,0],self.coords[:,1],
                             self.coords[:,2],self.coordsIndex) 
    mlab.show()

  def view_geodesic_contours(self):
    rings = np.zeros_like(self.phi)
    rings[np.where(np.abs(np.mod(self.phi,20)) < 2)] = 1
    s = mlab.triangular_mesh(self.coords[:,0],self.coords[:,1],
                             self.coords[:,2],self.coordsIndex, 
                             scalars=rings) 
    mlab.show()


  def calculate_geodesics(self, source_vertex):
    # calculate the POSITIVE semidefinite laplacian operator
    self.u_0 = np.zeros(self.n_coords)
    self.u_0[source_vertex] = 1.0
    if self.L_c is None or self.A is None:
      L_c, A = self.laplacian_operator()
      self.A = csc_matrix(A)
      # calculate the NEGITIVE semidefinite laplacian operator
      self.L_c = -1.0*csc_matrix(L_c)
    print 'Calculated Laplacian and Area matrix'
    t = self.mean_edge_length
    self.u_t = linalg.spsolve(self.A - t*(self.L_c), self.u_0)
    print 'Solved for u_t'
    self.grad_u_t = self.gradient(self.u_t)
    print 'Solved for grad_u_t'
    self.grad_u_t_mag  = np.sqrt(np.sum(self.grad_u_t**2, axis=1))
    self.X = -1.0*self.grad_u_t/(self.grad_u_t_mag).reshape([-1,1])
    # some of the vectors may have been zero length - ensure this
    # is true before and after
    self.X[self.grad_u_t_mag == 0] = 0
    print 'Generated X'
    self.div_X = self.divergence(self.X)
    print 'Generated div_X'
    phi = linalg.spsolve(self.L_c, self.div_X)
    self.phi = phi - phi[source_vertex]
    print 'Generated distances'
    print self.phi
    return self.phi



