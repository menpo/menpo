import numpy as np
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
    self.calculated_geodesics = {}
    self.last_key = None
    self.landmarks = {}
    # some file types include normals - if they are not none, import them
    #if kwargs['normals']:
    #self.normals            = np.array(kwargs.get(['normals']))
    #self.normalsIndex       = np.array(kwargs.get)['normalsIndex']))
    #else:
    #  self.normals, self.normalsIndex = None, None

  def view(self):
    figure = mlab.gcf()
    mlab.clf()
    s = self._render_face()
    mlab.show()
    return s

  def _render_face(self):
    return mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.coordsIndex, 
                             color=(0.5,0.5,0.5)) 

  @property
  def number_of_landmarks(self):
    return len(self.landmarks)

  def view_with_landmarks(self):
    self.view()
    num_landmarks = self.number_of_landmarks
    for num, key in enumerate(self.landmarks):
      i = self.landmarks[key]
      colors = np.ones_like(i)*((num*1.0+0.1)/num_landmarks)
      print 'key: ' + `key` + ' color: ' + `colors`
      s = mlab.points3d(self.coords[i,0], self.coords[i,1],
                        self.coords[i,2], 
                        colors, scale_factor=5.0,
                        vmax=1.0, vmin=0.0) 
    mlab.show()

  def view_location_of_vertex(self, i):
    self.view()
    s = mlab.points3d(self.coords[i,0], self.coords[i,1],
                      self.coords[i,2], 
                      color=(1,1,1), scale_factor=5.0) 
    mlab.show()

  def view_geodesic_contours(self, phi):
    rings = np.mod(phi,20)
    s = mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.coordsIndex, 
                             scalars=rings) 
    mlab.show()

  def view_last_geodesic_contours(self):
    if self.last_key:
      self.view_geodesic_contours(self.calculated_geodesics[self.last_key]['phi'])
    else:
      print "No geodesics have been calculated for this face"

  def calculate_geodesics_for_all_landmarks(self):
    for key in self.landmarks:
      self.calculate_geodesics(self.landmarks[key])

  def calculate_geodesics(self, source_vertex):
    key = tuple(sorted(set(source_vertex)))
    geodesic = self.calculated_geodesics.get(key)
    if geodesic is not None:
      print 'already calculated this geodesic, returning it'
      return geodesic['phi']
    else:
      # calculate the POSITIVE semidefinite laplacian operator
      u_0 = np.zeros(self.n_coords)
      u_0[source_vertex] = 1.0
      if self.L_c is None or self.A is None:
        L_c, A = self.laplacian_operator()
        self.A = csc_matrix(A)
        # calculate the NEGITIVE semidefinite laplacian operator
        self.L_c = -1.0*csc_matrix(L_c)
      print 'Calculated Laplacian and Area matrix'
      t = self.mean_edge_length
      u_t = linalg.spsolve(self.A - t*(self.L_c), u_0)
      print 'Solved for u_t'
      grad_u_t = self.gradient(u_t)
      print 'Solved for grad_u_t'
      grad_u_t_mag  = np.sqrt(np.sum(grad_u_t**2, axis=1))
      X = -1.0*grad_u_t/(grad_u_t_mag).reshape([-1,1])
      # some of the vectors may have been zero length - ensure this
      # is true before and after
      X[grad_u_t_mag == 0] = 0
      print 'Generated X'
      div_X = self.divergence(X)
      print 'Generated div_X'
      phi = linalg.spsolve(self.L_c, div_X)
      phi = phi - phi[source_vertex[0]]
      geodesic = {}
      geodesic['u_0']          = u_0
      geodesic['u_t']          = u_t
      geodesic['grad_u_t']     = grad_u_t
      geodesic['grad_u_t_mag'] = grad_u_t_mag
      geodesic['X']            = X
      geodesic['div_X']        = div_X
      geodesic['phi']          = phi
      self.calculated_geodesics[key] = geodesic
      self.last_key = key
      print 'Generated distances'
      return phi

