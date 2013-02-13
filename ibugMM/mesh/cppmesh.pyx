# distutils: language = c++
# distutils: sources = ./ibugMM/mesh/mesh.cpp ./ibugMM/mesh/vertex.cpp ./ibugMM/mesh/halfedge.cpp ./ibugMM/mesh/vec3.cpp ./ibugMM/mesh/triangle.cpp

from libcpp.vector cimport vector
from libcpp.set    cimport set
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
import scipy.sparse as sparse
cimport numpy as np
import cython
cimport cython

# externally declare the C++ Mesh, Vertex, Triangle and HalfEdge classes 
cdef extern from "mesh.h":
  cdef cppclass Mesh:
    Mesh(double   *coords,      unsigned n_coords,
         unsigned *coordsIndex, unsigned n_triangles) except +
    unsigned n_triangles, n_coords
    void verifyAttachements()
    void calculateLaplacianOperator(unsigned* i_sparse, unsigned* j_sparse,
                                    double* v_sparse)
    void verifyMesh()
    unsigned n_half_edges
    unsigned n_full_edges
    void generateEdgeIndex(unsigned* edgeIndex)
    void reduceTriangleScalarToVertices(double* triangle_scalar, double* vertex_scalar)
    void reduceTriangleScalarPerVertexToVertices(double* triangle_scalar_p_vert,
                                                 double* vertex_scalar)
    #void calculateGradient(double* v_scalar_field, double* t_vector_gradient)
    #void calculateDivergence(double* t_vector_field, double* v_scalar_divergence)
    #double meanEdgeLength()
    #void triangleAreas(double* areas)
    vector[Vertex*] vertices
    vector[Triangle*] triangles 

cdef extern from "vertex.h":
  cdef cppclass Vertex:
    set[HalfEdge*] halfedges
    void printStatus()

cdef extern from "triangle.h":
  cdef cppclass Triangle:
    void printStatus()

cdef extern from "halfedge.h":
  cdef cppclass HalfEdge:
    pass

# Wrap the Mesh class to produce CppMesh
cdef class CppMesh:
  cdef Mesh* thisptr

  def __cinit__(self, np.ndarray[double,   ndim=2, mode="c"] coords      not None, 
                      np.ndarray[unsigned, ndim=2, mode="c"] coordsIndex not None, **kwargs):
    self.coords = coords;
    self.tri_index = coordsIndex
    self.thisptr = new Mesh(     &coords[0,0],      coords.shape[0], 
                            &coordsIndex[0,0], coordsIndex.shape[0])
    self.edge_index = self._calculate_edge_index()
    self._initialize_cache()
    print 'after construction, CppMesh cache size is ' + "%0.2f" % self.cache_size + 'MB.'

  def __dealloc__(self):
    del self.thisptr

  def check_for_unrefereced_vertices(self):
    # minus one as 0 indexed
    diff_in_max_index = self.n_vertices - 1 - np.max(self.tri_index) 
    if diff_in_max_index == 0:
      pass
    elif diff_in_max_index < 0:
      print 'tri_index refers to a non existant vertex'
    else:
      print 'tri_index does not refer to every vertex'
    set_diff = np.setdiff1d(np.arange(self.n_vertices, dtype=np.uint32), 
                            np.unique(self.tri_index))
    if set_diff.size != 0:
      print 'the following vertices are unreferenced:'
      print set_diff

  @property
  def n_vertices(self):
    return self.thisptr.n_coords

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles

  @property
  def n_fulledges(self):
    return self.thisptr.n_full_edges

  @property
  def n_halfedges(self):
    return self.thisptr.n_half_edges

  @property
  def n_edges(self):
    return self.thisptr.n_half_edges - self.thisptr.n_full_edges

  @property
  def mean_edge_length(self):
    return self._retrieve_from_cache('mean_edge_length')

  @property
  def tri_area(self):
    return self._retrieve_from_cache('tri_area')

  @property
  def voronoi_vertex_area(self):
    return self._retrieve_from_cache('voronoi_vertex_area')

  @property
  def voronoi_vertex_area_matrix(self):
    return self._retrieve_from_cache('voronoi_vertex_area_matrix')

  @property
  def cache_size(self):
    """Returns the current memory usage of the cache in MB
    """
    mem = 0
    for v in self._cache.values():
      if type(v) is np.ndarray:
        mem+= v.nbytes
      elif type(v) is sparse.csc_matrix:
        mem+= v.data.nbytes
    return (1.0*mem)/(2**20)

  def verify_mesh(self):
    self.thisptr.verifyMesh()

  def vertex_status(self, n_vertex):
    assert n_vertex >= 0 and n_vertex < self.n_vertices
    deref(self.thisptr.vertices[n_vertex]).printStatus()

  def tri_status(self, n_triangle):
    assert n_triangle >= 0 and n_triangle < self.n_triangles
    deref(self.thisptr.triangles[n_triangle]).printStatus()

  def heat_geodesics(self, source_vertices):
    u_0 = np.zeros(self.n_vertices)
    u_0[source_vertices] = 1.0
    if self._cache.get('u_t_solver') is None:
      print 'solving u_t for the first time'
      A = self.voronoi_vertex_area_matrix
      L_c = self.laplacian()
      t   = self.mean_edge_length
      self._cache['u_t_solver'] = sparse.linalg.factorized(A - t*L_c)
    u_t_solver = self._cache['u_t_solver']
    u_t = u_t_solver(u_0)
    #print 'Solved for u_t'
    grad_u_t = self.gradient(u_t)
    #print 'Solved for grad_u_t'
    grad_u_t_mag  = np.sqrt(np.sum(grad_u_t * grad_u_t, axis=1))
    # some of the vectors may have been zero length - ensure this
    # is true before and after
    grad_u_t[grad_u_t_mag == 0] = 0
    X = -1.0*grad_u_t/grad_u_t_mag[:,np.newaxis]
    #print 'Generated X'
    div_X = self.divergence(X)
    #print 'Generated div_X'
    if self._cache['phi_solver'] is None:
      print 'solving phi for the first time'
      L_c = self.laplacian()
      self._cache['phi_solver'] = sparse.linalg.factorized(L_c)
    phi_solver = self._cache['phi_solver']
    phi = phi_solver(div_X)
    print 'variance in source vertex distances is ' + `np.var(phi[source_vertices])`
    phi = phi - phi[source_vertices[0]]
    geodesic = {}
    geodesic['u_0']          = u_0
    geodesic['u_t']          = u_t
    geodesic['grad_u_t']     = grad_u_t
    geodesic['grad_u_t_mag'] = grad_u_t_mag
    geodesic['X']            = X
    geodesic['div_X']        = div_X
    geodesic['phi']          = phi
    return geodesic

  def divergence(self, t_vector_field):
    """
    """
    e_01  = self._retrieve_from_cache('e_01')
    e_12  = self._retrieve_from_cache('e_12')
    e_20  = self._retrieve_from_cache('e_20')
    cot_0 = self._retrieve_from_cache('cot_0')
    cot_1 = self._retrieve_from_cache('cot_1')
    cot_2 = self._retrieve_from_cache('cot_2')
    X_dot_e_01 = np.sum(t_vector_field * e_01, axis=1)
    X_dot_e_12 = np.sum(t_vector_field * e_12, axis=1)
    X_dot_e_20 = np.sum(t_vector_field * e_20, axis=1)
    div_X_tri = np.zeros_like(self.tri_index, dtype=np.float)  
    div_X_tri[:,0] = (cot_2 * X_dot_e_01) - (cot_1 * X_dot_e_20)
    div_X_tri[:,1] = (cot_0 * X_dot_e_12) - (cot_2 * X_dot_e_01)
    div_X_tri[:,2] = (cot_1 * X_dot_e_20) - (cot_0 * X_dot_e_12)
    return 0.5*self.reduce_tri_scalar_per_vertex_to_vertices(div_X_tri)

  def gradient(self, np.ndarray[double, ndim=1, mode="c"] v_scalar_field not None):
    """
    """
    X_per_tri = v_scalar_field[self.tri_index]
    N_x_e_01  = self._retrieve_from_cache('N_x_e01')      
    N_x_e_12  = self._retrieve_from_cache('N_x_e12')      
    N_x_e_20  = self._retrieve_from_cache('N_x_e20')      
    sum_total = (N_x_e_01 * X_per_tri[:,2,np.newaxis] +
                 N_x_e_12 * X_per_tri[:,0,np.newaxis] +
                 N_x_e_20 * X_per_tri[:,1,np.newaxis])
    return sum_total/(2*self.tri_area[:,np.newaxis])

  def laplacian(self, weighting='cotangent'):
    """ Calculates the NEGITIVATE SEMI-DEFINITE Laplacian Operator for this mesh.

    A number of different weightings can be used to construct the laplacian - 
    by default, the cotangent method is used (although this can be changed by
    passing in the appropriate weighting arugment)
    """
    cache_key = 'laplacian_' + weighting
    if self._cache.get(cache_key) is None:
      self._cache[cache_key] = self._neg_sd_laplacian(weighting)
    return self._cache[cache_key]

  def reduce_tri_scalar_per_vertex_to_vertices(self, 
      np.ndarray[double, ndim=2, mode="c"] triangle_scalar not None):
    cdef np.ndarray[double, ndim=1, mode='c'] vertex_scalar = np.zeros(self.n_vertices)
    self.thisptr.reduceTriangleScalarPerVertexToVertices(&triangle_scalar[0,0], &vertex_scalar[0])
    return vertex_scalar

  def reduce_tri_scalar_to_vertices(self, 
      np.ndarray[double, ndim=1, mode="c"] triangle_scalar not None):
    cdef np.ndarray[double, ndim=1, mode='c'] vertex_scalar = np.zeros(self.n_vertices)
    self.thisptr.reduceTriangleScalarToVertices(&triangle_scalar[0], &vertex_scalar[0])
    return vertex_scalar

  # --- HELPER ROUTINES for generating cached values ----- #

  def _neg_sd_laplacian(self, weighting='cotangent'):
    cdef np.ndarray[unsigned, ndim=1, mode='c'] i_sparse = np.zeros(
        [self.n_halfedges*2],dtype=np.uint32)
    cdef np.ndarray[unsigned, ndim=1, mode='c'] j_sparse = np.zeros(
        [self.n_halfedges*2],dtype=np.uint32)
    cdef np.ndarray[double,   ndim=1, mode='c'] v_sparse = np.zeros(
        [self.n_halfedges*2])
    self.thisptr.calculateLaplacianOperator(&i_sparse[0], &j_sparse[0], &v_sparse[0])
    L_c = sparse.coo_matrix((v_sparse, (i_sparse, j_sparse)))
    # we return the negitive -> switch
    return -1.0*sparse.csc_matrix(L_c)

  def _calculate_edge_index(self):
    cdef np.ndarray[unsigned, ndim=2, mode='c'] edge_index = np.zeros(
        [self.n_edges, 2], dtype=np.uint32)
    self.thisptr.generateEdgeIndex(&edge_index[0,0])
    return edge_index

  def _retrieve_from_cache(self, value):
    if self._cache.get(value) is None:
      self._initialize_cache()
    return self._cache[value]

  def _initialize_cache(self):
    self._cache = {}
    self._cache['u_t_solver'] = None
    self._cache['phi_solver'] = None

    # find mean edge length
    c_edge = self.coords[self.edge_index]
    edge_vectors = c_edge[:,1,:] - c_edge[:,0,:]
    self._cache['mean_edge_length'] = np.mean(
        np.sqrt(np.sum(edge_vectors * edge_vectors, axis=1)))

    c_tri = self.coords[self.tri_index]

    # build edges in counter-clockwise manner
    e_01 = c_tri[:,1,:] - c_tri[:,0,:]
    e_12 = c_tri[:,2,:] - c_tri[:,1,:]
    e_20 = c_tri[:,0,:] - c_tri[:,2,:]
    mag_e_01 = np.sqrt(np.sum(e_01 * e_01, axis=1))
    mag_e_12 = np.sqrt(np.sum(e_12 * e_12, axis=1))
    mag_e_20 = np.sqrt(np.sum(e_20 * e_20, axis=1))
    unit_e_01 = e_01 / (mag_e_01[...,np.newaxis])
    unit_e_12 = e_12 / (mag_e_12[...,np.newaxis])
    unit_e_20 = e_20 / (mag_e_20[...,np.newaxis])

    cos_0 = -1.0 * np.sum(unit_e_01 * unit_e_20, axis=1)
    cos_1 = -1.0 * np.sum(unit_e_01 * unit_e_12, axis=1)
    cos_2 = -1.0 * np.sum(unit_e_12 * unit_e_20, axis=1)

    cross_0 = -1.0 * np.cross(unit_e_01, unit_e_20)
    cross_1 = -1.0 * np.cross(unit_e_12, unit_e_01)
    cross_2 = -1.0 * np.cross(unit_e_20, unit_e_12)

    sin_0 = np.sqrt(np.sum(cross_0 * cross_0, axis=1))
    sin_1 = np.sqrt(np.sum(cross_1 * cross_1, axis=1))
    sin_2 = np.sqrt(np.sum(cross_2 * cross_2, axis=1))

    unit_tri_normal = cross_0/sin_0[:,np.newaxis]
    area_cross = -1.0 * np.cross(e_01, e_20)
    # areas
    tri_area = 0.5*np.sqrt(np.sum(area_cross*area_cross, axis=1))
    voronoi_vertex_area = self.reduce_tri_scalar_to_vertices(tri_area)/3.0
    A_i = np.arange(self.n_vertices)
    Acoo = sparse.coo_matrix((voronoi_vertex_area, (A_i,A_i)))
    self._cache['voronoi_vertex_area_matrix'] = sparse.csc_matrix(Acoo)

    cot_0 = cos_0/sin_0
    cot_1 = cos_1/sin_1
    cot_2 = cos_2/sin_2
    self._cache['c_tri']     = c_tri
    self._cache['e_01']      = e_01
    self._cache['e_12']      = e_12
    self._cache['e_20']      = e_20
    self._cache['mag_e_01']  = mag_e_01
    self._cache['mag_e_12']  = mag_e_12
    self._cache['mag_e_20']  = mag_e_20
    self._cache['unit_e_01'] = unit_e_01
    self._cache['unit_e_12'] = unit_e_12
    self._cache['unit_e_20'] = unit_e_20
    self._cache['cos_0']     = cos_0
    self._cache['cos_1']     = cos_1
    self._cache['cos_2']     = cos_2
    self._cache['cross_0']   = cross_0
    self._cache['cross_1']   = cross_1
    self._cache['cross_2']   = cross_2
    self._cache['sin_0']     = sin_0
    self._cache['sin_1']     = sin_1
    self._cache['sin_2']     = sin_2
    self._cache['cot_0']     = cot_0
    self._cache['cot_1']     = cot_1
    self._cache['cot_2']     = cot_2
    self._cache['unit_tri_normal'] = unit_tri_normal
    self._cache['tri_area'] = tri_area
    self._cache['voronoi_vertex_area'] = voronoi_vertex_area 

    # few other specifics for gradient
    self._cache['N_x_e01'] = np.cross(unit_tri_normal, e_01)
    self._cache['N_x_e12'] = np.cross(unit_tri_normal, e_12)
    self._cache['N_x_e20'] = np.cross(unit_tri_normal, e_20)

  #def _old_gradient(self, np.ndarray[double, ndim=1, mode="c"] v_scalar_field not None):
  #  """
  #  Return the gradient (per face) of the per vertex scalar field 

  #  C++ effects:
  #  vertex_scalar   - the scalar field value (per vertex) that we are taking
  #                    the gradient of.
  #  triangle_vector - the resulting gradient (per triangle)
  #  :param s_field: scalar field value per vertex
  #  :type s_field: ndarray[1,n_vertices]
  #  :return: Gradient evaluated over each triangle
  #  :rtype: ndarray[float]
  # 
  #  """
  #  cdef np.ndarray[double, ndim=2,mode ='c'] t_vector_gradient = np.zeros(
  #      [self.n_triangles,3])
  #  self.thisptr.calculateGradient(&v_scalar_field[0], &t_vector_gradient[0,0])
  #  return t_vector_gradient

  #def _old_divergence(self, np.ndarray[double, ndim=2, mode="c"] t_vector_field not None):
  #  """
  #  Return the divergence (per vertex) of the field stored in triangle_vector.

  #  C++ effects:
  #  triangle_vector - input
  #  vertex_scalar   - result storage

  #  :return: Gradient evaluated over each triangle
  #  :rtype: ndarray[float]
  # 
  #  """
  #  cdef np.ndarray[double, ndim=1, mode='c'] v_scalar_divergence = np.zeros(self.n_vertices)
  #  self.thisptr.calculateDivergence(&t_vector_field[0,0], &v_scalar_divergence[0])
  #  return v_scalar_divergence



