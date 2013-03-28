# distutils: language = c++
# distutils: sources = ./pybug/shape/mesh/cpp/mesh.cpp ./pybug/shape/mesh/cpp/vertex.cpp ./pybug/shape/mesh/cpp/halfedge.cpp ./pybug/shape/mesh/cpp/vec3.cpp ./pybug/shape/mesh/cpp/triangle.cpp

from libcpp.vector cimport vector
from libcpp.set    cimport set
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
import scipy.sparse as sparse
cimport numpy as np
import cython
cimport cython

# externally declare the C++ Mesh, Vertex, Triangle and HalfEdge classes
cdef extern from "./cpp/mesh.h":
  cdef cppclass Mesh:
    Mesh(double   *coords, unsigned n_vertices,
         unsigned *tri_index, unsigned n_triangles) except +
    unsigned n_vertices
    unsigned n_triangles
    unsigned n_halfedges
    unsigned n_fulledges
    vector[Vertex*] vertices
    vector[Triangle*] triangles
    vector[HalfEdge*] halfedges
    void laplacian(unsigned* i_sparse, unsigned* j_sparse,
                   double* v_sparse)
    void cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
                             double* v_sparse, double* cotangent_weights)
    void verify_mesh()
    void generate_edge_index(unsigned* edgeIndex)
    void reduce_tri_scalar_to_vertices(double* triangle_scalar, double* vertex_scalar)
    void reduce_tri_scalar_per_vertex_to_vertices(double* triangle_scalar_p_vert,
                                                 double* vertex_scalar)

  cdef enum LaplacianWeightType:
    combinatorial
    distance

cdef extern from "./cpp/vertex.h":
  cdef cppclass Vertex:
    set[HalfEdge*] halfedges
    void status()

cdef extern from "./cpp/triangle.h":
  cdef cppclass Triangle:
    void status()

cdef extern from "./cpp/halfedge.h":
  cdef cppclass HalfEdge:
    pass

# Wrap the Mesh class to produce CppTriangleMesh
cdef class CppTriangleMesh:
  cdef Mesh* thisptr

  def __cinit__(self, np.ndarray[double,   ndim=2, mode="c"] coords      not None,
                      np.ndarray[unsigned, ndim=2, mode="c"] tri_index not None, **kwargs):
    self.coords = coords;
    self.tri_index = tri_index
    #print &coords[0,0]
    self.thisptr = new Mesh(&coords[0,0], coords.shape[0],
        &tri_index[0,0], tri_index.shape[0])
    ##self._translate_to_origin()
    ##self._rescale_to_unit_mean_length()
    #self.edge_index = self._calculate_edge_index()
    #self._initialize_cache()
    #self._check_for_unreferenced_vertices()
    #print 'after construction, CppMesh cache size is ' + "%0.2f" % self.cache_size + 'MB.'

  def __dealloc__(self):
    del self.thisptr

  def _translate_to_origin(self):
    self.coords = np.subtract(self.coords, np.mean(self.coords, axis=0), out=self.coords)

  def _rescale_to_unit_mean_length(self):
    self.coords = np.divide(self.coords, np.mean(np.sqrt(np.sum((self.coords**2), axis=1))), out=self.coords)

  @property
  def n_vertices(self):
    return self.thisptr.n_vertices

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles

  @property
  def n_fulledges(self):
    return self.thisptr.n_fulledges

  @property
  def n_halfedges(self):
    return self.thisptr.n_halfedges

  @property
  def n_edges(self):
    return self.thisptr.n_halfedges - self.thisptr.n_fulledges

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
    self.thisptr.verify_mesh()

  def vertex_status(self, n_vertex):
    assert n_vertex >= 0 and n_vertex < self.n_vertices
    deref(self.thisptr.vertices[n_vertex]).status()

  def tri_status(self, n_triangle):
    assert n_triangle >= 0 and n_triangle < self.n_triangles
    deref(self.thisptr.triangles[n_triangle]).status()

  def _check_for_unreferenced_vertices(self):
    """Prints message if any coord is unaccounted for in tri_index
    """
    set_diff = np.setdiff1d(np.arange(self.n_vertices, dtype=np.uint32),
                            np.unique(self.tri_index))
    if set_diff.size == 0:
      return
    else:
      print 'the following vertices are unreferenced:'
      print set_diff
      diff_in_max_index = self.n_vertices - 1 - np.max(self.tri_index)
      if diff_in_max_index == 0:
        pass
      elif diff_in_max_index < 0:
        print '(tri_index refers to a non-existant vertex)'
      else:
        print '(tri_index does not refer to every vertex)'

  def divergence(self, t_vector_field not None):
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
      self._cache[cache_key] = self._laplacian(weighting)
    return self._cache[cache_key]

  def _laplacian(self, weighting='cotangent'):
    if weighting == 'cotangent':
      return self._cot_laplacian()
    else:
      print "Don't know how to process a laplacian of type " + `weighting`

  def _cot_laplacian(self):
    """A specific routine for calculating the Cotangent Laplacian.

    Evaluating the cotangent of every angle is fairly inefficiant in serial in
    C++. It's much faster to compute all cotangent angles in numpy and then
    pass the results into C++, so the only role that C++ plays is in looping
    around it's pointer structure.
    """
    cot_0 = self._retrieve_from_cache('cot_0')
    cot_1 = self._retrieve_from_cache('cot_1')
    cot_2 = self._retrieve_from_cache('cot_2')
    cdef np.ndarray[double,   ndim=2, mode='c'] cots = np.ascontiguousarray(
        np.vstack([cot_0, cot_1, cot_2]).T)
    cdef np.ndarray[unsigned, ndim=1, mode='c'] i_sparse = np.zeros(
        [self.n_halfedges*2],dtype=np.uint32)
    cdef np.ndarray[unsigned, ndim=1, mode='c'] j_sparse = np.zeros(
        [self.n_halfedges*2],dtype=np.uint32)
    cdef np.ndarray[double,   ndim=1, mode='c'] v_sparse = np.zeros(
        [self.n_halfedges*2])
    self.thisptr.cotangent_laplacian(&i_sparse[0], &j_sparse[0], &v_sparse[0], &cots[0,0])
    L_c = sparse.coo_matrix((v_sparse, (i_sparse, j_sparse)))
    # we return the negitive -> switch
    return -0.5*sparse.csc_matrix(L_c)

  def reduce_tri_scalar_per_vertex_to_vertices(self,
      np.ndarray[double, ndim=2, mode="c"] triangle_scalar not None):
    cdef np.ndarray[double, ndim=1, mode='c'] vertex_scalar = np.zeros(self.n_vertices)
    self.thisptr.reduce_tri_scalar_per_vertex_to_vertices(&triangle_scalar[0,0], &vertex_scalar[0])
    return vertex_scalar

  def reduce_tri_scalar_to_vertices(self,
      np.ndarray[double, ndim=1, mode="c"] triangle_scalar not None):
    cdef np.ndarray[double, ndim=1, mode='c'] vertex_scalar = np.zeros(self.n_vertices)
    self.thisptr.reduce_tri_scalar_to_vertices(&triangle_scalar[0], &vertex_scalar[0])
    return vertex_scalar

  # --- HELPER ROUTINES for generating cached values ----- #

  def _calculate_edge_index(self):
    cdef np.ndarray[unsigned, ndim=2, mode='c'] edge_index = np.zeros(
        [self.n_edges, 2], dtype=np.uint32)
    self.thisptr.generate_edge_index(&edge_index[0,0])
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

