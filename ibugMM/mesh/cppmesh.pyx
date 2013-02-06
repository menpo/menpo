# distutils: language = c++
# distutils: sources = ./ibugMM/mesh/mesh.cpp ./ibugMM/mesh/vertex.cpp ./ibugMM/mesh/halfedge.cpp ./ibugMM/mesh/vec3.cpp ./ibugMM/mesh/triangle.cpp

from libcpp.vector cimport vector
from libcpp.set    cimport set
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, linalg
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
                                    double* v_sparse,   double* vertex_areas)
    void calculateGradient(double* v_scalar_field, double* t_vector_gradient)
    void calculateDivergence(double* t_vector_field, double* v_scalar_divergence)
    void verifyMesh()
    unsigned n_half_edges
    unsigned n_full_edges
    double meanEdgeLength()
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
    self.coordsIndex = coordsIndex
    self.thisptr = new Mesh(     &coords[0,0],      coords.shape[0], 
                            &coordsIndex[0,0], coordsIndex.shape[0])
    # cached values
    self._mean_edge_length = None
    self._lagrangian_c = None
    self._area_matrix  = None
    self.u_t_solver = None
    self.phi_solver = None

  def __dealloc__(self):
    del self.thisptr

  @property
  def n_coords(self):
    return self.thisptr.n_coords

  @property
  def mean_edge_length(self):
    if self._mean_edge_length is None:
      self._mean_edge_length = self.thisptr.meanEdgeLength()
    return self._mean_edge_length

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles

  @property
  def n_full_edges(self):
    return self.thisptr.n_full_edges

  @property
  def n_half_edges(self):
    return self.thisptr.n_half_edges

  def verify_mesh(self):
    self.thisptr.verifyMesh()

  def geodesic(self):
    cdef np.ndarray[double, ndim=1, mode='c'] u_0 = np.zeros(self.n_coords)
    u_0[3000] = 1
    cdef np.ndarray[double, ndim=1, mode='c'] u_t = np.zeros(self.n_coords)
    L_c,A = self.laplacian_operator()
    grad_u = self.gradient(u_0)
    div    = self.divergence(grad_u)
    return L_c, A, grad_u, div

  def vertex_status(self, n_vertex):
    assert n_vertex >= 0 and n_vertex < self.n_coords
    deref(self.thisptr.vertices[n_vertex]).printStatus()

  def triangle_status(self, n_triangle):
    assert n_triangle >= 0 and n_triangle < self.n_triangles
    deref(self.thisptr.triangles[n_triangle]).printStatus()

  @property
  def neg_sd_lagrangian_c(self):
    self._regenerate_neg_sd_laplacian_c_and_area_matrix()
    return self._lagrangian_c

  @property
  def area_matrix(self):
    self._regenerate_neg_sd_laplacian_c_and_area_matrix()
    return self._area_matrix

  def _regenerate_neg_sd_laplacian_c_and_area_matrix(self):
    if self._lagrangian_c is None or self._area_matrix is None:
      L_c, A = self._pos_sd_laplacian_c_and_area_matrix()
      # calculate the NEGITIVE semidefinite laplacian operator
      self._lagrangian_c = -1.0*L_c
      self._area_matrix = A

  def _pos_sd_laplacian_c_and_area_matrix(self):
    cdef np.ndarray[unsigned, ndim=1, mode='c'] i_sparse = np.zeros(
        [self.thisptr.n_half_edges*2],dtype=np.uint32)
    cdef np.ndarray[unsigned, ndim=1, mode='c'] j_sparse = np.zeros(
        [self.thisptr.n_half_edges*2],dtype=np.uint32)
    cdef np.ndarray[double,   ndim=1, mode='c'] v_sparse = np.zeros(
        [self.thisptr.n_half_edges*2])
    cdef np.ndarray[double, ndim=1, mode='c'] vertex_areas = np.zeros(self.n_coords)
    self.thisptr.calculateLaplacianOperator(&i_sparse[0], &j_sparse[0], &v_sparse[0], &vertex_areas[0])
    L_c = coo_matrix((v_sparse, (i_sparse, j_sparse)))
    A_i = np.arange(self.n_coords)
    A = coo_matrix((vertex_areas, (A_i,A_i)))
    return csc_matrix(L_c), csc_matrix(A)

  def gradient(self, np.ndarray[double, ndim=1, mode="c"] v_scalar_field not None):
    """
    Return the gradient (per face) of the per vertex scalar field 

    C++ effects:
    vertex_scalar   - the scalar field value (per vertex) that we are taking
                      the gradient of.
    triangle_vector - the resulting gradient (per triangle)
    :param s_field: scalar field value per vertex
    :type s_field: ndarray[1,n_coords]
    :return: Gradient evaluated over each triangle
    :rtype: ndarray[float]
   
    """
    cdef np.ndarray[double, ndim=2,mode ='c'] t_vector_gradient = np.zeros(
        [self.n_triangles,3])
    self.thisptr.calculateGradient(&v_scalar_field[0], &t_vector_gradient[0,0])
    return t_vector_gradient

  def divergence(self, np.ndarray[double, ndim=2, mode="c"] t_vector_field not None):
    """
    Return the divergence (per vertex) of the field stored in triangle_vector.

    C++ effects:
    triangle_vector - input
    vertex_scalar   - result storage

    :return: Gradient evaluated over each triangle
    :rtype: ndarray[float]
   
    """
    cdef np.ndarray[double, ndim=1, mode='c'] v_scalar_divergence = np.zeros(self.n_coords)
    self.thisptr.calculateDivergence(&t_vector_field[0,0], &v_scalar_divergence[0])
    return v_scalar_divergence

  def heat_geodesics(self, source_vertices):
    A   = self.area_matrix
    L_c = self.neg_sd_lagrangian_c
    t   = self.mean_edge_length
    u_0 = np.zeros(self.n_coords)
    u_0[source_vertices] = 1.0
    if self.u_t_solver is None:
      print 'solving for the first time'
      self.u_t_solver = linalg.factorized(A - t*L_c)
    u_t = self.u_t_solver(u_0)
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
    if self.phi_solver is None:
      print 'solving phi for the first time'
      self.phi_solver = linalg.factorized(L_c)
    phi = self.phi_solver(div_X)
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

