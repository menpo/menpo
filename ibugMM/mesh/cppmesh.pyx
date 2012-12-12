# distutils: language = c++
# distutils: sources = ./ibugMM/mesh/mesh.cpp ./ibugMM/mesh/vertex.cpp ./ibugMM/mesh/halfedge.cpp ./ibugMM/mesh/vec3.cpp ./ibugMM/mesh/triangle.cpp


from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
from scipy.sparse import coo_matrix
cimport numpy as np
import cython
cimport cython

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
    double   n_full_edges
    unsigned n_full_edges


cdef class CppMesh:
  cdef Mesh* thisptr

  def __cinit__(self, np.ndarray[double,   ndim=2, mode="c"] coords      not None, 
                      np.ndarray[unsigned, ndim=2, mode="c"] coordsIndex not None, **kwargs):
    self.coords = coords;
    self.coordsIndex = coordsIndex
    self.thisptr = new Mesh(     &coords[0,0],      coords.shape[0], 
                            &coordsIndex[0,0], coordsIndex.shape[0])

  def __dealloc__(self):
    del self.thisptr

  @property
  def n_coords(self):
    return self.thisptr.n_coords

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles
  @property
  def n_full_edges(self):
    return self.thisptr.n_full_edges

  def geodesic(self):
    cdef np.ndarray[double, ndim=1, mode='c'] u_0 = np.zeros(self.n_coords)
    cdef np.ndarray[double, ndim=1, mode='c'] u_t = np.zeros(self.n_coords)
    L_c,A = self.laplacian_operator()
    grad_u = self.gradient(u_0)
    div    = self.divergence(grad_u)
    return L_c, A, grad_u, div

  def laplacian_operator(self):
    cdef np.ndarray[unsigned, ndim=1, mode='c'] i_sparse = np.zeros(
        [self.thisptr.n_full_edges*2 + self.thisptr.n_coords],dtype=np.uint32)
    cdef np.ndarray[unsigned, ndim=1, mode='c'] j_sparse = np.zeros(
        [self.thisptr.n_full_edges*2 + self.thisptr.n_coords],dtype=np.uint32)
    cdef np.ndarray[double,   ndim=1, mode='c'] v_sparse = np.zeros(
        [self.thisptr.n_full_edges*2 + self.thisptr.n_coords])
    cdef np.ndarray[double, ndim=1, mode='c'] vertex_areas = np.zeros(self.n_coords)
    self.thisptr.calculateLaplacianOperator(&i_sparse[0], &j_sparse[0], &v_sparse[0], &vertex_areas[0])
    L_c = coo_matrix((v_sparse, (i_sparse, j_sparse)))
    A_i = np.arange(self.n_coords)
    A = coo_matrix((vertex_areas, (A_i,A_i)))
    return L_c, A

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

  #def _set_vertex_scalar(self, np.ndarray[double, ndim=1, mode="c"] 
  #                             vertex_scalar not None):
  #  if vertex_scalar.shape[0] != self.n_coords:
  #    raise Exception('trying to attach a vertex scalar of incorrect dimensionality')
  #  else:
  #    self.thisptr.vertexScalar = &vertex_scalar[0]

  #def _set_vertex_vector(self, np.ndarray[double, ndim=2, mode="c"] 
  #                             vertex_vector not None):
  #  if vertex_vector.shape[0] != self.n_coords or vertex_vector.shape[1] != 3:
  #    raise Exception('trying to attach a vertex vector of incorrect dimensionality')
  #  else:
  #    self.thisptr.vertexVec3 = &vertex_vector[0,0]

  #def _set_triangle_scalar(self, np.ndarray[double, ndim=1, mode="c"] 
  #                             triangle_scalar not None):
  #  if triangle_scalar.shape[0] != self.n_triangles:
  #    raise Exception('trying to attach a triangle scalar of incorrect dimensionality')
  #  else:
  #    self.thisptr.triangleScalar = &triangle_scalar[0]

  #def _set_triangle_vector(self, np.ndarray[double, ndim=2, mode="c"] 
  #                             triangle_vector not None):
  #  if triangle_vector.shape[0] != self.n_triangles or triangle_vector.shape[1] != 3:
  #    raise Exception('trying to attach a triangle vector of incorrect dimensionality')
  #  else:
  #    self.thisptr.triangleVec3 = &triangle_vector[0,0]
#
#  def _set_i_sparse(self, np.ndarray[unsigned, ndim=1, mode="c"] 
#                               i_sparse not None):
#    self.thisptr.i_sparse = &i_sparse[0]
#
#  def _set_j_sparse(self, np.ndarray[unsigned, ndim=1, mode="c"] 
#                               j_sparse not None):
#    self.thisptr.j_sparse = &j_sparse[0]
#
#  def _set_v_sparse(self, np.ndarray[double, ndim=1, mode="c"] 
#                               v_sparse not None):
#    self.thisptr.v_sparse = &v_sparse[0]

  #def verify_attachments(self):
  #  self.thisptr.verifyAttachements()

