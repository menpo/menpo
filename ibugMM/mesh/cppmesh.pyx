# distutils: language = c++
# distutils: sources = ./ibugMM/mesh/mesh.cpp ./ibugMM/mesh/vertex.cpp ./ibugMM/mesh/halfedge.cpp ./ibugMM/mesh/vec3.cpp ./ibugMM/mesh/triangle.cpp


import numpy as np
cimport numpy as np

cdef extern from "mesh.h":
  cdef cppclass Mesh:
    Mesh(double   *coords,      unsigned n_coords,
         unsigned *coordsIndex, unsigned n_triangles) except +
    unsigned n_triangles, n_coords
    void verifyAttachements()
    void calculateLaplacianOfScalar()
    double* vertexScalar
    double* vertexVec3
    double* triangleScalar
    double* triangleVec3

cdef class CppMesh:
  cdef Mesh* thisptr
  cdef np.ndarray vertex_scalar
  cdef np.ndarray vertex_vector
  cdef np.ndarray triangle_scalar
  cdef np.ndarray triangle_vector

  def __cinit__(self, np.ndarray[double,   ndim=2, mode="c"] coords      not None, 
                      np.ndarray[unsigned, ndim=2, mode="c"] coordsIndex not None, **kwargs):
    
    self.coords = coords;
    self.coordsIndex = coordsIndex
    self.thisptr = new Mesh(     &coords[0,0],      coords.shape[0], 
                            &coordsIndex[0,0], coordsIndex.shape[0])

  def __init__(self, **kwargs):
    self.vertex_scalar = np.zeros([self.n_coords])
    self.vertex_vector = np.zeros([self.n_coords,3])
    self.triangle_scalar = np.zeros([self.n_triangles])
    self.triangle_vector = np.zeros([self.n_triangles,3])
    self._set_vertex_scalar(self.vertex_scalar)
    self._set_vertex_vector(self.vertex_vector)
    self._set_triangle_scalar(self.triangle_scalar)
    self._set_triangle_vector(self.triangle_vector)

  def __dealloc__(self):
    del self.thisptr

  @property
  def n_coords(self):
    return self.thisptr.n_coords

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles

  def calculateLaplacianOfScalar(self):
    self.thisptr.calculateLaplacianOfScalar()

 
  def _set_vertex_scalar(self, np.ndarray[double, ndim=1, mode="c"] 
                               vertex_scalar not None):
    if vertex_scalar.shape[0] != self.n_coords:
      raise Exception('trying to attach a vertex scalar of incorrect dimensionality')
    else:
      self.thisptr.vertexScalar = &vertex_scalar[0]

  def _set_vertex_vector(self, np.ndarray[double, ndim=2, mode="c"] 
                               vertex_vector not None):
    if vertex_vector.shape[0] != self.n_coords or vertex_vector.shape[1] != 3:
      raise Exception('trying to attach a vertex vector of incorrect dimensionality')
    else:
      self.thisptr.vertexVec3 = &vertex_vector[0,0]

  def _set_triangle_scalar(self, np.ndarray[double, ndim=1, mode="c"] 
                               triangle_scalar not None):
    if triangle_scalar.shape[0] != self.n_triangles:
      raise Exception('trying to attach a triangle scalar of incorrect dimensionality')
    else:
      self.thisptr.triangleScalar = &triangle_scalar[0]

  def _set_triangle_vector(self, np.ndarray[double, ndim=2, mode="c"] 
                               triangle_vector not None):
    if triangle_vector.shape[0] != self.n_triangles or triangle_vector.shape[1] != 3:
      raise Exception('trying to attach a triangle vector of incorrect dimensionality')
    else:
      self.thisptr.triangleVec3 = &triangle_vector[0,0]

  def verify_attachments(self):
    self.thisptr.verifyAttachements()

