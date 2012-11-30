# distutils: language = c++
# distutils: sources = mesh.cpp vertex.cpp halfedge.cpp vec3.cpp triangle.cpp

import numpy as np
cimport numpy as np

cdef extern from "mesh.h":
  cdef cppclass Mesh:
    Mesh(double   *coords,      unsigned n_coords,
         unsigned *coordsIndex, unsigned n_triangles) except +
    unsigned n_triangles

cdef class CppMesh:
  cdef Mesh* thisptr

  def __cinit__(self, np.ndarray[double,   ndim=2, mode="c"] coords      not None, 
                      np.ndarray[unsigned, ndim=2, mode="c"] coordsIndex not None):
    n_coords, n_triangles = coords.shape[0],coordsIndex.shape[0]
    self.thisptr = new Mesh(&coords[0,0],      n_coords, 
                            &coordsIndex[0,0], n_triangles)

  def __dealloc__(self):
    del self.thisptr

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles

