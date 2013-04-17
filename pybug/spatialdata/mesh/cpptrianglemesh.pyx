# distutils: language = c++
# distutils: sources = ./pybug/spatialdata/mesh/cpp/mesh.cpp ./pybug/spatialdata/mesh/cpp/vertex.cpp ./pybug/spatialdata/mesh/cpp/halfedge.cpp ./pybug/spatialdata/mesh/cpp/triangle.cpp

from libcpp.vector cimport vector
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
cimport numpy as np
import cython
cimport cython

class MeshConstructionError(Exception):
    pass

# externally declare the C++ Mesh, Vertex, Triangle and HalfEdge classes
cdef extern from "./cpp/mesh.h":
    cdef cppclass Mesh:
        Mesh(unsigned *tri_index, unsigned n_tris, unsigned n_points) except +
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
        void reduce_tri_scalar_to_vertices(
                double* triangle_scalar, double* vertex_scalar)
        void reduce_tri_scalar_per_vertex_to_vertices(
                double* triangle_scalar_p_vert, double* vertex_scalar)

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

    def __cinit__(self, points,
            np.ndarray[unsigned, ndim=2, mode="c"] trilist not None):
        if points.shape[1] != 3:
            raise MeshConstructionError("A CppTriangleMesh can only be in 3 "
                   + "dimensions (attempting with " + str(points.shape[1]) +
                                        ")")
        self.thisptr = new Mesh(&trilist[0,0], trilist.shape[0],
                                points.shape[0])

    def __dealloc__(self):
        del self.thisptr

    @property
    def n_fulledges(self):
        return self.thisptr.n_fulledges

    @property
    def n_halfedges(self):
        return self.thisptr.n_halfedges

    @property
    def n_edges(self):
        return self.thisptr.n_halfedges - self.thisptr.n_fulledges

    def verify_mesh(self):
        self.thisptr.verify_mesh()

    def vertex_status(self, n_vertex):
        assert 0 <= n_vertex < self.thisptr.n_vertices
        deref(self.thisptr.vertices[n_vertex]).status()

    def tri_status(self, n_triangle):
        assert 0 <= n_triangle < self.thisptr.n_triangles
        deref(self.thisptr.triangles[n_triangle]).status()

    def reduce_tri_scalar_per_vertex_to_vertices(self,
            np.ndarray[double, ndim=2, mode="c"] tri_s not None):
        cdef np.ndarray[double, ndim=1, mode='c'] vert_s = \
            np.zeros(self.thisptr.n_vertices)
        self.thisptr.reduce_tri_scalar_per_vertex_to_vertices(
            &tri_s[0,0], &vert_s[0])
        return vert_s

    def reduce_tri_scalar_to_vertices(self,
            np.ndarray[double, ndim=1, mode="c"] triangle_scalar not None):
        cdef np.ndarray[double, ndim=1, mode='c'] vertex_scalar = \
            np.zeros(self.thisptr.n_vertices)
        self.thisptr.reduce_tri_scalar_to_vertices(&triangle_scalar[0],
            &vertex_scalar[0])
        return vertex_scalar

