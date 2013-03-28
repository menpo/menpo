# distutils: language = c++
# distutils: sources = ./pybug/shape/representation/mesh/cpp/mesh.cpp ./pybug/shape/representation/mesh/cpp/vertex.cpp ./pybug/shape/representation/mesh/cpp/halfedge.cpp ./pybug/shape/representation/mesh/cpp/triangle.cpp

from libcpp.vector cimport vector
from libcpp.set        cimport set
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
            raise MeshConstructionError("A CppTriangleMesh can only be in 3 "\
                   " dimensions (attemping with " + str(points.shape[1]) + ")")
        self.thisptr = new Mesh(&trilist[0,0], trilist.shape[0], points.shape[0])

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
        assert n_vertex >= 0 and n_vertex < self.thisptr.n_vertices
        deref(self.thisptr.vertices[n_vertex]).status()

    def tri_status(self, n_triangle):
        assert n_triangle >= 0 and n_triangle < self.thisptr.n_triangles
        deref(self.thisptr.triangles[n_triangle]).status()

