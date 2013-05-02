# distutils: language = c++
# distutils: sources = ./pybug/io/mesh/cpp/assimputils.cpp ./pybug/io/mesh/cpp/assimpwrapper.cpp
# distutils: libraries = assimp
from libcpp.vector cimport vector
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.string cimport string
import numpy as np
cimport numpy as np
import cython
cimport cython


# externally declare the C++ Mesh, Vertex, Triangle and HalfEdge classes
cdef extern from "./cpp/assimp.h":
    cdef cppclass AssimpWrapper:
        AssimpWrapper(string path) except +
        unsigned int trimesh_index
        unsigned int n_meshes()
        unsigned int n_points(unsigned int mesh_no)
        unsigned int n_tris(unsigned int mesh_no)
        unsigned int n_tcoord_sets(unsigned int mesh_no)
        string texture_path()
        void import_points(unsigned int mesh_no, double* points)
        void import_trilist(unsigned int mesh_no, unsigned int* trilist)
        void import_tcoords(unsigned int mesh_no, int index, double* tcoords)

# Use the wrapper to build our PyAssimpImporter
cdef class PyAssimpImporter:
    cdef AssimpWrapper* thisptr

    def __cinit__(self, string path):
        self.thisptr = new AssimpWrapper(path)

    def __dealloc__(self):
        del self.thisptr

    @property
    def n_meshes(self):
        return self.thisptr.n_meshes()
"""
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
"""
