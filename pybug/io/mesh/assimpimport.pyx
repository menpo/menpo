# distutils: language = c++
# distutils: sources = ./pybug/io/mesh/cpp/assimputils.cpp ./pybug/io/mesh/cpp/assimpwrapper.cpp
# distutils: libraries = assimp
from libcpp.string cimport string
import numpy as np
cimport numpy as np


# externally declare the C++ classes
cdef extern from "./cpp/assimpwrapper.h":
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
    def _trimesh_index(self):
        return self.thisptr.trimesh_index

    @property
    def n_meshes(self):
        return self.thisptr.n_meshes()

    @property
    def n_points(self):
        return self.thisptr.n_points(self._trimesh_index)

    @property
    def n_tris(self):
        return self.thisptr.n_tris(self._trimesh_index)

    @property
    def n_tcoord_sets(self):
        return self.thisptr.n_tcoord_sets(self._trimesh_index)

    @property
    def texture_path(self):
        return self.thisptr.texture_path()

    @property
    def points(self):
        cdef np.ndarray[double, ndim=2, mode='c'] points = \
            np.empty([self.n_points, 3])
        self.thisptr.import_points(self._trimesh_index, &points[0, 0])
        return points

    @property
    def trilist(self):
        cdef np.ndarray[unsigned int, ndim=2, mode='c'] trilist = \
            np.empty([self.n_tris, 3], dtype=np.uint32)
        self.thisptr.import_trilist(self._trimesh_index, &trilist[0, 0])
        return trilist

    @property
    def tcoords(self):
        cdef np.ndarray[double, ndim=2, mode='c'] tcoords = \
            np.empty([self.n_points, 2])
        self.thisptr.import_tcoords(self._trimesh_index, 0, &tcoords[0, 0])
        return tcoords
