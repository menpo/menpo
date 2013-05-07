# distutils: language = c++
# distutils: sources = ./pybug/io/mesh/cpp/assimpwrapper.cpp
# distutils: libraries = assimp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np


# externally declare the C++ classes
cdef extern from "./cpp/assimpwrapper.h":

    cdef const string NO_TEXTURE_PATH

    cdef cppclass AssimpImporter:
        AssimpImporter(string path) except +
        AssimpScene* get_scene()

    cdef cppclass AssimpScene:
        vector[AssimpMesh*] meshes
        unsigned int n_meshes()
        string texture_path()

    cdef cppclass AssimpMesh:
        unsigned int n_points()
        unsigned int n_faces()
        unsigned int n_tcoord_sets()
        bool is_trimesh()
        bool is_pointcloud()
        void points(double* points)
        void trilist(unsigned int* trilist)
        void tcoords(int index, double* tcoords)


cdef class AIImporter:
    cdef AssimpImporter* importer
    cdef AssimpScene* scene
    cdef public list meshes

    def __cinit__(self, string path):
        self.importer = new AssimpImporter(path)
        self.scene = self.importer.get_scene()
        self.meshes = []
        for i in range(self.n_meshes):
            if self.scene.meshes[i].is_trimesh():
                self.meshes.append(AITriMeshImporter(self, i))

    def __dealloc__(self):
        del self.importer

    @property
    def n_meshes(self):
        return self.scene.n_meshes()

    @property
    def texture_path(self):
        if self.scene.texture_path() == NO_TEXTURE_PATH:
            return None
        else:
            return self.scene.texture_path()


cdef class AITriMeshImporter:
    cdef AssimpMesh* thisptr

    def __cinit__(self, AIImporter wrapper, unsigned int mesh_index):
        self.thisptr = wrapper.scene.meshes[mesh_index]

    @property
    def n_points(self):
        return self.thisptr.n_points()

    @property
    def n_tris(self):
        return self.thisptr.n_faces()

    @property
    def n_tcoord_sets(self):
        return self.thisptr.n_tcoord_sets()

    @property
    def points(self):
        cdef np.ndarray[double, ndim=2, mode='c'] points = \
            np.empty([self.n_points, 3])
        self.thisptr.points(&points[0, 0])
        return points

    @property
    def trilist(self):
        cdef np.ndarray[unsigned int, ndim=2, mode='c'] trilist = \
            np.empty([self.n_tris, 3], dtype=np.uint32)
        self.thisptr.trilist(&trilist[0, 0])
        return trilist

    @property
    def tcoords(self):
        cdef np.ndarray[double, ndim=2, mode='c'] tcoords = \
            np.empty([self.n_points, 2])
        self.thisptr.tcoords(0, &tcoords[0, 0])
        return tcoords

    def __str__(self):
        msg = 'n_points: %d\n' % self.n_points
        msg += 'n_tris:   %d\n' % self.n_tris
        msg += 'n_tcoord_sets %d' % self.n_tcoord_sets
        return msg
    