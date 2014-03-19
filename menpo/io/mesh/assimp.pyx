# distutils: language = c++
# distutils: sources = ./menpo/io/mesh/cpp/assimpwrapper.cpp
# distutils: libraries = assimp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np


# externally declare the C++ classes
cdef extern from "./cpp/assimpwrapper.h":

    cdef string NO_TEXTURE_PATH

    cdef cppclass AssimpImporter:
        AssimpImporter(string path) except +IOError
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
    r"""
    Wrap the C++-assimp importer. Can import multiple meshes per file type.

    Parameters
    ----------
    path : string
        Absolute file path of the mesh.
    """
    cdef AssimpImporter* importer
    cdef AssimpScene* scene
    cdef public list meshes

    def __cinit__(self, string path):
        self.meshes = []
        self.filepath = path

    def build_scene(self):
        r"""
        Builds the scene in assimp and creates a TriMesh importer for each
        mesh.
        """
        self.importer = new AssimpImporter(self.filepath)
        self.scene = self.importer.get_scene()
        for i in range(self.n_meshes):
            if self.scene.meshes[i].is_trimesh():
                self.meshes.append(AITriMeshImporter(self, i))

    def __dealloc__(self):
        del self.importer

    @property
    def n_meshes(self):
        r"""
        The number of meshes found.

        :type: int
        """
        return self.scene.n_meshes()

    @property
    def assimp_texture_path(self):
        r"""
        The relative texture filepath found by assimp.

        :type: string or ``None``
        """
        if self.scene.texture_path() == NO_TEXTURE_PATH:
            return None
        else:
            return self.scene.texture_path()


cdef class AITriMeshImporter:
    r"""
    Exposes properties of the mesh found by assimp. Technically the mesh has
    already been built and this class exposes it's attributes.

    Parameters
    ----------
    wrapper : :class:`AIImporter`
        The main importer that contains the meshes
    mesh_index : unsigned int
        The index in to the main importer for this particular mesh.
    """
    cdef AssimpMesh* thisptr

    def __cinit__(self, AIImporter wrapper, unsigned int mesh_index):
        self.thisptr = wrapper.scene.meshes[mesh_index]

    @property
    def n_points(self):
        r"""
        Number of points in the mesh

        :type: int
        """
        return self.thisptr.n_points()

    @property
    def n_tris(self):
        r"""
        Number of triangles in the triangle list.

        :type: int
        """
        return self.thisptr.n_faces()

    @property
    def n_tcoord_sets(self):
        r"""
        Number of texture coordinates for the mesh.

        :type: int
        """
        return self.thisptr.n_tcoord_sets()

    @property
    def points(self):
        r"""
        The array of points.

        :type: (``n_points``, 3) c-contiguous double ndarray
        """
        cdef np.ndarray[double, ndim=2, mode='c'] points = \
            np.empty([self.n_points, 3])
        self.thisptr.points(&points[0, 0])
        return points

    @property
    def trilist(self):
        r"""
        The triangle list.

        :type: (``n_tris``, 3) c-contiguous unsigned int ndarray
        """
        cdef np.ndarray[unsigned int, ndim=2, mode='c'] trilist = \
            np.empty([self.n_tris, 3], dtype=np.uint32)
        self.thisptr.trilist(&trilist[0, 0])
        return trilist

    @property
    def tcoords(self):
        r"""
        The texture coordinates.

        :type: (``n_points``, 2) c-contiguous unsigned int ndarray
        """
        cdef np.ndarray[double, ndim=2, mode='c'] tcoords = \
            np.empty([self.n_points, 2])
        self.thisptr.tcoords(0, &tcoords[0, 0])
        return tcoords

    @property
    def colour_per_vertex(self):
        # TODO: Support colour per vertex in assimp
        return None

    def __str__(self):
        msg = 'n_points: %d\n' % self.n_points
        msg += 'n_tris:   %d\n' % self.n_tris
        msg += 'n_tcoord_sets %d' % self.n_tcoord_sets
        return msg
