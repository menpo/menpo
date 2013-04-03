# distutils: language = c++
# distutils: sources = ./pybug/transform/geodesics/cpp/exact/kirsanov_geodesic_wrapper.cpp

import numpy as np
cimport numpy as np
import cython
cimport cython

# externally declare the exact geodesic code
cdef extern from "./cpp/exact/kirsanov_geodesic_wrapper.h":
    cdef cppclass KirsanovGeodesicWrapper:
        KirsanovGeodesicWrapper(double* points, unsigned n_vertices,
                unsigned* tri_index, unsigned n_triangles) except +
        void all_exact_geodesics_from_source_vertices(unsigned* source_vertices,
                unsigned n_sources, double* phi, unsigned* best_source)
        #void all_dijkstra_geodesics_from_source_vertices(
        #        unsigned* source_vertices, unsigned n_sources, double* phi,
        #        unsigned* best_source)
        #void all_subdivision_geodesics_from_source_vertices(
        #        unsigned* source_vertices, unsigned n_sources,
        #        double* phi, unsigned* best_source,
        #        unsigned subdivision_level)

cdef class KirsanovGeodesics:
    cdef KirsanovGeodesicWrapper* kirsanovptr
    cdef int n_points
    cdef int n_tris

    def __cinit__(self, np.ndarray[double, ndim=2, mode="c"] points not None,
            np.ndarray[unsigned, ndim=2, mode="c"] trilist not None, **kwargs):
        self.n_points = points.shape[0]
        self.n_tris = trilist.shape[0]
        self.kirsanovptr = new KirsanovGeodesicWrapper(
                &points[0,0], points.shape[0],
                &trilist[0,0], trilist.shape[0])

    def __dealloc__(self):
        del self.kirsanovptr

    def geodesics(self, source_vertices, method='exact'):
        cdef np.ndarray[unsigned, ndim=1, mode='c'] np_sources = np.array(
                source_vertices, dtype=np.uint32)
        cdef np.ndarray[double, ndim=1, mode='c'] phi = np.zeros(
                [self.n_points])
        cdef np.ndarray[unsigned, ndim=1, mode='c'] best_source = np.zeros(
                [self.n_points],dtype=np.uint32)
        if method == 'exact':
            self.kirsanovptr.all_exact_geodesics_from_source_vertices(
                    &np_sources[0], np_sources.size, &phi[0], &best_source[0])
        #elif method == 'dijkstra':
        #    self.kirsanovptr.all_dijkstra_geodesics_from_source_vertices(&np_sources[0],
        #            np_sources.size, &phi[0], &best_source[0])
        #elif method == 'subdivision':
        #    self.kirsanovptr.all_subdivision_geodesics_from_source_vertices(
        #            &np_sources[0], np_sources.size, &phi[0], &best_source[0], 3)
        else:
            print "The '" + `method` + "' method for calculating exact geodesics \
                    is not understood (only 'exact' can be used at this time)"
            return None
        geodesic = {}
        geodesic['phi'] = phi
        geodesic['best_source'] = best_source
        return geodesic
    
