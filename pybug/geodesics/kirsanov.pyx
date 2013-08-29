# distutils: language = c++
# distutils: sources = ./pybug/geodesics/cpp/exact/kirsanov_geodesic_wrapper.cpp

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
    r"""
    Cython wrapper for the cpp class used to calculated the Kirsanov Geodesics.

    Parameters
    ----------
    points : (N, D) c-contiguous double ndarray
        The cartesian points
    trilist : (M, 3) c-contiguous unsigned ndarray
        The triangulation of the given points

    References
    ----------
    .. [1] Surazhsky, Vitaly, et al.
        "Fast exact and approximate geodesics on meshes."
        ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
    """
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
        r"""
        Calculate the geodesic distance for all points from the
        given ``source_indexes``.

        Parameters
        -----------
        source_vertices : (N,) c-contiguous unsigned ndarray
            List of indexes to calculate the geodesics for
        method : {'exact'}
            The method using to calculate the geodesics. Only the 'exact'
            method is currently supported

            Default: exact

        Returns
        -------
        TODO: document these?
        phi : UNKNOWN
        best_source : UNKNOWN

        Raises
        -------
        ValueError
            If the given method is not 'exact'
        """
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
            raise ValueError("The '" + `method` + "' method for calculating "
                            "geodesics is not understood "
                            "(only 'exact' can be used at this time)")

        geodesic = {'phi': phi, 'best_source': best_source}
        return geodesic
