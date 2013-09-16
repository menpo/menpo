# distutils: language = c
# distutils: sources = ./pybug/transform/fastpwa/pwa.c
# distutils: extra_compile_args = -std=c99

import numpy as np
cimport numpy as np

cdef extern from "./fastpwa/pwa.h":
    ctypedef struct TriangleCollection:
        pass

    ctypedef struct AlphaBetaIndex:
        pass

    TriangleCollection initTriangleCollection(double *vertices,
                                              unsigned int *trilist,
                                              unsigned int n_triangles)

    void arrayCachedAlphaBetaIndexForPoints(AlphaBetaIndex **hashMap,
                                      TriangleCollection *tris,
                                      double *points,
                                      unsigned int n_points, int *indexes,
                                      double *alphas, double *betas)
    void arrayAlphaBetaIndexForPoints(TriangleCollection *tris,
                                      double *points,
                                      unsigned int n_points, int *indexes,
                                      double *alphas, double *betas)
    void clearCacheAndDelete(AlphaBetaIndex **hashMap)
    void deleteTriangleCollection(TriangleCollection *tris)

cdef class CLookupPWA:
    cdef TriangleCollection tris
    cdef AlphaBetaIndex *hashMap
    cdef n_tris

    def __cinit__(self,
                  double[:, ::1] points not None,
                  np.ndarray[unsigned, ndim=2, mode="c"] trilist not None):
        hashMap = NULL
        if points.shape[1] != 2:
            pass
        self.n_tris = trilist.shape[0]
        self.tris =  initTriangleCollection(&points[0,0], &trilist[0,0],
                                            trilist.shape[0])

    def _init_source_triangles(self,
                  double[:, ::1] points not None,
                  np.ndarray[unsigned, ndim=2, mode="c"] trilist not None):
        hashMap = NULL
        if points.shape[1] != 2:
            raise Exception
        elif points.shape[0] != self.n_tris:
            raise Exception
        self.tris =  initTriangleCollection(&points[0,0], &trilist[0,0],
                                            self.n_tris)

    def _init_target_triangles(self,
                  double[:, ::1] points not None,
                  np.ndarray[unsigned, ndim=2, mode="c"] trilist not None):
        self.target_tris = initTriangleCollection(&points[0, 0],
                                                  &trilist[0, 0],
                                                  trilist.shape[0])

    def __dealloc__(self):
        deleteTriangleCollection(&self.tris)
        clearCacheAndDelete(&self.hashMap)

    def index_alpha_beta(self, np.ndarray[double, ndim=2,
                                               mode="c"] points
                         not None):
        cdef np.ndarray[double, ndim=1, mode='c'] alphas = \
            np.zeros(points.shape[0], dtype=np.float64)
        cdef np.ndarray[double, ndim=1, mode='c'] betas = \
            np.zeros(points.shape[0], dtype=np.float64)
        cdef np.ndarray[int, ndim=1, mode='c'] indexes = \
            np.zeros(points.shape[0], dtype=np.int32)
        arrayCachedAlphaBetaIndexForPoints(&self.hashMap, &self.tris,
                                          &points[0,0],
                                     points.shape[0], &indexes[0],
                                     &alphas[0], &betas[0])
        return indexes, alphas, betas
