# distutils: language = c
# distutils: sources = ./pybug/transform/fastpwa/pwa.c
# distutils: extra_compile_args = -std=c99


from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
cimport numpy as np
import cython
cimport cython

cdef extern from "./fastpwa/pwa.h":
    ctypedef struct TriangleCollection:
        pass

    ctypedef struct AlphaBetaIndex:
        pass

    TriangleCollection initTriangleCollection(double *vertices,
                                              unsigned int *trilist,
                                              unsigned int n_triangles)

    void arrayAlphaBetaIndexForPoints(AlphaBetaIndex **hashMap,
                                      TriangleCollection *tris,
                                      double *points,
                                      unsigned int n_points, int *indexes,
                                      double *alphas, double *betas)
    void clearCacheAndDelete(AlphaBetaIndex **hashMap)
    void deleteTriangleCollection(TriangleCollection *tris)

cdef class FastPiecewiseAffine:
    cdef TriangleCollection tris
    cdef AlphaBetaIndex *hashMap

    def __cinit__(self,
                  np.ndarray[double, ndim=2, mode="c"] points not None,
                  np.ndarray[unsigned, ndim=2, mode="c"] trilist not None):
        hashMap = NULL
        if points.shape[1] != 2:
            pass
        self.tris =  initTriangleCollection(&points[0,0], &trilist[0,0],
                                            trilist.shape[0])


    def __dealloc__(self):
        deleteTriangleCollection(&self.tris)
        clearCacheAndDelete(&self.hashMap)

    def alpha_beta_index(self, np.ndarray[double, ndim=2, mode="c"] points
                         not None):
        cdef np.ndarray[double, ndim=1, mode='c'] alphas = \
            np.zeros(points.shape[0])
        cdef np.ndarray[double, ndim=1, mode='c'] betas = \
            np.zeros(points.shape[0])
        cdef np.ndarray[int, ndim=1, mode='c'] indexes = \
            np.zeros(points.shape[0], dtype=np.int32)
        arrayAlphaBetaIndexForPoints(&self.hashMap, &self.tris, &points[0,0],
                                     points.shape[0], &indexes[0],
                                     &alphas[0], &betas[0])
        return alphas, betas, indexes
