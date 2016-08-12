# distutils: language = c++
# distutils: sources = ./menpo/transform/piecewiseaffine/cpp/quadtree.cpp


import numpy as np
cimport numpy as cnp
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp cimport bool
import cython


cdef extern from "./cpp/quadtree.h":
    ctypedef struct BoundingBox:
        double min_y
        double min_x
        double max_y
        double max_x

    ctypedef struct QuadNode:
        unsigned long index
        BoundingBox bounding_box

    ctypedef struct QuadTree:
        BoundingBox bounding_box
        unsigned int depth
        unsigned int max_items
        unsigned int max_depth
        vector[QuadNode] nodes
        vector[QuadTree] children

    ctypedef struct AlphaBeta:
        double alpha
        double beta

    AlphaBeta point_in_triangle(const unsigned int i, const unsigned int j,
                                const unsigned int k, const double *vertices,
                                const double y, const double x)

    QuadTree build_quadtree(const unsigned long max_items,
                            const unsigned long max_depth,
                            const double *vertices,
                            const unsigned long n_vertices,
                            const unsigned int *trilist,
                            const unsigned long n_triangles) except +

    void dealloc_quadtree(QuadTree &qtree)

    void rect_intersect(const QuadTree &qtree, const BoundingBox bb,
                        set[unsigned long]& results)


cdef draw(QuadTree qtree, object bbs=None):
    cdef:
        QuadTree c

    if bbs is None:
        bbs = []
    for c in qtree.children:
        draw(c, bbs=bbs)
    bbs.append(((qtree.bounding_box.min_y, qtree.bounding_box.min_x),
                (qtree.bounding_box.max_y, qtree.bounding_box.max_x)))
    return bbs


cdef n_nodes(QuadTree qtree):
    cdef:
        unsigned long total = 0
        QuadTree c
    for c in qtree.children:
        total += n_nodes(c)
    total += qtree.nodes.size()
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef search(double[:, ::1] points, unsigned int[:, ::1] trilist,
            double[:, ::1] search_points, QuadTree &qtree,
            bool shortcut_lookup=True):
    cdef:
        unsigned int n_search_points = search_points.shape[0]
        cnp.ndarray[double, ndim=1, mode='c'] alphas = np.zeros(n_search_points,
                                                                dtype=np.float64)
        cnp.ndarray[double, ndim=1, mode='c'] betas = np.zeros(n_search_points,
                                                               dtype=np.float64)
        cnp.ndarray[int, ndim=1, mode='c'] indexes = np.zeros(n_search_points,
                                                              dtype=np.int32)
        set[unsigned long] results
        unsigned int qq = 0, i = 0, j = 0, k = 0
        double y = 0, x = 0
        unsigned long tt = 0
        long previous_tt = -1
        AlphaBeta ab
        BoundingBox bb

    # fill the arrays with the C results
    for qq in range(n_search_points):
        y, x = search_points[qq, 0], search_points[qq, 1]
        # Assume we have high locality in the query points and so look in the
        # same triangle as the last successful lookup first
        if shortcut_lookup and previous_tt != -1:
            ab = point_in_triangle(i, j, k, &points[0, 0], y, x)
            if ab.alpha >= 0 and ab.beta >= 0:
                alphas[qq] = ab.alpha
                betas[qq] = ab.beta
                indexes[qq] = previous_tt
            else:
                previous_tt = -1

        # If that failed, then we fall back to the quadtree
        # (or if shortcut_lookup is False)
        if not shortcut_lookup or previous_tt == -1:
            bb.min_y, bb.max_y = y, y
            bb.min_x, bb.max_x = x, x
            rect_intersect(qtree, bb, results)
            for tt in results:
                i = trilist[tt, 0]
                j = trilist[tt, 1]
                k = trilist[tt, 2]
                ab = point_in_triangle(i, j, k, &points[0, 0], y, x)
                if ab.alpha >= 0 and ab.beta >= 0:
                    alphas[qq] = ab.alpha
                    betas[qq] = ab.beta
                    indexes[qq] = tt
                    previous_tt = tt
                    break
            else:
                previous_tt = -1
                alphas[qq] = -1
                betas[qq] = -1
                indexes[qq] = -1
        results.clear()
    return indexes, alphas, betas


cdef class QuadTreePWA:
    cdef unsigned long n_tris
    cdef unsigned long n_points
    cdef double[:, ::1] points
    cdef unsigned int[:, ::1] trilist
    cdef bool shortcut_lookup
    cdef QuadTree _qtree

    def __cinit__(self,
                  double[:, ::1] points not None,
                  unsigned[:, ::1] trilist not None,
                  unsigned long max_items=5, unsigned long max_depth=1000,
                  bool shortcut_lookup=True):
        if points.shape[1] != 2:
            raise ValueError('Only 2D PointClouds are supported.')
        self.n_tris = trilist.shape[0]
        self.n_points = points.shape[0]
        self.points = points
        self.trilist = trilist
        self.shortcut_lookup = shortcut_lookup
        self._qtree = build_quadtree(max_items, max_depth,
                                     &self.points[0, 0], self.n_points,
                                     &self.trilist[0, 0], self.n_tris)

    def __dealloc__(self):
        dealloc_quadtree(self._qtree)

    def __reduce__(self):
        return self.__class__, (np.asarray(self.points),
                                np.asarray(self.trilist))

    def index_alpha_beta(self, double[:, ::1] points not None):
        return search(self.points, self.trilist, points, self._qtree,
                      shortcut_lookup=self.shortcut_lookup)

    def draw(self):
        from menpo.shape import bounding_box
        for bb in draw(self._qtree):
            bounding_box(*bb).view()

    def n_nodes(self):
        return n_nodes(self._qtree)
