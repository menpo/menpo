#pragma once
#include "uthash.h"

typedef struct {
  double x;
  double y;
} Point;

Point initPoint(double *points);
void pointPrint(Point p);
Point pointAdd(Point p1, Point p2);
Point pointSubtract(Point p1, Point p2);
double pointDot(Point p1, Point p2);

typedef struct {
  Point i;
  Point j;
  Point k;
} Triangle;

Triangle initTriangle(unsigned int *indices, double *vertices);
void trianglePrint(Triangle t);
void alphaBetaForTriangle(Triangle t, Point p, double *alpha, double *beta);

typedef struct {
  Triangle *triangles;
  unsigned int n_triangles;
} TriangleCollection;

TriangleCollection initTriangleCollection(double *vertices, unsigned int *trilist,
                                          unsigned int n_triangles);
void deleteTriangleCollection(TriangleCollection *tris);
void containingTriangleAndAlphaBetaForPoint(TriangleCollection *tris, Point p,
                                           int *index, double *alpha, double *beta);

typedef struct {
  Point queryPoint;
  double alpha;
  double beta;
  int index;
  UT_hash_handle hh;
} AlphaBetaIndex;

AlphaBetaIndex* retrieveAlphaBetaFromCache(AlphaBetaIndex **hash, Point queryPoint);
// should only be called after retrieveAlphaBetaFromCache has returned NULL
void addAlphaBetaIndexToCache(AlphaBetaIndex **hash, Point queryPoint, int index, double alpha, double beta);
void cachedAlphaBetaIndexForPointInTriangleCollection(AlphaBetaIndex **hash, TriangleCollection *tris, Point point,
                                                      int *index, double *alpha, double *beta);
void arrayCachedAlphaBetaIndexForPoints(AlphaBetaIndex **hash, TriangleCollection *tris,
                                  double *points, unsigned int n_points,
                                  int *indexes, double *alphas, double *betas);
void arrayAlphaBetaIndexForPoints(TriangleCollection *tris,
                                  double *points, unsigned int n_points,
                                  int *indexes, double *alphas, double *betas);
void arrayMapForPointsAndTargetPoints(AlphaBetaIndex **hash, TriangleCollection *sourceTris,
                                  TriangleCollection *targetTris, double *points, unsigned int n_points,
                                  double *mappedPoints);
void clearCacheAndDelete(AlphaBetaIndex **hash);

