#include "pwa.h"

#include <stdio.h>
#include <stdlib.h>
#include "uthash.h"

//
// ----- POINT -----
//
Point initPoint(double *points)
{
  Point p = {*points, *(points + 1)};
  return p;
}

void pointPrint(Point p)
{
  printf("x=%.2f, y=%.2f", p.x, p.y);
}

Point pointAdd(Point p1, Point p2)
{
  Point p3 = {p1.x + p2.x, p1.y + p2.y};
  return p3;
}

Point pointSubtract(Point p1, Point p2)
{
  Point p3 = {p1.x - p2.x, p1.y - p2.y};
  return p3;
}

double pointDot(Point p1, Point p2)
{
  return p1.x * p2.x + p1.y * p2.y;
}

//
// ----- TRIANGLE -----
//
Triangle initTriangle(unsigned int *indices, double *vertices)
{
  Triangle tri;
  tri.i = initPoint(&vertices[*(indices) * 2]);
  tri.j = initPoint(&vertices[*(indices + 1) * 2]);
  tri.k = initPoint(&vertices[*(indices + 2) * 2]);
  return tri;
}

void trianglePrint(Triangle t)
{
  printf("i: ");
  pointPrint(t.i);
  printf("\nj: ");
  pointPrint(t.j);
  printf("\nk: ");
  pointPrint(t.k);
  printf("\n");
}

void alphaBetaForTriangle(Triangle t, Point p, double *alpha, double *beta)
{
  Point ip = pointSubtract(p, t.i);
  Point ij = pointSubtract(t.j, t.i);
  Point ik = pointSubtract(t.k, t.i);
  double dot_jj = pointDot(ij, ij);
  double dot_kk = pointDot(ik, ik);
  double dot_jk = pointDot(ij, ik);
  double dot_pj = pointDot(ip, ij);
  double dot_pk = pointDot(ip, ik);
  double d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk);
  *alpha = (dot_kk * dot_pj - dot_jk * dot_pk) * d;
  *beta = (dot_jj * dot_pk - dot_jk * dot_pj) * d;
}

//
// ----- TRIANGLECOLLECTION -----
//
TriangleCollection initTriangleCollection(double *vertices, unsigned int *trilist,
                                          unsigned int n_triangles)
{
  unsigned int i;
  TriangleCollection tris;
  tris.n_triangles = n_triangles;
  tris.triangles = (Triangle *)malloc(n_triangles * sizeof(Triangle));
  for (i = 0; i < n_triangles; i++) {
    tris.triangles[i] = initTriangle(&trilist[i * 3], vertices);
  }
  return tris;
}

void deleteTriangleCollection(TriangleCollection *tris)
{
  free(tris->triangles);
}

void containingTriangleAndAlphaBetaForPoint(TriangleCollection *tris, Point p,
                                            int *index, double *alpha, double *beta)
{
  unsigned int i;
  *index = -1; // no matching triangle
  for (i = 0; i < tris->n_triangles; i++) {
    alphaBetaForTriangle(tris->triangles[i], p, alpha, beta);
    //printf("Triangle %u\n", i);
    //trianglePrint(tris.triangles[i]);
    //printf("has alpha=%1.2f, beta=%1.2f\n", *alpha, *beta);
    if (*alpha >= 0 && *beta >= 0 && *alpha + *beta <= 1.0) {
      //printf(" This triangle contains the point ");
      //pointPrint(p);
      //printf("\n\n");
      *index = (int)i;
      return;
    } else {
      //printf(" This triangle does not contain the point ");
      //pointPrint(p);
      //printf("\n\n");
    }
  }
}

//
// ----- HASHMAP -----
//
AlphaBetaIndex* retrieveAlphaBetaFromCache(AlphaBetaIndex **hash, Point queryPoint)
{
  AlphaBetaIndex *resultInHash = NULL;
  // check to see if there is already this result in the hash
  HASH_FIND(hh, *hash, &queryPoint, sizeof(Point), resultInHash);
  return resultInHash;
}

// should only be called after retrieveAlphaBetaFromCache has returned NULL
void addAlphaBetaIndexToCache(AlphaBetaIndex **hash, Point queryPoint, int index, double alpha, double beta)
{
  // dynamically allocate a new result object
  AlphaBetaIndex *result;
  result = (AlphaBetaIndex *)malloc(sizeof(AlphaBetaIndex));
  memset(result, 0, sizeof(AlphaBetaIndex));
  result->queryPoint = queryPoint;
  result->index = index;
  result->alpha = alpha;
  result->beta = beta;
  HASH_ADD(hh, *hash, queryPoint, sizeof(Point), result);
}

void cachedAlphaBetaIndexForPointInTriangleCollection(AlphaBetaIndex **hash, TriangleCollection *tris, Point point,
                                                      int *index, double *alpha, double *beta)
{
  // check to see if the point is in the hashmap
  AlphaBetaIndex *cachedResult = retrieveAlphaBetaFromCache(hash, point);
  if (cachedResult) {
    //printf("cache hit\n");
    *alpha = cachedResult->alpha;
    *beta = cachedResult->beta;
    *index = cachedResult->index;
  } else {
    //printf("cache miss\n");
    // no entry in the cache - calculate the alpha/beta and cache it
    containingTriangleAndAlphaBetaForPoint(tris, point, index, alpha, beta);
    addAlphaBetaIndexToCache(hash, point, *index, *alpha, *beta);
  }
}

void arrayCachedAlphaBetaIndexForPoints(AlphaBetaIndex **hash, TriangleCollection *tris, double *points, unsigned int n_points,
                                  int *indexes, double *alphas, double *betas)
{
  unsigned int i;
  for (i = 0; i < n_points; i++) {
    // build a point object
    Point queryPoint = initPoint(points + i * 2);
    cachedAlphaBetaIndexForPointInTriangleCollection(hash, tris, queryPoint,
                                                     indexes + i, alphas + i, betas + i);
  }
}

void arrayAlphaBetaIndexForPoints(TriangleCollection *tris, double *points, unsigned int n_points,
                                  int *indexes, double *alphas, double *betas)
{
  unsigned int i;
  for (i = 0; i < n_points; i++) {
    // build a point object
    Point queryPoint = initPoint(points + i * 2);
    containingTriangleAndAlphaBetaForPoint(tris, queryPoint, indexes + i, alphas + i, betas + i);
  }
}

void clearCacheAndDelete(AlphaBetaIndex **hash)
{
  AlphaBetaIndex *currentResult, *tmp;
  HASH_ITER(hh, *hash, currentResult, tmp) {
    HASH_DEL(*hash, currentResult);  /* delete; users advances to next */
    free(currentResult);            /* optional- if you want to free  */
  }
}

