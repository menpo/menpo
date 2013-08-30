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
  *alpha = (dot_jj * dot_pk - dot_jk * dot_pj) * d;
  *beta = (dot_kk * dot_pj - dot_jk * dot_pk) * d;
}

//
// ----- TRIANGLECOLLECTION -----
//
TriangleCollection initTriangleCollection(double *vertices, unsigned int *trilist,
                                          unsigned int n_triangles)
{
  TriangleCollection tris;
  tris.n_triangles = n_triangles;
  tris.triangles = (Triangle *)malloc(n_triangles * sizeof(Triangle));
  for (unsigned int i = 0; i < n_triangles; i++) {
    tris.triangles[i] = initTriangle(&trilist[i * 3], vertices);
  }
  return tris;
}

void deleteTriangleCollection(TriangleCollection *tris)
{
  free(tris->triangles);
}

int containingTriangleAndAlphaBetaForPoint(TriangleCollection tris, Point p,
                                           double *alpha, double *beta)
{
  int index = -1; // means no matching triangle
  for (unsigned int i = 0; i < tris.n_triangles; i++) {
    alphaBetaForTriangle(tris.triangles[i], p, alpha, beta);
    //printf("Triangle %u\n", i);
    //trianglePrint(tris.triangles[i]);
    //printf("has alpha=%1.2f, beta=%1.2f\n", *alpha, *beta);
    if (*alpha >= 0 && *beta >= 0 && *alpha + *beta <= 1.0) {
      //printf(" This triangle contains the point ");
      //pointPrint(p);
      //printf("\n\n");
      index = (int)i;
      break;
    } else {
      //printf(" This triangle does not contain the point ");
      //pointPrint(p);
      //printf("\n\n");
    }
  }
  return index;
}

