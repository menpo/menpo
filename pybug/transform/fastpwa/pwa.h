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
int containingTriangleAndAlphaBetaForPoint(TriangleCollection tris, Point p,
                                           double *alpha, double *beta);

