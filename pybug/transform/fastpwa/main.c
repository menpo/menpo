#include <stdio.h>
#include "pwa.h"


int main(int argc, char** argv)
{
  double vertices [] = {0., 0.,
                        1., 0.,
                        2., 1.3,
                        0., 1.};
  unsigned int trilist [] = {0, 1, 3,
                             1, 2, 3};
  // make our hash
  AlphaBetaIndex *alphaBetaIndexHash = NULL;
  TriangleCollection tris = initTriangleCollection(vertices, trilist, 2);
  printf("Built a TrangleCollection with %u triangles\n", tris.n_triangles);
  double queryPoints [] = {0., 0.1,
                           0.2, 0.4};
  double alpha [2];
  double beta [2];
  int index [2];
  arrayAlphaBetaIndexForPoints(&alphaBetaIndexHash, &tris, queryPoints, 2, index, alpha, beta);
  arrayAlphaBetaIndexForPoints(&alphaBetaIndexHash, &tris, queryPoints, 2, index, alpha, beta);
  deleteTriangleCollection(&tris);
  return 0;
}

