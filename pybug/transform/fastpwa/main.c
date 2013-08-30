#include <stdio.h>
#include "pwa.h"

typedef struct {
  Point queryPoint;
  int index;
  double alpha;
  double beta;
  UT_hash_handle hh;
} AlphaBetaResult;

AlphaBetaResult *hashMap = NULL;

void addAlphaBetaResultToCache(Point queryPoint, int index, double alpha, double beta)
{
  // dynamically allocate a new result object
  AlphaBetaResult *result, *resultInHash;
  result = malloc(sizeof(AlphaBetaResult));
  memset(result, 0, sizeof(AlphaBetaResult));
  result->queryPoint = queryPoint;
  result->index = index;
  result->alpha = alpha;
  result->beta = beta;
  // check to see if there is already this result in the hash
  HASH_FIND(hh, hashMap, &(result->queryPoint), sizeof(Point), resultInHash);
  if (resultInHash) {
    printf("found a result already!");
  } else {
    printf("No hash for this yet, adding...");
    HASH_ADD(hh, hashMap, queryPoint, sizeof(Point), result);
  }
}

int main(int argc, char** argv)
{
  double vertices [] = {0., 0.,
                        1., 0.,
                        2., 1.3,
                        0., 1.};
  unsigned int trilist [] = {0, 1, 3,
                             1, 2, 3};
  TriangleCollection tris = initTriangleCollection(vertices, trilist, 2);
  printf("Built a TrangleCollection with %u triangles\n", tris.n_triangles);
  Point testPoint = {0.6, 0.7};
  printf("Test point: ");
  pointPrint(testPoint);
  printf("\n");
  double alpha, beta;
  int index = containingTriangleAndAlphaBetaForPoint(tris, testPoint, &alpha, &beta);
  if (index >=0) {
    printf("Test point lies in triangle %u with alpha = %1.2f, beta = %1.2f\n", index, alpha, beta);
  } else {
    printf("Test point doesn't lie in any triangle");
  }
  
  HASH_ADD(results, point, 
  
  return 0;
}

