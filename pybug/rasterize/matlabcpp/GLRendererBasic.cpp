#include <mex.h>
#include <iostream>

#include "MM3DRenderer.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *coord;             /* 4xn_vertices input */
  unsigned int *coordIndex;        /* 3xn_triangles input */
  double *textureVector;     /* 4xn_triangles input */
  size_t n_coords;           /* size of matrix */
  size_t n_triangles;         /* size of matrix */

 
  if(nrhs!=3) {
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
		"Three inputs required.");
  }

  if(nlhs!=1) {
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
		"One output required.");
  }

  if(mxGetM(prhs[0])!=4) {
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector",
		"Input must be a 4 x n homogeneous vector.");
  }

  /* create pointers to the input data */
  coord         = (double*) mxGetPr(prhs[0]);
  coordIndex    = (unsigned int*) mxGetPr(prhs[1]);
  textureVector = (double*) mxGetPr(prhs[2]);
  
  n_coords    = mxGetN(prhs[0]);
  n_triangles  = mxGetN(prhs[1]);
  printf ("n_coords: %u\n", n_coords);
  printf ("n_triangles: %u\n", n_triangles);
  
  MM3DRenderer renderer(coord, n_coords, textureVector, coordIndex, n_triangles);
  // fake the usual args that get passed in to a C main function from the OS
  char p0[] = "foo.exe";
  char p1[] = "-x";
  char p2[] = "myfile";
  char p3[] = "-f";
  char p4[] = "myflag";
  char *params[] = { p0, p1, p2, p3, p4, NULL };
  renderer.startFramework(1, params);
  
  // alloc space for the return matrix
  plhs[0] = mxCreateDoubleMatrix(3,static_cast<int>(n_coords),mxREAL);
  double *outMatrix;      /* output matrix */
  /* get a pointer to the real data in the output matrix */
  outMatrix = mxGetPr(plhs[0]);
  outMatrix[0] = 1;
}
