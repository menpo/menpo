#include <mex.h>
#include <iostream>
#include <stdint.h>

#include "MM3DRenderer.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *tpsCoord;             // 4 x n_vertices   HomoG warped coords per vertex
  float  *coord;                // 3 x n_vertices   Unwarped coord for each vertex.
  float  *texCoord;             // 2 x n_vertices   Indexes into textureImage per vertex.
  unsigned int  *coordIndex;    // 3 x n_triangles  triangle list
  unsigned char *textureImage;  // tWidth*tHeight*4 RGBA texture indexed by texCoord
  
  int* dimensionVector;
  int frameWidth;
  int frameHeight;
  size_t n_coords;
  size_t n_triangles;
  size_t textureWidth;
  size_t textureHeight;
  bool interactiveMode;
  
  // sanity checks
  if(nrhs!=8)
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
		"Eight inputs required.");
  if(nlhs!=2)
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
		"Two outputs required.");
  if(mxGetM(prhs[0])!=4)
	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector",
		"Input must be a 4 x n homogeneous vector.");
  
  // associate inputs
  tpsCoord        = (double*)       mxGetPr(prhs[0]);
  coord           = (float*)        mxGetPr(prhs[1]);
  coordIndex      = (unsigned int*) mxGetPr(prhs[2]);
  texCoord        = (float*)        mxGetPr(prhs[3]);
  textureImage    = (uint8_t*)      mxGetPr(prhs[4]);
  frameWidth      = mxGetScalar(prhs[5]);
  frameHeight     = mxGetScalar(prhs[6]);
  interactiveMode = *mxGetLogicals(prhs[7]);
  n_coords        = mxGetN(prhs[0]);
  n_triangles     = mxGetN(prhs[2]);
  dimensionVector = (int*)mxGetDimensions(prhs[4]);
  textureWidth    = dimensionVector[1];
  textureHeight   = dimensionVector[2];
  
  // print key statistics
  printf ("n_coords: %u\n", n_coords);
  printf ("n_triangles: %u\n", n_triangles);
  printf ("textureWidth: %u\n", textureWidth);
  printf ("textureHeight: %u\n", textureHeight);  
  printf ("frameWidth:  %u:\n", frameWidth);
  printf ("frameHeight: %u:\n", frameHeight);
  
  // fake the usual args that get passed in to a C main function from the OS
  char p0[] = "foo.exe";
  char p1[] = "-x";
  char p2[] = "afile";
  char p3[] = "-f";
  char p4[] = "aflag";
  char *params[] = { p0, p1, p2, p3, p4, NULL };
  
  // alloc space for the returns from the render
  uint8_t* pixels  = new uint8_t[frameWidth*frameHeight*4];
  float* xyzCoords = new   float[frameWidth*frameHeight*3];
  if(interactiveMode)
  {
       // create our renderer object
      MM3DRenderer renderer(tpsCoord, coord, n_coords, coordIndex, n_triangles, 
                       texCoord, textureImage, textureWidth, textureHeight, true);
      printf("Launching interactive viewer\n");
      renderer.render(1, params);
  }
  else
  {
      // create our renderer object
      MM3DRenderer renderer(tpsCoord, coord, n_coords, coordIndex, n_triangles, 
                       texCoord, textureImage, textureWidth, textureHeight, false);
      // actually call OpenGL (results stored in pixels/xyzCoords)
      printf("Grabbing framebuffer\n");
      renderer.returnFBPixels(1, params, pixels, xyzCoords, frameWidth, frameHeight);
  }

  // alloc space for the return matrix
  uint8_t* frameBufferOut;
  float*   coordBufferOut;
  plhs[0] = mxCreateNumericMatrix(1, frameWidth*frameHeight*4, mxUINT8_CLASS, mxREAL);
  plhs[1] = mxCreateNumericMatrix(1, frameWidth*frameHeight*3, mxSINGLE_CLASS, mxREAL);
  frameBufferOut = (uint8_t *)mxGetData(plhs[0]);
  coordBufferOut = (float *)mxGetData(plhs[1]);
  
  // write out the results
  for(int i = 0; i < frameWidth*frameHeight*4; i++)
      frameBufferOut[i] = pixels[i];
  for(int i = 0; i < frameWidth*frameHeight*3; i++)
      coordBufferOut[i] = xyzCoords[i];
  
  // clean up and return
  delete[] pixels;
  delete[] xyzCoords;
  return;
}
