#pragma once 
#include "mesh.h"

class Vec3;

class HalfEdge : public MeshAttribute
{
  public:
	HalfEdge(Mesh* meshIn, Vertex* v0In, Vertex* v1In, Triangle* triangleIn);
	HalfEdge* halfedge;
	Vertex* v0;
	Vertex* v1;
	Vertex* v2;
	int v0_tri_i;
	int v1_tri_i;
	int v2_tri_i;
	Triangle* triangle;
	Vec3 differenceVec3();
  bool partOfFullEdge();
  HalfEdge* counterclockwiseAroundTriangle();
  double alphaAngle();
  double betaAngle();
  double gammaAngle();
  double length();
};
