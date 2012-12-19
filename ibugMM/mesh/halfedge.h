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
	Triangle* triangle;
	Vec3 differenceVec3();
  bool partOfFullEdge();
  HalfEdge* clockwiseAroundTriangle();
  double alphaAngle();
  double betaAngle();
  double gammaAngle();
  double length();
};
