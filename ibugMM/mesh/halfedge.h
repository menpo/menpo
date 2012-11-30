#pragma once 
#include "mesh.h"

class Vec3;

class HalfEdge : public MeshAttribute
{
  public:
	HalfEdge(Mesh* meshIn, Vertex* v0In, Vertex* v1In, Triangle* triangleIn);
	const HalfEdge* halfedge;
	Vertex* v0;
	Vertex* v1;
	Triangle* triangle;
	Vec3 differenceVec3();
};
