#include <iostream>
#include "vec3.h"
#include "halfedge.h"
#include "vertex.h"

HalfEdge::HalfEdge(Mesh* meshIn, Vertex* v0In, Vertex* v1In, 
	               Triangle* triangleIn) : MeshAttribute(meshIn)
{
  v0 = v0In;
  v1 = v1In;
  triangle = triangleIn;
  halfedge = v1->getHalfEdgeTo(v0);
  if(halfedge != NULL)
  {
	std::cout << "Opposite half edge exists!" << std::endl;
  }
  else
	std::cout << "Opposite half edge does not exist." << std::endl;
}

Vec3 HalfEdge::differenceVec3()
{
  return *v1 - *v0;
}

