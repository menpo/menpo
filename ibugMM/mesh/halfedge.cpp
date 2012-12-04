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
	  //std::cout << "Opposite half edge exists!" << std::endl;
    //std::cout << "setting opposite half edge to me" << std::endl;
    halfedge->halfedge = this;
    
  }
//else
	//std::cout << "Opposite half edge does not exist." << std::endl;
}

Vec3 HalfEdge::differenceVec3()
{
  return *v1 - *v0;
}

bool HalfEdge::partOfFullEdge()
{
  if(halfedge != NULL)
    return true;
  else
    return false;
}

HalfEdge* HalfEdge::clockwiseAroundTriangle()
{
  return v1->halfEdgeOnTriangle(triangle);
}

double HalfEdge::alphaAngle()
{
  Vertex* A = clockwiseAroundTriangle()->v1;
  Vertex* B = v0;
  Vertex* C = v1;
  return angleBetweenVerticies(A,B,C);
}

double HalfEdge::betaAngle()
{
  Vertex* A = v0;
  Vertex* B = v1;
  Vertex* C = clockwiseAroundTriangle()->v1;
  return angleBetweenVerticies(A,B,C);
}

double HalfEdge::gammaAngle()
{
  Vertex* A = v1;
  Vertex* B = clockwiseAroundTriangle()->v1;
  Vertex* C = v0;
  return angleBetweenVerticies(A,B,C);
}

double HalfEdge::angleBetweenVerticies(Vertex* A, Vertex* B, Vertex* C)
{
  //std::cout << "Angle for A = " << *A << " B = " << *B << " C = " << *C << std::endl;
  Vec3 a = *A - *B;
  Vec3 b = *C - *B;
  //std::cout << a << "   " << b << std::cout;
  a.normalize();
  b.normalize();
  return a.dot(b);
}
