#include <iostream>
#include "vec3.h"
#include "halfedge.h"
#include "vertex.h"
#include <cmath>


HalfEdge::HalfEdge(Mesh* meshIn, Vertex* v0In, Vertex* v1In, 
	               Triangle* triangleIn) : MeshAttribute(meshIn)
{
	mesh->n_half_edges++;
  v0 = v0In;
  v1 = v1In;
  triangle = triangleIn;
  halfedge = v1->getHalfEdgeTo(v0);
  if(halfedge != NULL)
  {
	  //std::cout << "Opposite half edge exists!" << std::endl;
    //std::cout << "setting opposite half edge to me" << std::endl;
    halfedge->halfedge = this;
	  mesh->n_full_edges++;
  }
  else
    // first time weve encountered this -> add to the mesh set of edges
    meshIn->addEdge(this);
}

Vec3 HalfEdge::differenceVec3()
{
  return *v1 - *v0;
}

double HalfEdge::length()
{
  return differenceVec3().mag();
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

