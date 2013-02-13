#include <iostream>
#include "vec3.h"
#include "halfedge.h"
#include "vertex.h"
#include "triangle.h"
#include <cmath>


HalfEdge::HalfEdge(Mesh* meshIn, Vertex* v0In, Vertex* v1In, 
	               Triangle* triangleIn) : MeshAttribute(meshIn)
{
	mesh->n_half_edges++;
  v0 = v0In;
  v1 = v1In;
  triangle = triangleIn;
  halfedge = v1->getHalfEdgeTo(v0);
  // v2 is thus always the opposite vertex to this half edge
  v2 = counterclockwiseAroundTriangle()->v1;
  v0_tri_i = triangle->vertex_index_number(v0);
  v1_tri_i = triangle->vertex_index_number(v1);
  v2_tri_i = triangle->vertex_index_number(v2);
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

HalfEdge* HalfEdge::counterclockwiseAroundTriangle()
{
  HalfEdge* he_new;
  if(v1->id == triangle->v0->id)
	  he_new = triangle->e0;
  else if(v1->id == triangle->v1->id)
	  he_new = triangle->e1;
  else if(v1->id == triangle->v2->id)
	  he_new = triangle->e2;
  else
	  std::cout << "ERROR: cannot find HE!" << std::endl;

//  HalfEdge* he_old = v1->halfEdgeOnTriangle(triangle);
//  if (he_old != he_new)
//	std::cout << "Disagreement in methods!" << std::endl;
  return he_new;
}

double HalfEdge::alphaAngle()
{
  Vertex* A = counterclockwiseAroundTriangle()->v1;
  Vertex* B = v0;
  Vertex* C = v1;
  return angleBetweenVerticies(A,B,C);
}

double HalfEdge::betaAngle()
{
  Vertex* A = v0;
  Vertex* B = v1;
  Vertex* C = counterclockwiseAroundTriangle()->v1;
  return angleBetweenVerticies(A,B,C);
}

double HalfEdge::gammaAngle()
{
  Vertex* A = v1;
  Vertex* B = counterclockwiseAroundTriangle()->v1;
  Vertex* C = v0;
  return angleBetweenVerticies(A,B,C);
}

