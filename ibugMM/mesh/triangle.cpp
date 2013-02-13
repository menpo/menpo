#include <iostream>
#include <iomanip>
#include "triangle.h"
#include "vertex.h"
#include "vec3.h"
#include "halfedge.h"

Triangle::Triangle(Mesh* meshIn, unsigned triId, 
	       Vertex* v0In, Vertex* v1In, Vertex* v2In) : MeshAttribute(meshIn)
{
  id = triId;
  v0 = v0In;
  v1 = v1In;
  v2 = v2In;
  // add this triangle to every vertex
  v0->addTriangle(this);
  v1->addTriangle(this);
  v2->addTriangle(this);
  // add every vertex to each other 
  v0->addVertex(v1);
  v0->addVertex(v2);
  v1->addVertex(v0);
  v1->addVertex(v2);
  v2->addVertex(v0);
  v2->addVertex(v1);
  // add the half edges
  e0 = v0->addHalfEdgeTo(v1,this);
  e1 = v1->addHalfEdgeTo(v2,this);
  e2 = v2->addHalfEdgeTo(v0,this);
}

Triangle::~Triangle()
{
}

Vec3 Triangle::normal()
{
  Vec3 d0 = e0->differenceVec3();
  Vec3 d1 = e1->differenceVec3();
  return d0^d1;
}

double Triangle::area()
{
  return 0.5*(normal().mag());
}

Vec3 Triangle::gradient(double* v_scalar_field)
{
  Vec3 N = normal();
  N.normalize();
  double areaFace = area();
  Vec3 e0_v = e0->differenceVec3();
  Vec3 e1_v = e1->differenceVec3();
  Vec3 e2_v = e2->differenceVec3();
  return ((N^e0_v)*v_scalar_field[v2->id] + 
          (N^e1_v)*v_scalar_field[v0->id] + 
          (N^e2_v)*v_scalar_field[v1->id])/(2*areaFace);
}

void Triangle::printStatus()
{
  std::cout << "    TRIANGLE " << id << "        " << std::endl; 
  HalfEdge* h01 = v0->halfEdgeOnTriangle(this);
  HalfEdge* h12 = v1->halfEdgeOnTriangle(this);
  HalfEdge* h20 = v2->halfEdgeOnTriangle(this);
  unsigned width = 12;
  std::cout  << std::setw(width) << "V0(" << v0->id << ")";
  if(h01->partOfFullEdge())
    std::cout << "============";
  else
    std::cout << "------------";
  std::cout  << std::setw(width) << "V1(" << v1->id << ")";
  if(h12->partOfFullEdge())
    std::cout << "============";
  else
    std::cout << "------------";
  std::cout  << std::setw(width) << "V2(" << v2->id << ")";
  if(h20->partOfFullEdge())
    std::cout << "============";
  else
    std::cout << "------------";
  std::cout << std::setw(width) << "V0(" << v0->id << ")" << std::endl;

  std::cout  << std::setw(width) << " ";
  if(h01->partOfFullEdge())
    std::cout  << std::setw(width) << h01->halfedge->triangle->id;
  else
    std::cout << " -- ";
  std::cout  << std::setw(width) << " ";
  if(h12->partOfFullEdge())
    std::cout << std::setw(width) << h12->halfedge->triangle->id;
  else
    std::cout << " -- ";
  std::cout  << std::setw(width) << " ";
  if(h20->partOfFullEdge())
    std::cout << std::setw(width) << h20->halfedge->triangle->id;
  else
    std::cout << " -- ";

}

void Triangle::reduceScalarToVertices(double* triangle_scalar, double* vertex_scalar)
{
  vertex_scalar[v0->id] += triangle_scalar[id];
  vertex_scalar[v1->id] += triangle_scalar[id + 1];
  vertex_scalar[v2->id] += triangle_scalar[id + 2];
}

void Triangle::reduceScalarPerVertexToVertices(
    double* triangle_scalar_per_vertex, double* vertex_scalar)
{
  //std::cout << "Triangle no " << id << " has values " << triangle_scalar[id*3] << " " 
  //  << triangle_scalar[(id*3)+1] << " " << triangle_scalar[(id*3)+2] << std::endl;
  //std::cout << "Writing these out to id's " << v0->id << " " << v1->id << " " << v2->id << std::endl;
  vertex_scalar[v0->id] += triangle_scalar_per_vertex[(id * 3)];
  vertex_scalar[v1->id] += triangle_scalar_per_vertex[(id * 3) + 1];
  vertex_scalar[v2->id] += triangle_scalar_per_vertex[(id * 3) + 2];
}

int Triangle::vertex_index_number(Vertex* vertex)
{
  if(v0->id == vertex->id)
    return 0;
  else if(v1->id == vertex->id)
    return 1;
  else if(v2->id == vertex->id)
    return 2;
  else
	  std::cout << "ERROR: cannot find vertex on this tri!" << std::endl;
  return 4;
}
