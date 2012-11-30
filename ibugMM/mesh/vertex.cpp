#include "vertex.h"
#include "halfedge.h"
#include "vec3.h"
#include <iostream>

Vertex::Vertex(Mesh* meshIn, unsigned vertid, double* coordsIn): MeshAttribute(meshIn)
{
  id = vertid;
  coords = coordsIn;
}

void Vertex::addTriangle(Triangle* triangle)
{
  triangles.insert(triangle);
}

void Vertex::addVertex(Vertex* vertex)
{
  std::cout << "V:" << this << " is now connected to V:" << vertex << std::endl;
  vertices.insert(vertex);
}

// returns the created half edge so it can be attached to the triangle if so desired
HalfEdge* Vertex::addHalfEdgeTo(Vertex* vertex, Triangle* triangle)
{
  if(getHalfEdgeTo(vertex) == NULL)
  {
	HalfEdge* halfedge = new HalfEdge(this->mesh,this,vertex,triangle);
	halfedges.insert(halfedge);
	std::cout << "V:" << this << " is now connected to HE:" << halfedge << std::endl;
	return halfedge;
  }
  else
  {
	std::cout << "This vertex seems to already be connected! Doing nothing." << std::endl;
	return NULL;
  }
}

HalfEdge* Vertex::getHalfEdgeTo(Vertex* vertex)
{
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	if((*he)->v1 == vertex)
	{
	  std::cout << "V:" << this << " has a HE to V:" << vertex << std::endl;
	  return *he;
	}
  }
  std::cout << "V:" << this << " does not have a HE to V:" << vertex << std::endl;
  return NULL;
}

Vertex::~Vertex()
{
  halfedges.clear();
}

Vec3 Vertex::operator-(Vertex v)
{
  Vec3 a = *this;
  Vec3 b = v;
  return a - b;
}

double* Vertex::vertexScalar()
{
  return &(mesh->vertexScalar[id]);
}

double* Vertex::vertexVec3()
{
  return &(mesh->vertexVec3[id*3]);
}
