#pragma once
#include <set>
#include "mesh.h"

class Triangle;
class Vertex;
class HalfEdge;
class Vec3;

class Vertex : public MeshAttribute
{
  public:
	Vertex(Mesh* mesh, unsigned id, double* coords);
	double *coords;
	// see triangle id
	unsigned id;
	// triangles attached to this vertex
	std::set<Triangle*> triangles;
	// other vertices attached to this vertex
	std::set<Vertex*> vertices;
	// half edges STARTING from this vertex 
	std::set<HalfEdge*> halfedges;
	// Mesh construction methods
	void addTriangle(Triangle* triangle);
	void addVertex(Vertex* vertex);
	HalfEdge* addHalfEdgeTo(Vertex* vertex, Triangle* triangle);
	~Vertex();
	// a pointer to a structure identical to 
	double* vertexScalar();
	double* vertexVec3();

	// utility methods
	HalfEdge* getHalfEdgeTo(Vertex* vertex);
	double getArea();
	Vec3 operator-(Vertex v);
	Vec3 operator*(Vertex v);
	Vec3 operator^(Vertex v);
};

