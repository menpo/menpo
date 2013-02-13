#pragma once
#include "mesh.h"

class Vertex;
class Vec3;
class HalfEdge;

class Triangle : public MeshAttribute
{
  public:
	Vertex* v0;
	Vertex* v1;
	Vertex* v2;
	// the corresponding half edges for this triangle
	// (note v0->-e0->-v1->-e1->v2->-e2->v0)
	HalfEdge* e0;
	HalfEdge* e1;
	HalfEdge* e2;
	//double* triangleScalar();
	//double* triangleVec3();
	// the 'number' of the triangle. Used to point into data matrices at
	// the correct point.
	unsigned id;
	Triangle(Mesh* mesh, unsigned id, Vertex* v0, Vertex* v1, Vertex* v2);
	~Triangle();
	Vec3 normal();
	double area();
	Vec3 gradient(double* scalar_field);
  void printStatus();
  void reduceScalarPerVertexToVertices(double* triangle_scalar_per_vertex, double* vertex_scalar);
  void reduceScalarToVertices(double* triangle_scalar, double* vertex_scalar);
  int vertex_index_number(Vertex* vertex);
};
