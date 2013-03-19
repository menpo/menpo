#pragma once
#include "mesh.h"

class Vertex;
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
        Triangle(Mesh* mesh, unsigned id, Vertex* v0, Vertex* v1, Vertex* v2);
        ~Triangle();

        // algorithms
        void reduce_scalar_per_vertex_to_vertices(
                double* triangle_scalar_per_vertex, double* vertex_scalar);
        void reduce_scalar_to_vertices(
                double* triangle_scalar, double* vertex_scalar);

        // utilities
        void status();
};
