#pragma once
#include "mesh.h"

class HalfEdge : public MeshAttribute
{
    public:
        HalfEdge* halfedge; // the opposite halfedge in the fulledge (or NULL)
        // TODO rename these to distinguish from v0 on triangle
        Vertex* v0;
        Vertex* v1;
        Vertex* v2; // opposite vertex on my triangle
        int v0_tri_i; // The position no. of each vertex on triangle
        int v1_tri_i; // i.e., if MY v0 is actually the triangle's v1,
        int v2_tri_i; // v0_tri_i == 1, v1_tri_i == 2 ... (always CCW)
        Triangle* triangle;
        HalfEdge(Mesh* mesh, Vertex* v0, Vertex* v1,
                Triangle* triangle, unsigned id_on_tri_of_v0);
        ~HalfEdge();
        bool part_of_fulledge();
        HalfEdge* ccw_around_tri();
        double length();
};

