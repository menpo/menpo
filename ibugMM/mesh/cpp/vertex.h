#pragma once
#include <set>
#include <ostream>
#include "mesh.h"

class HalfEdge;
class Triangle;
class Vertex;

class Vertex : public MeshAttribute
{
    public:
        double* coords;
        std::set<HalfEdge*> halfedges; //halfedges STARTING from this vertex
        std::set<Triangle*> triangles; //ALL triangles attached to this vertex
        std::set<Vertex*> vertices; //ALL other vertices attached to this vertex
        Vertex(Mesh* mesh, unsigned id, double* coords);
        ~Vertex();

        // mesh construction methods
        HalfEdge* add_halfedge_to(Vertex* vertex, Triangle* triangle,
                unsigned id_on_tri_of_v0);
        void add_triangle(Triangle* triangle);
        void add_vertex(Vertex* vertex);

        // halfedge retrieval methods about this vertex
        HalfEdge* halfedge_on_triangle(Triangle* triangle);
        HalfEdge* halfedge_to_vertex(Vertex* vertex);
        HalfEdge* halfedge_to_or_from_vertex(Vertex* vertex);

        // algorithms
        void laplacian(unsigned* i_sparse, unsigned* j_sparse, double* v_sparse,
                unsigned& sparse_pointer, LaplacianWeightType weight_type);
        double laplacian_distance_weight(HalfEdge* he);
        void cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
                double* w_sparse, unsigned& sparse_pointer,
                double* cot_per_tri_vertex);

        // utility methods
        void status();
        void verify_halfedge_connectivity();
        void test_contiguous(std::set<Vertex*>* vertices_visited);
        friend std::ostream& operator<<(std::ostream& out, const Vertex& vertex);
};

