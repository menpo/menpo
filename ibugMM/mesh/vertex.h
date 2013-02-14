#pragma once
#include <set>
#include <ostream>
#include "mesh.h"

class HalfEdge;
class Triangle;
class Vertex;
class Vec3;

class Vertex : public MeshAttribute
{
    friend std::ostream& operator<<(std::ostream& out, const Vertex& vertex);
    public:
    Vertex(Mesh* mesh, unsigned id, double* coords);
    ~Vertex();
    double* coords;
    unsigned id;
    // half edges STARTING from this vertex 
    std::set<HalfEdge*> halfedges;
    // ALL triangles attached to this vertex
    std::set<Triangle*> triangles;
    // ALL other vertices attached to this vertex
    std::set<Vertex*> vertices;

    // Mesh construction methods
    HalfEdge* add_halfedge_to(Vertex* vertex, Triangle* triangle, 
            unsigned id_on_tri_of_v0);
    void add_triangle(Triangle* triangle);
    void add_vertex(Vertex* vertex);

    // algorithms
    void laplacian(unsigned* i_sparse, unsigned* j_sparse, double* v_sparse,
            unsigned& sparse_pointer, LaplacianWeightType weight_type);
    void cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
            double* w_sparse, unsigned& sparse_pointer, 
            double* cot_per_tri_vertex);
    // different Laplacian weightings
    double distance_weight(HalfEdge* he);

    // utility methods
    HalfEdge* halfedge_on_triangle(Triangle* triangle);
    HalfEdge* halfedge_to_vertex(Vertex* vertex);
    HalfEdge* halfedge_to_or_from_vertex(Vertex* vertex);
    void verifyHalfEdgeConnectivity();
    int verticesAndHalfEdges();
    double getArea();
    void divergence(double* t_vector_field, double* v_scalar_divergence);
    Vec3 operator-(Vertex v);
    Vec3 operator*(Vertex v);
    Vec3 operator^(Vertex v);
    void printStatus();
};

