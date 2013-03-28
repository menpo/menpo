#pragma once

#include <vector>
#include <set>

class Triangle;
class Vertex;
class HalfEdge;

enum LaplacianWeightType {combinatorial, distance};

// C++ class layer built on top of simple C data arrays. Mesh is composed
// of triangles, halfedges, and vertices, each of which is a light C++ class
// containing pointers to neighbours. This allows for algorithms to be written
// in C++/Cython that can efficiently loop over the triangle structure. The
// actual organisation of the data itself is not delt with by this framework
// - it simply works on pointers to C style arrays passed into the methods
// defined on this class. This makes it very easy to efficiently interface
// to this framework from python/matlab without having to perform copies
// everytime we want to run an algorithm.
//
// Triangles and vertices both have an unsigned 'id' field that can be safely
// used to index into arrays. Array arguments follow a structure to identify
// their required size:
//    double* t_vector_field
//            ^   ^
//    one entry    3 values (x,y,z) per entry
//    per Tri
//                                                => shape = [n_triangles, 3]
//
// and on the 342'nd Triangle, this.id = 341, so
//
//   x = t_vector_field[this.id*3 + 0]
//   y = t_vector_field[this.id*3 + 1]
//   z = t_vector_field[this.id*3 + 2]
//
// are the relevent entries in the array.
//
// Note that this framework expects all arrays to be allocated to the
// correct size before method invocation!
//
class Mesh
{
    public:
        Mesh(unsigned *tri_index, unsigned n_triangles, unsigned n_vertices);
        ~Mesh();
        // pointer to an array dim(n_verticesx3) containing the coordinates
        // for each vertex of the mesh
        unsigned n_vertices;
        unsigned n_triangles;
        unsigned n_fulledges;
        unsigned n_halfedges;
        // storage for the c++ objects for each triangle and vertex
        std::vector<Triangle*> triangles;
        std::vector<Vertex*> vertices;
        std::set<HalfEdge*> edges;

        void add_edge(HalfEdge* halfedge);
        void generate_edge_index(unsigned* edgeIndex);

        void laplacian(unsigned* i_sparse, unsigned* j_sparse,
                double*   v_sparse, LaplacianWeightType weight_type);
        void cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
                double*   v_sparse, double* cotangents_per_vertex);
        void reduce_tri_scalar_to_vertices(double* triangle_scalar,
                double* vertex_scalar);
        void reduce_tri_scalar_per_vertex_to_vertices(
                double* triangle_scalar_per_vertex, double* vertex_scalar);

        // utilities
        void verify_mesh();
        void test_contiguous();
        std::vector< std::set<Vertex*> > contiguous_regions();
        void test_chiral_consistency();
};


class MeshAttribute
{
    public:
        Mesh *mesh;
        unsigned id;
        MeshAttribute(Mesh* mesh, unsigned id);
};

