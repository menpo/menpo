#pragma once

#include <vector>

class Triangle;
class Vertex;

enum LaplacianWeightType {cotangent, combinatorial, distance};

// C++ class layer built on top of simple C data arrays. Mesh is composed 
// of triangles, halfedges, and vertices, each of which is a light C++
// containing pointers to neighbours. This allows for algorithms to be written
// in C++/Cython that can be quick while for looping over vertices. The actual 
// organisation of the data itself is not delt with by this framework - it
// simply works on pointers to C style arrays passed into the methods defined
// on this class. This makes it very easy to efficiently interface to this 
// framework from python/matlab without having to perform copies everytime 
// we want to run an algorithm.
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
// and on the 342'nd Triangle, this.id = 342, so
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
    Mesh(double   *coords,      unsigned n_coords,
        unsigned *coordsIndex, unsigned n_triangles);
    ~Mesh();
    // pointer to an array dim(n_coordsx3) containing the coordinates
    // for each vertex of the mesh
    double* coords;
    // pointer to an array dim(n_coordsx3) containing the coordinates
    // for each vertex of the mesh
    unsigned* coordsIndex;
    unsigned n_coords;
    unsigned n_triangles;
    unsigned n_full_edges;
    unsigned n_half_edges;
    // storage for the c++ objects for each triangle and vertex
    std::vector<Triangle*> triangles;
    std::vector<Vertex*> vertices;

    void calculateLaplacianOperator(unsigned* i_sparse, unsigned* j_sparse, 
		                            double*   v_sparse, double*   vertex_areas);
    void calculateGradient(double* v_scalar_field, double* t_vector_gradient);
    void calculateDivergence(double* t_vector_field, double* v_scalar_divergence);
	void verifyMesh();
  double meanEdgeLength();
};


class MeshAttribute
{
  public:
    Mesh *mesh;
    MeshAttribute(Mesh* mesh);
};

double angleBetweenVerticies(Vertex* A, Vertex* B, Vertex* C);
double cotOfAngle(double angle);

