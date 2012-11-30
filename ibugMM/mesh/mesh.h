#pragma once

#include <vector>

class Triangle;
class Vertex;

// C++ class layer built on top of simple C data arrays. Mesh is composed 
// of triangles, halfedges, and vertices, each of which is a light C++
// containing pointers to neighbours. This allows for algorithms to be written
// in C++/Cython that can be quick while for looping over vertices. The actual 
// organisation of the data itself is not delt with by this framework - it
// simply works on pointers to C style arrays. This makes it very easy to 
// efficiently interface to this framework from python/matlab without having
// to perform copies everytime we want to run an algorithm.
// The C arrays that can be used are:
//
//   doub* coords - raw position info (can be modified safely)
//   uint* coordsIndex - triangle list (CANNOT be modified)
//   doub* vertexScalar - an array containing a scalar for each vert
//   doub* vertexVec3  - an array containing a 3-vector for each vert
//   doub* triangleScalar - an array containing a scalar for each tri
//   doub* triangleVec3 - an array containing a 3-vector for each tri
//
// All these arrays are accessable from the appropriate C++ classes
// e.g. Vertex->vertexScalar() on the i'th vertex will always point 
// to the correct point in the vertexScalar array attached to mesh
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
	// storage for the c++ objects for each triangle and vertex
	std::vector<Triangle*> triangles;
	std::vector<Vertex*> vertices;
	// pointer to the start of an array dim(n_coords) containing a single
	// value for each vertex.
	double* vertexScalar;
	// pointer to the start of an array dim(n_coords X 3) containing a vec3 
	// value for each vertex.
	double* vertexVec3;
	// pointer to the start of an array dim(n_triangles) containing a single 
	// value for each triangle.
	double* triangleScalar;
	// pointer to the start of an array dim(n_triangles X 3) containing a vec3 
	// value for each triangle.
	double* triangleVec3;
};

class MeshAttribute
{
  public:
	Mesh *mesh;
	MeshAttribute(Mesh* mesh);
};
