#include <iostream>
#include <vector>
#include "mesh.h"
// MeshAttributes
#include "triangle.h"
#include "vertex.h"
#include "vec3.h"
#include <cmath>

const double PI_2 = 2.0*atan(1.0);

Mesh::Mesh(double   *coordsIn,      unsigned n_coordsIn,
	       unsigned *coordsIndexIn, unsigned n_trianglesIn)
{
  coords   = coordsIn;
  n_coords = n_coordsIn;
  coordsIndex = coordsIndexIn;
  n_triangles = n_trianglesIn;
  // set the no. of full edges to 0
  // (on creation halfedge pairs will increment this)
  n_full_edges = 0;
  //vertexMatrix.reserve(n_coords*12);
  // build a Vertex object for each coord set passed in
  for(unsigned i = 0; i < n_coords; i++)
	vertices.push_back(new Vertex(this, i,  &coords[i*3]));
  for(unsigned i = 0; i < n_triangles; i++)
  {
	// get the index into the vertex positions
	unsigned l = coordsIndex[i*3    ];
	unsigned m = coordsIndex[i*3 + 1];
	unsigned n = coordsIndex[i*3 + 2];
	// build a new triangle, passing in the pointers to the vertices it will
	// be made from (the triangle in it's construction will build edges and
	// connect them)
	triangles.push_back(new Triangle(this, i, vertices[l],vertices[m],vertices[n]));
  }
  std::cout << "n_full_edges = " << n_full_edges << std::endl;
}

Mesh::~Mesh()
{
  triangles.clear();
  // now all the triangles are clear we're good to delete all the vertices
  // we initially made
  vertices.clear();
}

void Mesh::verifyAttachements()
{
  unsigned v = n_coords - 1;
  unsigned t = n_triangles - 1;
  std::cout << "ver0: scalar = " <<    *(vertices[0]->vertexScalar()) << std::endl;
  std::cout << "ver0: vector = " << Vec3(vertices[0]->vertexVec3())   << std::endl;
  std::cout << "vern: scalar = " <<    *(vertices[v]->vertexScalar()) << std::endl;
  std::cout << "vern: vector = " << Vec3(vertices[v]->vertexVec3())   << std::endl;
  std::cout << "tri0: scalar = " <<    *(triangles[0]->triangleScalar()) << std::endl;
  std::cout << "tri0: vector = " << Vec3(triangles[0]->triangleVec3())   << std::endl;
  std::cout << "trin: scalar = " <<    *(triangles[t]->triangleScalar()) << std::endl;
  std::cout << "trin: vector = " << Vec3(triangles[t]->triangleVec3())   << std::endl;
}

void Mesh::calculateLaplacianOperator()
{
  // we expect that the attachments at i_sparse, j_sparse
  // and v_sparse have already been set to the correct 
  // dimentions before this call 
  // (each should be of length n_coords + 2*n_full_edges)
  // the first n_coord entries are the diagonals. -> the i'th
  // value of both i_sparse and j_sparse is just i
  for(unsigned int i = 0; i < n_coords; i++)
  {
	i_sparse[i] = i;
	j_sparse[i] = i;
  }
  // set the sparse_pointer to the end of the diagonal elements
  unsigned sparse_pointer = n_coords;
  // now loop through each vertex and call the laplacian method.
  // This method will populate the sparse matrix arrays with the
  // position and value that should be assiged to the matrix
  std::vector<Vertex*>::iterator v;
  for(v = vertices.begin(); v != vertices.end(); v++)
	(*v)->calculateLaplacianOperator(sparse_pointer);
  //vertices[0]->calculateLaplacianOperator();
  std::cout << "After laplacian, sparse pointer at " << sparse_pointer << std::endl;
}

MeshAttribute::MeshAttribute(Mesh* meshIn)
{
  mesh = meshIn;
}

double angleBetweenVerticies(Vertex* A, Vertex* B, Vertex* C)
{
  //std::cout << "Angle for A = " << *A << " B = " << *B << " C = " << *C << std::endl;
  Vec3 a = *A - *B;
  Vec3 b = *C - *B;
  //std::cout << a << "   " << b << std::cout;
  a.normalize();
  b.normalize();
  return std::acos(a.dot(b));
}

double cotOfAngle(double theta)
{
  return std::tan(PI_2 - theta);
}
