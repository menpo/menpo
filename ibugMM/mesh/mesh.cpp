#include <iostream>
#include <vector>
#include "mesh.h"
// MeshAttributes
#include "triangle.h"
#include "vertex.h"
#include "vec3.h"


Mesh::Mesh(double   *coordsIn,      unsigned n_coordsIn,
	         unsigned *coordsIndexIn, unsigned n_trianglesIn)
{
  coords   = coordsIn;
  n_coords = n_coordsIn;
  coordsIndex = coordsIndexIn;
  n_triangles = n_trianglesIn;
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

void Mesh::calculateLaplacianOfScalar()
{
  vertices[0]->calculateLaplacianOfScalar();

}

MeshAttribute::MeshAttribute(Mesh* meshIn)
{
  mesh = meshIn;
}

