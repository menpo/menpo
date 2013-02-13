#pragma once
#include <set>
#include <ostream>
#include "mesh.h"

class Triangle;
class Vertex;
class HalfEdge;
class Vec3;

class Vertex : public MeshAttribute
{
  friend std::ostream& operator<<(std::ostream& out, const Vertex& vertex);
  public:
  Vertex(Mesh* mesh, unsigned id, double* coords);
  double *coords;
  unsigned id;
  // ALL triangles attached to this vertex
  std::set<Triangle*> triangles;
  // ALL other vertices attached to this vertex
  std::set<Vertex*> vertices;
  // half edges STARTING from this vertex 
  std::set<HalfEdge*> halfedges;
  // Mesh construction methods
  void addTriangle(Triangle* triangle);
  void addVertex(Vertex* vertex);
  HalfEdge* addHalfEdgeTo(Vertex* vertex, Triangle* triangle);
  
  // algorithms
  void divergence(double* t_vector_field, double* v_scalar_divergence);
  ~Vertex();
  void calculateLaplacianOperator(unsigned* i_sparse, unsigned* j_sparse,
	                                double* v_sparse, unsigned& sparse_pointer, 
								                  LaplacianWeightType weight_type);
  void cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
                           double* w_sparse, unsigned& sparse_pointer, 
                           double* cotangents_per_vertex);
  // different Laplacian weightings
  double cotWeight(HalfEdge* he);
  double distanceWeight(HalfEdge* he);
  double combinatorialWeight(HalfEdge* he);

  // utility methods
  HalfEdge* halfEdgeOnTriangle(Triangle* triangle);
  HalfEdge* getHalfEdgeTo(Vertex* vertex);
  HalfEdge* getHalfEdgeToOrFrom(Vertex* vertex);
  void verifyHalfEdgeConnectivity();
  int verticesAndHalfEdges();
  double getArea();
  Vec3 operator-(Vertex v);
  Vec3 operator*(Vertex v);
  Vec3 operator^(Vertex v);
  void printStatus();
};

