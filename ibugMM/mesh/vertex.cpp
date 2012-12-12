#include "vertex.h"
#include "halfedge.h"
#include "vec3.h"
#include <iostream>
#include <ostream>
#include "triangle.h"

Vertex::Vertex(Mesh* meshIn, unsigned vertid, double* coordsIn): MeshAttribute(meshIn)
{
  id = vertid;
  coords = coordsIn;
}

void Vertex::addTriangle(Triangle* triangle)
{
  triangles.insert(triangle);
}

void Vertex::addVertex(Vertex* vertex)
{
  //std::cout << "V:" << this << " is now connected to V:" << vertex << std::endl;
  vertices.insert(vertex);
}

// returns the created half edge so it can be attached to the triangle if so desired
HalfEdge* Vertex::addHalfEdgeTo(Vertex* vertex, Triangle* triangle)
{
  if(getHalfEdgeTo(vertex) == NULL)
  {
	HalfEdge* halfedge = new HalfEdge(this->mesh,this,vertex,triangle);
	halfedges.insert(halfedge);
	//std::cout << "V:" << this << " is now connected to HE:" << halfedge << std::endl;
	return halfedge;
  }
  else
  {
	std::cout << "This vertex seems to already be connected! Doing nothing." << std::endl;
	return NULL;
  }
}

HalfEdge* Vertex::getHalfEdgeTo(Vertex* vertex)
{
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	if((*he)->v1 == vertex)
	{
	  //std::cout << "V:" << this << " has a HE to V:" << vertex << std::endl;
	  return *he;
	}
  }
  //std::cout << "V:" << this << " does not have a HE to V:" << vertex << std::endl;
  return NULL;
}

Vertex::~Vertex()
{
  halfedges.clear();
}

Vec3 Vertex::operator-(Vertex v)
{
  Vec3 a = *this;
  Vec3 b = v;
  return a - b;
}

//double* Vertex::vertexScalar()
//{
//  return &(mesh->vertexScalar[id]);
//}
//
//double* Vertex::vertexVec3()
//{
//  return &(mesh->vertexVec3[id*3]);
//}

std::ostream& operator<<(std::ostream& out, const Vertex& v)
{
  out << "V:" << v.id << " (" << v.coords[0] << "," 
	<< v.coords[1] << "," << v.coords[2] << ")";
  return out;
}

HalfEdge* Vertex::halfEdgeOnTriangle(Triangle* triangle)
{
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	if((*he)->triangle == triangle)
	{
	  //std::cout << "V:" << this << " has a HE to V:" << vertex << std::endl;
	  return *he;
	}
  }
  return NULL;
  //std::cout << "V:" << this << " does not have a HE to V:" << vertex << std::endl;
}

void Vertex::calculateLaplacianOperator(unsigned* i_sparse, unsigned* j_sparse,
	                                    double* v_sparse, unsigned& sparse_pointer, 
								        double* vertex_areas)
{
  // sparse_pointer points into how far into the sparse_matrix structures
  // we should be recording results for this vertex
  unsigned i = id;
  double vertexArea = 0.;
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	if((*he)->partOfFullEdge())
	{
	  unsigned j = (*he)->v1->id;
	  //std::cout << *this << " fulledge to " << *((*he)->v1) << std::endl;
	  //double u_diff = *((*he)->v1->vertexScalar()) - *vertexScalar();
	  //std::cout << "u_i - u_j = " << u_diff << std::endl;
	  //std::cout << "theta = " << (*he)->gammaAngle();
	  double cotOp = cotOfAngle((*he)->gammaAngle()) + 
		cotOfAngle((*he)->halfedge->gammaAngle());
	  // write out to the i'th row of the vertexSquarematrix: 
	  // += cotOp to the j'th position 
	  i_sparse[sparse_pointer] = i;
	  j_sparse[sparse_pointer] = j;
	  v_sparse[sparse_pointer] = cotOp;
	  // increment the pointer
	  sparse_pointer++;
	  // -= cotOp to the i'th position 
	  i_sparse[i] = i;
	  j_sparse[j] = j;
	  v_sparse[i] = -cotOp;
	  vertexArea += (*he)->triangle->area();
	}
	//else
	//std::cout << *this << " halfedge to " << *((*he)->v1) << std::endl;
  }
  // store the areas in the array that is passed in
  vertex_areas[id] = (vertexArea*2.0)/3.0; 
}

void Vertex::divergence(double* t_vector_field, double* v_scalar_divergence)
{
  std::set<HalfEdge*>::iterator he;
  double divergence = 0;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	Vec3 field(&t_vector_field[((*he)->triangle->id)*3]);
	std::cout << "field = " << field << std::endl;
	Vec3 e1 = (*he)->differenceVec3();
	Vec3 e2 = (*he)->clockwiseAroundTriangle()->clockwiseAroundTriangle()->halfedge->differenceVec3();
	double cottheta2 = cotOfAngle((*he)->betaAngle());
	double cottheta1 = cotOfAngle((*he)->gammaAngle());
	std::cout << "cottheta = " << cottheta1 << " " << cottheta2 << std::endl;
	divergence += cottheta1*(e1.dot(field)) + cottheta2*(e2.dot(field));
  }
  std::cout << "divergence is " << divergence/2.0 << std::endl;
  v_scalar_divergence[id] = divergence/2.0;
}
