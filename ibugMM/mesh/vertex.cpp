#include <iostream>
#include <ostream>
#include <assert.h>
#include "vertex.h"
#include "halfedge.h"
#include "vec3.h"
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
  bool has_a_full_edge = false;
  unsigned i = id;
  double vertexArea = 0.;
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	unsigned j = (*he)->v1->id;
	//std::cout << *this << " fulledge to " << *((*he)->v1) << std::endl;
	//double u_diff = *((*he)->v1->vertexScalar()) - *vertexScalar();
	//std::cout << "u_i - u_j = " << u_diff << std::endl;
	//std::cout << "theta = " << (*he)->gammaAngle();
	double cotOp = cotOfAngle((*he)->gammaAngle());
	if((*he)->partOfFullEdge())
	{
	  has_a_full_edge = true;
	  cotOp += cotOfAngle((*he)->halfedge->gammaAngle());
	}
	// write out to the i'th row of the vertexSquarematrix: 
	// += cotOp to the j'th position 
	i_sparse[sparse_pointer] = i;
	j_sparse[sparse_pointer] = j;
	//if(v_sparse[sparse_pointer] != 0)
	//  std::cout << "this matrix value is already taken?" << std::endl;
	v_sparse[sparse_pointer] = cotOp/2.0;
	// increment the pointer
	sparse_pointer++;
	// -= cotOp to the i'th position 
	v_sparse[i] -= cotOp/2.0;
	vertexArea += (*he)->triangle->area();
	//else
	//std::cout << *this << " halfedge to " << *((*he)->v1) << std::endl;
  }
  // store the areas in the array that is passed in
  vertex_areas[id] = vertexArea/3.0; 
  if(!has_a_full_edge)
	std::cout << "Vertex " << id << " does not have any full edges around it (" << halfedges.size() << " halfedges around it)" << std::endl;
}

void Vertex::divergence(double* t_vector_field, double* v_scalar_divergence)
{
  //std::cout << "Calculating diergence for vertex no. " << id << "(" << halfedges.size() << " halfedges)" << std::endl ;
  std::set<HalfEdge*>::iterator he;
  double divergence = 0;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	Vec3 field(&t_vector_field[((*he)->triangle->id)*3]);
	//std::cout << "field = " << field << std::endl;
	Vec3 e1 = (*he)->differenceVec3();
	//std::cout << "Got diff vec!" << std::endl;
	// *-1 as we want to reverse the direction
	Vec3 e2 = (*he)->clockwiseAroundTriangle()->clockwiseAroundTriangle()->differenceVec3()*-1;
	//std::cout << "Got other diff vec!" << std::endl;
	double cottheta2 = cotOfAngle((*he)->betaAngle());
	double cottheta1 = cotOfAngle((*he)->gammaAngle());
	//std::cout << "cottheta1 = " << cottheta1 << " cottheta2 = " << cottheta2 << std::endl;
	divergence += cottheta1*(e1.dot(field)) + cottheta2*(e2.dot(field));
  }
  //std::cout << "       divergence is " << divergence/2.0 << std::endl << std::endl;
  v_scalar_divergence[id] = divergence/2.0;
}

void Vertex::verifyHalfEdgeConnectivity()
{
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	Triangle* triangle = (*he)->triangle;
	Vertex* t_v0 = triangle->v0;
	Vertex* t_v1 = triangle->v1;
	Vertex* t_v2 = triangle->v2;
	if(t_v0 != this && t_v1 != this && t_v2 != this)
	  std::cout << "this halfedge does not live on it's triangle!" << std::endl;
	if((*he)->v0 != this)
	  std::cout << "half edge errornously connected" << std::endl;
	if((*he)->clockwiseAroundTriangle()->clockwiseAroundTriangle()->v1 != (*he)->v0)
	  std::cout << "cannie spin raarnd the triangle like man!" << std::endl;
	if((*he)->partOfFullEdge())
	{
	  if((*he)->halfedge->v0 != (*he)->v1 || (*he)->halfedge->v1 != (*he)->v0)
		std::cout << "some half edges aren't paired up with there buddies!" << std::endl;
	}
  }
}

void Vertex::printStatus()
{
  std::cout << "V" << id << std::endl;
  std::set<HalfEdge*>::iterator he;
  for(he = halfedges.begin(); he != halfedges.end(); he++)
  {
	std::cout << "|" ;
	if((*he)->partOfFullEdge())
	  std::cout << "=";
	else
	  std::cout << "-";
	std::cout << "V" << (*he)->v1->id;
	std::cout << " (T" << (*he)->triangle->id; 
	if((*he)->partOfFullEdge())
	  std::cout << "=T" << (*he)->halfedge->triangle->id;
	std::cout << ")" << std::endl;
  }
}
