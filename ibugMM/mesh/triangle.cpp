#include "triangle.h"
#include "vertex.h"
#include "vec3.h"
#include "halfedge.h"

Triangle::Triangle(Mesh* meshIn, unsigned triId, 
	       Vertex* v0In, Vertex* v1In, Vertex* v2In) : MeshAttribute(meshIn)
{
  id = triId;
  v0 = v0In;
  v1 = v1In;
  v2 = v2In;
  // add this triangle to every vertex
  v0->addTriangle(this);
  v1->addTriangle(this);
  v2->addTriangle(this);
  // add every vertex to each other 
  v0->addVertex(v1);
  v0->addVertex(v2);
  v1->addVertex(v0);
  v1->addVertex(v2);
  v2->addVertex(v0);
  v2->addVertex(v1);
  // add the half edges
  e0 = v0->addHalfEdgeTo(v1,this);
  e1 = v1->addHalfEdgeTo(v2,this);
  e2 = v2->addHalfEdgeTo(v0,this);
}

Triangle::~Triangle()
{
}

Vec3 Triangle::normal()
{
  Vec3 d0 = e0->differenceVec3();
  Vec3 d1 = e1->differenceVec3();
  return d0^d1;
}

double Triangle::area()
{
  return 0.5*(normal().mag());
}

//void Triangle::gradient()
//{
//  Vec3 u0(*v0);
//  //Vec3 u1(*v1);
//  //Vec3 u2(*v2);
//  //Vec3 e0v = e0->differenceVec3();
//  //Vec3 e1v = e0->differenceVec3();
//  //Vec3 e2v = e0->differenceVec3();
//  //Vec3 N = normal();
//  //N.normalize();
//
//  return u0;
//}

//double* Triangle::triangleScalar()
//{
//  return &(mesh->triangleScalar[id]);
//}
//
//double* Triangle::triangleVec3()
//{
//  return &(mesh->triangleVec3[id*3]);
//}

Vec3 Triangle::gradient(double* v_scalar_field)
{
  Vec3 N = normal();
  N.normalize();
  double areaFace = area();
  Vec3 e0_v = e0->differenceVec3();
  Vec3 e1_v = e1->differenceVec3();
  Vec3 e2_v = e2->differenceVec3();
  return ((N^e0_v)*v_scalar_field[v2->id] + 
          (N^e1_v)*v_scalar_field[v0->id] + 
          (N^e2_v)*v_scalar_field[v1->id])/(2*areaFace);
}
