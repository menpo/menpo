#include <iostream>
#include <iomanip>
#include "triangle.h"
#include "vertex.h"
#include "vec3.h"
#include "halfedge.h"

Triangle::Triangle(Mesh* meshIn, unsigned triId, 
        Vertex* v0In, Vertex* v1In, Vertex* v2In) : MeshAttribute(meshIn) {
    id = triId;
    v0 = v0In;
    v1 = v1In;
    v2 = v2In;
    v0->add_triangle(this);
    v1->add_triangle(this);
    v2->add_triangle(this);
    v0->add_vertex(v1);
    v0->add_vertex(v2);
    v1->add_vertex(v0);
    v1->add_vertex(v2);
    v2->add_vertex(v0);
    v2->add_vertex(v1);
    e0 = v0->add_halfedge_to(v1, this, 0);
    e1 = v1->add_halfedge_to(v2, this, 1);
    e2 = v2->add_halfedge_to(v0, this, 2);
}

Triangle::~Triangle() {
}

void Triangle::reduce_scalar_to_vertices(double* triangle_scalar, double* vertex_scalar) {
    vertex_scalar[v0->id] += triangle_scalar[id];
    vertex_scalar[v1->id] += triangle_scalar[id + 1];
    vertex_scalar[v2->id] += triangle_scalar[id + 2];
}

void Triangle::reduce_scalar_per_vertex_to_vertices(
        double* triangle_scalar_per_vertex, double* vertex_scalar) {
    vertex_scalar[v0->id] += triangle_scalar_per_vertex[(id * 3)];
    vertex_scalar[v1->id] += triangle_scalar_per_vertex[(id * 3) + 1];
    vertex_scalar[v2->id] += triangle_scalar_per_vertex[(id * 3) + 2];
}


void Triangle::printStatus() {
    std::cout << "    TRIANGLE " << id << "        " << std::endl; 
    HalfEdge* h01 = v0->halfedge_on_triangle(this);
    HalfEdge* h12 = v1->halfedge_on_triangle(this);
    HalfEdge* h20 = v2->halfedge_on_triangle(this);
    unsigned width = 12;
    std::cout  << std::setw(width) << "V0(" << v0->id << ")";
    if(h01->partOfFullEdge())
        std::cout << "============";
    else
        std::cout << "------------";
    std::cout  << std::setw(width) << "V1(" << v1->id << ")";
    if(h12->partOfFullEdge())
        std::cout << "============";
    else
        std::cout << "------------";
    std::cout  << std::setw(width) << "V2(" << v2->id << ")";
    if(h20->partOfFullEdge())
        std::cout << "============";
    else
        std::cout << "------------";
    std::cout << std::setw(width) << "V0(" << v0->id << ")" << std::endl;

    std::cout  << std::setw(width) << " ";
    if(h01->partOfFullEdge())
        std::cout  << std::setw(width) << h01->halfedge->triangle->id;
    else
        std::cout << " -- ";
    std::cout  << std::setw(width) << " ";
    if(h12->partOfFullEdge())
        std::cout << std::setw(width) << h12->halfedge->triangle->id;
    else
        std::cout << " -- ";
    std::cout  << std::setw(width) << " ";
    if(h20->partOfFullEdge())
        std::cout << std::setw(width) << h20->halfedge->triangle->id;
    else
        std::cout << " -- ";

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

