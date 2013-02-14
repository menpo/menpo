#include <iostream>
#include <iomanip>
#include "halfedge.h"
#include "triangle.h"
#include "vertex.h"

Triangle::Triangle(Mesh* mesh_in, unsigned tri_id, Vertex* v0_in,
        Vertex* v1_in, Vertex* v2_in) : MeshAttribute(mesh_in, tri_id) {
    v0 = v0_in;
    v1 = v1_in;
    v2 = v2_in;
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

Triangle::~Triangle() {}

void Triangle::reduce_scalar_to_vertices(double* triangle_scalar,
        double* vertex_scalar) {
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

void Triangle::status() {
    std::cout << "    TRIANGLE " << id << "        " << std::endl;
    HalfEdge* h01 = v0->halfedge_on_triangle(this);
    HalfEdge* h12 = v1->halfedge_on_triangle(this);
    HalfEdge* h20 = v2->halfedge_on_triangle(this);
    unsigned width = 12;
    std::cout  << std::setw(width) << "V0(" << v0->id << ")";
    if (h01->part_of_fulledge()) {
        std::cout << "============";
    }
    else {
        std::cout << "------------";
    }
    std::cout  << std::setw(width) << "V1(" << v1->id << ")";
    if (h12->part_of_fulledge()) {
        std::cout << "============";
    }
    else {
        std::cout << "------------";
    }
    std::cout  << std::setw(width) << "V2(" << v2->id << ")";
    if (h20->part_of_fulledge()) {
        std::cout << "============";
    }
    else {
        std::cout << "------------";
    }
    std::cout << std::setw(width) << "V0(" << v0->id << ")" << std::endl;

    std::cout  << std::setw(width) << " ";
    if (h01->part_of_fulledge()) {
        std::cout  << std::setw(width) << h01->halfedge->triangle->id;
    }
    else {
        std::cout << " -- ";
    }
    std::cout  << std::setw(width) << " ";
    if (h12->part_of_fulledge()) {
        std::cout << std::setw(width) << h12->halfedge->triangle->id;
    }
    else {
        std::cout << " -- ";
    }
    std::cout  << std::setw(width) << " ";
    if (h20->part_of_fulledge()) {
        std::cout << std::setw(width) << h20->halfedge->triangle->id;
    }
    else {
        std::cout << " -- ";
    }
}

