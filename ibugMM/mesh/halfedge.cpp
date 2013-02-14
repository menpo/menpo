#include <iostream>
#include <cmath>
#include "halfedge.h"
#include "triangle.h"
#include "vertex.h"


HalfEdge::HalfEdge(Mesh* mesh_in, Vertex* v0_in, Vertex* v1_in,
        Triangle* triangle_in,
        unsigned id_on_tri_of_v0) : MeshAttribute( mesh_in, id_on_tri_of_v0) {
    mesh->n_halfedges++;
    v0 = v0_in;
    v1 = v1_in;
    triangle = triangle_in;
    halfedge = v1->halfedge_to_vertex(v0);
    switch (id_on_tri_of_v0) {
        case 0:
            v0_tri_i = 0;
            v1_tri_i = 1;
            v2_tri_i = 2;
            v2 = triangle->v2;
            break;
        case 1:
            v0_tri_i = 1;
            v1_tri_i = 2;
            v2_tri_i = 0;
            v2 = triangle->v0;
            break;
        case 2:
            v0_tri_i = 2;
            v1_tri_i = 0;
            v2_tri_i = 1;
            v2 = triangle->v1;
            break;
    }
    if (halfedge != NULL) {
        // setting opposite halfedge to me
        halfedge->halfedge = this;
        mesh->n_fulledges++;
    }
    else {
        // first time weve encountered this
        mesh_in->add_edge(this);
    }
}

HalfEdge::~HalfEdge(){}

bool HalfEdge::part_of_fulledge() {
    if (halfedge != NULL) {
        return true;
    }
    else {
        return false;
    }
}

HalfEdge* HalfEdge::ccw_around_tri() {
    HalfEdge* he;
    if (v1->id == triangle->v0->id) {
        he = triangle->e0;
    }
    else if (v1->id == triangle->v1->id) {
        he = triangle->e1;
    }
    else if (v1->id == triangle->v2->id) {
        he = triangle->e2;
    }
    else {
        std::cout << "ERROR: cannot find HE!" << std::endl;
    }
    return he;
}

double HalfEdge::length() {
    // TODO actually calcuate this
    std::cout << "This isn't actually calculating the length" << std::endl;
    return 1.0;
}

