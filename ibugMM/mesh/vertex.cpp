#include <iostream>
#include <ostream>
#include <assert.h>
#include "vertex.h"
#include "halfedge.h"
#include "triangle.h"

Vertex::Vertex(Mesh* mesh_in, unsigned vertex_id,
        double* coords_in): MeshAttribute(mesh_in, vertex_id) {
    coords = coords_in;
}

Vertex::~Vertex() {
    halfedges.clear();
}

HalfEdge* Vertex::add_halfedge_to(Vertex* vertex, Triangle* triangle,
        unsigned id_on_tri_of_v0) {
    // returns the created half edge so it can be attached to the triangle if
    // so desired
    HalfEdge* he = halfedge_to_vertex(vertex);
    if(he == NULL) {
        HalfEdge* halfedge = new HalfEdge(this->mesh, this, vertex, triangle,
                id_on_tri_of_v0);
        halfedges.insert(halfedge);
        return halfedge;
    }
    else {
        std::cout << "CHIRAL CONSISTENCY: FAIL"
            << std::endl;
        std::cout << "    V" << id << " already has a half edge to V" <<
            vertex->id << " on triangle T" << he->triangle->id <<
            " yet triangle T" << triangle->id <<
            " is trying to create another one." << std::endl;
        std::cout << "    This means that one of the two triangles listed ";
        std::cout << "above has flipped normals, or the two triangles";
        std::cout << " are \n    a repeat of each other." << std::endl;
        std::cout << "    This repeated halfedge will not be created so ";
        std::cout << "expect segfaults if you try to continue." << std::endl;
        return NULL;
    }
}

void Vertex::add_triangle(Triangle* triangle) {
    triangles.insert(triangle);
}

void Vertex::add_vertex(Vertex* vertex) {
    vertices.insert(vertex);
}

HalfEdge* Vertex::halfedge_on_triangle(Triangle* triangle) {
    std::set<HalfEdge*>::iterator he;
    for(he = halfedges.begin(); he != halfedges.end(); he++) {
        if((*he)->triangle == triangle) {
            return *he;
        }
    }
    //std::cout << "V:" << this << " does not have a HE to V:" << vertex << std::endl;
    return NULL;
}

HalfEdge* Vertex::halfedge_to_vertex(Vertex* vertex) {
    std::set<HalfEdge*>::iterator he;
    for(he = halfedges.begin(); he != halfedges.end(); he++) {
        if((*he)->v1 == vertex) {
            return *he;
        }
    }
    return NULL;
}

HalfEdge* Vertex::halfedge_to_or_from_vertex(Vertex* vertex) {
    HalfEdge* he = halfedge_to_vertex(vertex);
    if (he != NULL) {
        return he;
    }
    else {
        he = vertex->halfedge_to_vertex(this);
    }
    if (he != NULL) {
        return he;
    }
    else {
        std::cout << "WARNING: could not find a half edge to or from"
            << std::endl;
        return NULL;
    }
}

void Vertex::laplacian(unsigned* i_sparse, unsigned* j_sparse,
        double* w_sparse, unsigned& sparse_pointer,
        LaplacianWeightType weight_type) {
    // sparse_pointer points into how far into the sparse_matrix structures
    // we should be recording results for this vertex
    unsigned i = id;
    std::set<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++) {
        unsigned j = (*v)->id;
        //if(i < j)
        //{
        HalfEdge* he = halfedge_to_or_from_vertex(*v);
        double w_ij = 0;
        switch(weight_type) {
            case distance:
                w_ij = laplacian_distance_weight(he);
                break;
            case combinatorial:
                w_ij = 1;
                break;
        }
        i_sparse[sparse_pointer] = i;
        j_sparse[sparse_pointer] = j;
        w_sparse[sparse_pointer] = -w_ij;
        sparse_pointer++;
        // and record the other way for free (Laplacian is symmetrical)
        //i_sparse[sparse_pointer] = j;
        //j_sparse[sparse_pointer] = i;
        //w_sparse[sparse_pointer] = -w_ij;
        //sparse_pointer++;
        w_sparse[i] += w_ij;
        // += cotOp to the i'th\'th position twice (for both times)
        //w_sparse[j] += w_ij;
        //}
        // else:no point calculating this point - as we know the Laplacian is symmetrical
    }
}

double Vertex::laplacian_distance_weight(HalfEdge* he) {
    double length = he->length();
    return 1.0/(length*length);
}

void Vertex::cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
        double* w_sparse, unsigned& sparse_pointer,
        double* cot_per_tri_vertex) {
    unsigned i = id;
    std::set<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++) {
        unsigned j = (*v)->id;
        HalfEdge* he = halfedge_to_or_from_vertex(*v);
        double w_ij = cot_per_tri_vertex[(he->triangle->id*3) + he->v2_tri_i];
        if (he->part_of_fulledge()) {
            w_ij += cot_per_tri_vertex[(he->halfedge->triangle->id*3) +
                he->halfedge->v2_tri_i];
        }
        else {
            //w_ij += w_ij;
        }
        i_sparse[sparse_pointer] = i;
        j_sparse[sparse_pointer] = j;
        w_sparse[sparse_pointer] = -w_ij;
        sparse_pointer++;
        w_sparse[i] += w_ij;
    }
}

void Vertex::verify_halfedge_connectivity() {
    std::set<HalfEdge*>::iterator he;
    for(he = halfedges.begin(); he != halfedges.end(); he++) {
        Triangle* triangle = (*he)->triangle;
        Vertex* t_v0 = triangle->v0;
        Vertex* t_v1 = triangle->v1;
        Vertex* t_v2 = triangle->v2;
        if(t_v0 != this && t_v1 != this && t_v2 != this)
            std::cout << "this halfedge does not live on it's triangle!"
                << std::endl;
        if((*he)->v0 != this)
            std::cout << "half edge errornously connected" << std::endl;
        if((*he)->ccw_around_tri()->ccw_around_tri()->v1 != (*he)->v0)
            std::cout << "cannie spin raarnd the triangle like man!"
                << std::endl;
        if((*he)->part_of_fulledge()) {
            if((*he)->halfedge->v0 != (*he)->v1 || (*he)->halfedge->v1 != (*he)->v0)
                std::cout << "some half edges aren't paired up !"
                    << std::endl;
        }
    }
}

void Vertex::status() {
    std::cout << "V" << id << std::endl;
    std::set<HalfEdge*>::iterator he;
    for(he = halfedges.begin(); he != halfedges.end(); he++) {
        std::cout << "|" ;
        if ((*he)->part_of_fulledge()) {
            std::cout << "=";
        }
        else {
            std::cout << "-";
        }
        std::cout << "V" << (*he)->v1->id;
        std::cout << " (T" << (*he)->triangle->id;
        if ((*he)->part_of_fulledge()) {
            std::cout << "=T" << (*he)->halfedge->triangle->id;
        }
        std::cout << ")" << std::endl;
    }
}

void Vertex::test_contiguous(std::set<Vertex*>* vertices_visited){
    std::set<Vertex*>::iterator v;
    for (v = vertices.begin(); v != vertices.end(); v++) {
        if (vertices_visited->erase(*v)) {
            (*v)->test_contiguous(vertices_visited);
        }
    }
}


std::ostream& operator<<(std::ostream& out, const Vertex& v) {
    out << "V:" << v.id << " (" << v.coords[0] << ","
        << v.coords[1] << "," << v.coords[2] << ")";
    return out;
}

