#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include "mesh.h"
#include "triangle.h"
#include "vertex.h"
#include "halfedge.h"

Mesh::Mesh(double *coords_in, unsigned n_vertices_in,
        unsigned *tri_index_in, unsigned n_triangles_in) {
    coords = coords_in;
    n_vertices = n_vertices_in;
    tri_index = tri_index_in;
    n_triangles = n_triangles_in;
    // set the no. of full/half edges to 0
    // (on creation halfedges pairs will increment these as suitable)
    n_fulledges = 0;
    n_halfedges = 0;
    // build a Vertex object for each coord set passed in
    for(unsigned i = 0; i < n_vertices; i++) {
        vertices.push_back(new Vertex(this, i, &coords[i*3]));
    }
    for(unsigned i = 0; i < n_triangles; i++) {
        // get the index into the vertex positions
        unsigned l = tri_index[i*3    ];
        unsigned m = tri_index[i*3 + 1];
        unsigned n = tri_index[i*3 + 2];
        // build a new triangle passing in the pointers to the vertices it will
        // be made from (the triangle in it's construction will build edges and
        // connect them)
        triangles.push_back(new Triangle(this, i, vertices[l],
                    vertices[m], vertices[n]));
    }
}

void Mesh::add_edge(HalfEdge* halfedge) {
    edges.insert(halfedge);
}

void Mesh::generate_edge_index(unsigned* edge_index) {
    std::set<HalfEdge*>::iterator he;
    unsigned count = 0;
    for(he = edges.begin(); he != edges.end(); he++, count++)
    {
        edge_index[count*2]     = (*he)->v0->id;
        edge_index[count*2 + 1] = (*he)->v1->id;
    }
}

Mesh::~Mesh() {
    triangles.clear();
    // now all the triangles are clear we're good to delete all the vertices
    // we initially made
    vertices.clear();
}

void Mesh::verify_mesh() {
    std::vector<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++) {
        (*v)->verify_halfedge_connectivity();
    }
    test_contiguous();
    test_chiral_consistency();
}

void Mesh::test_chiral_consistency() {
    std::cout << "CHIRALCONSISTENCY: ";
    std::set<HalfEdge*>::iterator edge;
    bool pass = true;
    unsigned fulledges_encountered = 0;
    unsigned halfedges_encountered = 0;
    for (edge = edges.begin(); edge != edges.end(); edge++) {
        halfedges_encountered++;
        if ((*edge)->part_of_fulledge()) {
            fulledges_encountered++;
            halfedges_encountered++;
            if ((*edge)->halfedge->v1 != (*edge)->v0 ||
                    (*edge)->halfedge->v0 != (*edge)->v1) {
                pass = false;
            }
        }
    }
    if (pass) {
        std::cout << "PASS" << std::endl;
    }
    else {
        std::cout << "FAIL" << std::endl;
    }
    std::cout << "EDGECOUNT: ";
    if (fulledges_encountered == n_fulledges &&
            halfedges_encountered == n_halfedges) {
        std::cout << "PASS" << std::endl;
    }
    else {
        std::cout << "FAIL" << std::endl;
    }
}

void Mesh::test_contiguous() {
    std::cout << "CONTIGUOUS: ";
    std::set<Vertex*>* vertices_visited = new std::set<Vertex*>(
            vertices.begin(), vertices.end());
    std::vector<Vertex*>::iterator v;
    v = vertices.begin();
    vertices_visited->erase(*v);
    (*v)->test_contiguous(vertices_visited);
    if (vertices_visited->empty()) {
        std::cout << "PASS" << std::endl;
    }
    else {
        std::cout << "FAIL" << std::endl;
        int num_unvisited = vertices_visited->size();
        std::cout << "  The following " << num_unvisited <<
            " vertices are not joined to V" << (*v)->id << ":";
        std::set<Vertex*>::iterator unref_v;
        int nl_count = 0;
        for (unref_v = vertices_visited->begin();
                unref_v != vertices_visited->end(); unref_v++, nl_count++) {
            if (!(nl_count % 8)) {
                std::cout << std::endl << "  ";
            }
            std::cout << "  V" << (*unref_v)->id << "   ";
        }
        std::cout << std::endl;
        int regions_count = 0;
        while (!vertices_visited->empty()) {
            regions_count++;
            unref_v = vertices_visited->begin();
            vertices_visited->erase(*unref_v);
            (*unref_v)->test_contiguous(vertices_visited);
        }
        std::cout << "  These unjoined vertices are grouped into " <<
            regions_count << " contiguous regions." << std::endl;
    }
    delete vertices_visited;
}

void Mesh::laplacian(unsigned* i_sparse, unsigned* j_sparse,
        double* v_sparse, LaplacianWeightType weight_type) {
    // pointers to structures used to define a sparse matrix of doubles
    // where the k'th value of each array is treated to mean:
    // sparse_matrix[i_sparse[k]][j_sparse[k]] = v_sparse[k]

    // we expect that the attachments at i_sparse, j_sparse
    // and v_sparse have already been set to the correct
    // dimentions before this call
    // (each should be of length 2*n_halfedges)
    // the first n_coord entries are the diagonals. => the i'th
    // value of both i_sparse and j_sparse is just i
    for(unsigned int i = 0; i < n_vertices; i++) {
        i_sparse[i] = i;
        j_sparse[i] = i;
    }
    // set the sparse_pointer to the end of the diagonal elements
    unsigned sparse_pointer = n_vertices;
    // now loop through each vertex and call the laplacian method.
    // This method will populate the sparse matrix arrays with the
    // position and value that should be assiged to the matrix
    std::vector<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++) {
        (*v)->laplacian(i_sparse, j_sparse, v_sparse,
                sparse_pointer, weight_type);
    }
}

void Mesh::cotangent_laplacian(unsigned* i_sparse, unsigned* j_sparse,
        double* v_sparse, double* cotangents) {
    for(unsigned int i = 0; i < n_vertices; i++) {
        i_sparse[i] = i;
        j_sparse[i] = i;
    }
    unsigned sparse_pointer = n_vertices;
    std::vector<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++){
        (*v)->cotangent_laplacian(i_sparse, j_sparse, v_sparse,
                sparse_pointer, cotangents);
    }
}

void Mesh::reduce_tri_scalar_per_vertex_to_vertices(
        double* triangle_scalar_per_vertex, double* vertex_scalar) {
    // this one is for when we have a scalar value defined at each vertex of each triangle
    std::vector<Triangle*>::iterator t;
    for(t = triangles.begin(); t != triangles.end(); t++)
        (*t)->reduce_scalar_per_vertex_to_vertices(triangle_scalar_per_vertex, vertex_scalar);
}

void Mesh::reduce_tri_scalar_to_vertices(double* triangle_scalar, double* vertex_scalar) {
    // this one is for when we have a scalar value defined at each triangle and needs to be
    // applied to each vertex
    std::vector<Triangle*>::iterator t;
    for(t = triangles.begin(); t != triangles.end(); t++)
        (*t)->reduce_scalar_to_vertices(triangle_scalar, vertex_scalar);
}


MeshAttribute::MeshAttribute(Mesh* mesh_in, unsigned id_in) {
    mesh = mesh_in;
    id = id_in;
}

