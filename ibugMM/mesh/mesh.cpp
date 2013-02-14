#include <iostream>
#include <vector>
#include <cmath>
#include "mesh.h"
// MeshAttributes
#include "triangle.h"
#include "vertex.h"
#include "vec3.h"
#include "halfedge.h"

const double PI_2 = 2.0*atan(1.0);

Mesh::Mesh(double *coords_in, unsigned n_vertices_in,
        unsigned *tri_index_in, unsigned n_triangles_in) {
    coords = coords_in;
    n_vertices = n_vertices_in;
    tri_index = tri_index_in;
    n_triangles = n_triangles_in;
    // set the no. of full/half edges to 0
    // (on creation halfedges pairs will increment these as suitable)
    n_full_edges = 0;
    n_half_edges = 0;
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

Mesh::~Mesh() {
    triangles.clear();
    // now all the triangles are clear we're good to delete all the vertices
    // we initially made
    vertices.clear();
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

void Mesh::laplacian(unsigned* i_sparse, unsigned* j_sparse,
        double* v_sparse, LaplacianWeightType weight_type) {
    // pointers to structures used to define a sparse matrix of doubles
    // where the k'th value of each array is treated to mean:
    // sparse_matrix[i_sparse[k]][j_sparse[k]] = v_sparse[k]

    // we expect that the attachments at i_sparse, j_sparse
    // and v_sparse have already been set to the correct 
    // dimentions before this call 
    // (each should be of length 2*n_half_edges)
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

void Mesh::verifyMesh() {
    std::vector<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++) {
        // TODO add more verificaton stages here
        (*v)->verifyHalfEdgeConnectivity();
    }
}

MeshAttribute::MeshAttribute(Mesh* meshIn) {
    mesh = meshIn;
}

void Mesh::addEdge(HalfEdge* halfedge) {
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

void Mesh::calculateGradient(double* v_scalar_field, double* t_vector_gradient)
{
    std::vector<Triangle*>::iterator t;
    for(t = triangles.begin(); t != triangles.end(); t++)
        ((*t)->gradient(v_scalar_field)).writeOutTo(&t_vector_gradient[((*t)->id)*3]);
}

void Mesh::calculateDivergence(double* t_vector_field, double* v_scalar_divergence)
{
    std::vector<Vertex*>::iterator v;
    for(v = vertices.begin(); v != vertices.end(); v++)
        (*v)->divergence(t_vector_field, v_scalar_divergence);
    //vertices[0]->divergence(t_vector_field, v_scalar_divergence);
}

void Mesh::triangleAreas(double* areas)
{
    std::vector<Triangle*>::iterator t;
    for(t = triangles.begin(); t != triangles.end(); t++)
        areas[(*t)->id] = (*t)->area();
}

double angleBetweenVerticies(Vertex* A, Vertex* B, Vertex* C)
{
    //std::cout << "Angle for A = " << *A << " B = " << *B << " C = " << *C << std::endl;
    Vec3 a = *A - *B;
    Vec3 b = *C - *B;
    //std::cout << a << "   " << b << std::cout;
    a.normalize();
    b.normalize();
    return std::acos(a.dot(b));
}

double cotOfAngle(double theta)
{
    return std::tan(PI_2 - theta);
}

double Mesh::meanEdgeLength()
{
    double edge_length = 0;
    std::set<HalfEdge*>::iterator he;
    for(he = edges.begin(); he != edges.end(); he++)
        edge_length += (*he)->length();
    return edge_length/(n_half_edges - n_full_edges);
}

