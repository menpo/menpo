#pragma once

#include <string>
#include <assimp/Importer.hpp>

class aiScene;

class AssimpWrapper{
    Assimp::Importer importer;
    const aiScene* scene;
    void verify_state();
    public:
    unsigned int trimesh_index;
    AssimpWrapper(std::string path);
    unsigned int n_meshes();
    unsigned int n_points(unsigned int mesh_no);
    unsigned int n_tcoord_sets(unsigned int mesh_no);
    unsigned int n_tris(unsigned int mesh_no);
    std::string texture_path();
    void import_points(unsigned int mesh_no, double* points);
    void import_trilist(unsigned int mesh_no, unsigned int* trilist);
    void import_tcoords(unsigned int mesh_no, int pindex, double* tcoords);
};

