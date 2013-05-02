#pragma once

#include <string>
#include <assimp/Importer.hpp>

class aiScene;

class AssimpWrapper{
    Assimp::Importer importer;
    const aiScene* scene;
    public:
    AssimpWrapper(std::string path);
    unsigned int n_meshes();
    unsigned int n_points(unsigned int mesh_no);
    unsigned int n_tcoord_sets(unsigned int mesh_no);
    unsigned int n_tris(unsigned int mesh_no);
    std::string texture_path(unsigned int mesh_no);
};
