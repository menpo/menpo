#pragma once

#include <string>
#include <vector>
#include <assimp/Importer.hpp>

// forward declarations
class aiScene;
class aiMesh;
class AssimpMesh;
class AssimpScene;
class AssimpWrapper;


class AssimpWrapper{
    AssimpScene* p_scene;
    Assimp::Importer importer;

    public:
    AssimpWrapper(std::string path);
    AssimpScene* get_scene();
};


class AssimpScene{
    const aiScene* p_scene;

    public:
    std::vector<AssimpMesh*> meshes;
    AssimpScene(const aiScene* scene);
    unsigned int n_meshes();
    std::string texture_path();
};


class AssimpMesh{
    aiMesh* p_mesh;
    AssimpScene* scene;

    public:
    AssimpMesh(aiMesh* mesh, AssimpScene* scene);
    bool is_trimesh();
    bool is_pointcloud();
    unsigned int n_points();
    unsigned int n_tcoord_sets();
    unsigned int n_faces();
    void points(double* points);
    void trilist(unsigned int* trilist);
    void tcoords(int pindex, double* tcoords);
};



