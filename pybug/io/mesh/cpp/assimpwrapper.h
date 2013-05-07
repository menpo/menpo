#pragma once

#include <string>
#include <vector>
#include <assimp/Importer.hpp>
const std::string NO_TEXTURE_PATH = "NO_TEXTURE_PATH";

// forward declarations
class aiScene;
class aiMesh;
class aiMaterial;
class AssimpMesh;
class AssimpScene;
class AssimpImporter;


// *************** IMPORTER *************** //
class AssimpImporter{
    Assimp::Importer importer;
    AssimpScene* p_scene;

    public:
    AssimpImporter(std::string path);
    ~AssimpImporter();
    AssimpScene* get_scene();
};


// *************** SCENE *************** //
class AssimpScene{
    const aiScene* p_scene;

    public:
    std::vector<AssimpMesh*> meshes;
    AssimpScene(const aiScene* scene);
    unsigned int n_meshes();
    std::string texture_path();
};


// *************** MESH *************** //
class AssimpMesh{
    aiMesh* p_mesh;
    AssimpScene* scene;

    public:
    AssimpMesh(aiMesh* mesh, AssimpScene* scene);
    unsigned int n_points();
    unsigned int n_faces();
    unsigned int n_tcoord_sets();
    bool has_points();
    bool has_lines();
    bool has_triangles();
    bool has_polygons();
    bool is_trimesh();
    bool is_pointcloud();
    void points(double* points);
    void trilist(unsigned int* trilist);
    void tcoords(int index, double* tcoords);
    void tcoords_with_alpha(int index, double* tcoords);
};


// *************** HELPER ROUTINES *************** //
unsigned int tcoords_mask(aiMesh* mesh, bool* has_tcoords);
std::string diffuse_texture_path_on_material(aiMaterial* mat);

