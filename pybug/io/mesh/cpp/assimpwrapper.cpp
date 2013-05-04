#include <iostream>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "assimpwrapper.h"
#include "assimputils.h"


AssimpWrapper::AssimpWrapper(std::string path) {
    // we only want raw info - don't care about a lot of the stuff that assimp
    // could give us back. Here we disable all that stuff.
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
            aiComponent_NORMALS                 |
            aiComponent_TANGENTS_AND_BITANGENTS |
            aiComponent_ANIMATIONS              |
            aiComponent_BONEWEIGHTS             |
            aiComponent_COLORS                  |
            aiComponent_LIGHTS                  |
            aiComponent_CAMERAS                 |
            0);
    const aiScene* aiscene = importer.ReadFile(path,
              aiProcess_RemoveComponent       |
              aiProcess_JoinIdenticalVertices |
              aiProcess_Triangulate           |
              aiProcess_FindDegenerates       |
              aiProcess_SortByPType);
    if(!aiscene) {
        throw "We couldn't find a scene.";
    }
    p_scene = new AssimpScene(aiscene);
}

AssimpScene* AssimpWrapper::get_scene() {
    return p_scene;
}


//void AssimpWrapper::verify_state()
//{
//    /* Checks that we have:
//     * one and only one mesh that is composed of solely triangles.
//     * if that mesh has texture coords:
//     *    we have one and only one texture path
//     * anything else raises an exception.
//     * Returns the mesh index of the triangle mesh
//     */
//    trimesh_index = 999;
//    bool has_tcoords = false;
//    for(int i = 0; i < n_meshes(); i++) {
//        aiMesh* mesh = scene->mMeshes[i];
//        // first check this mesh is only triangles.
//        bool has_points, has_lines, has_triangles, has_polygons;
//        mesh_flags(mesh, has_points, has_lines, has_triangles, has_polygons);
//        if (!(has_points || has_lines || has_polygons)) {
//            if(trimesh_index != 999) 
//                throw "Have two seperate trimeshes.";
//            else {
//                trimesh_index = i;
//                if(n_tcoord_sets(i) > 0)
//                    has_tcoords = true;
//            }
//        }
//    }
//    if(trimesh_index == 999)
//        throw "Never found a trimesh";
//    std::string path = texture_path();
//    if(path == NO_TEXTURE_PATH && has_tcoords)
//        throw "Importing mesh with tcoords but cant find texture";
//}
//

AssimpScene::AssimpScene(const aiScene* aiscene) {
    p_scene = aiscene;
    for(int i = 0; i < p_scene->mNumMeshes; i++) {
        meshes.push_back(new AssimpMesh(p_scene->mMeshes[i], this));
    }
}

unsigned int AssimpScene::n_meshes() {
    return p_scene->mNumMeshes;
}

std::string AssimpScene::texture_path(){
    // get the material that is attached to this mesh
    std::string path = NO_TEXTURE_PATH;
    for(int i = 0; i < p_scene->mNumMaterials; i++) {
        aiMaterial* mat = p_scene->mMaterials[i];
        std::string path = diffuse_texture_path_on_material(mat);
        if(path != NO_TEXTURE_PATH)
            return path;
    }
    return path;
}


AssimpMesh::AssimpMesh(aiMesh* mesh, AssimpScene* scene_in) {
    p_mesh = mesh;
    scene = scene_in;
}

unsigned int AssimpMesh::n_points(){
    return p_mesh->mNumVertices;
}

unsigned int AssimpMesh::n_faces(){
    return p_mesh->mNumFaces;
}

unsigned int AssimpMesh::n_tcoord_sets(){
    bool has_tcoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];
    return tcoords_mask(p_mesh , has_tcoords);
}

void AssimpMesh::points(double* points){
    read_points(p_mesh, points);
}

void AssimpMesh::trilist(unsigned int* trilist){
    read_trilist(p_mesh, trilist);
}

void AssimpMesh::tcoords(int pindex, double* tcoords){
    read_tcoords(p_mesh, pindex, tcoords);
}

bool AssimpMesh::is_trimesh(){
    bool has_points, has_lines, has_triangles, has_polygons;
    mesh_flags(p_mesh, has_points, has_lines, has_triangles, has_polygons);
    if (!(has_points || has_lines || has_polygons) && has_triangles) 
        return true;
    else
        return false;
}

bool AssimpMesh::is_pointcloud(){
    bool has_points, has_lines, has_triangles, has_polygons;
    mesh_flags(p_mesh, has_points, has_lines, has_triangles, has_polygons);
    if (!(has_points || has_lines || has_polygons || has_triangles)) 
        return true;
    else
        return false;
}

