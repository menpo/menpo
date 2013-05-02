#include <iostream>
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
    scene = importer.ReadFile(path,
            aiProcess_RemoveComponent       |
            aiProcess_JoinIdenticalVertices |
            aiProcess_Triangulate           |
            aiProcess_FindDegenerates       |
            aiProcess_SortByPType);
    if(!scene) {
        throw "We couldn't find a scene.";
    }
    verify_state();
}


void AssimpWrapper::verify_state()
{
    /* Checks that we have:
     * one and only one mesh that is composed of solely triangles.
     * if that mesh has texture coords:
     *    we have one and only one texture path
     * anything else raises an exception.
     * Returns the mesh index of the triangle mesh
     */
    trimesh_index = 999;
    bool has_tcoords = false;
    for(int i = 0; i < n_meshes(); i++) {
        aiMesh* mesh = scene->mMeshes[i];
        // first check this mesh is only triangles.
        bool has_points, has_lines, has_triangles, has_polygons;
        mesh_flags(mesh, has_points, has_lines, has_triangles, has_polygons);
        if (!(has_points || has_lines || has_polygons)) {
            if(trimesh_index != 999) 
                throw "Have two seperate trimeshes.";
            else {
                trimesh_index = i;
                if(n_tcoord_sets(i) > 0)
                    has_tcoords = true;
            }
        }
    }
    if(trimesh_index == 999)
        throw "Never found a trimesh";
    std::string path = texture_path();
    if(path == NO_TEXTURE_PATH && has_tcoords)
        throw "Importing mesh with tcoords but cant find texture";
}

unsigned int AssimpWrapper::n_meshes(){
    return scene->mNumMeshes;
}

unsigned int AssimpWrapper::n_points(unsigned int mesh_no){
    return scene->mMeshes[mesh_no]->mNumVertices;
}

unsigned int AssimpWrapper::n_tris(unsigned int mesh_no){
    return scene->mMeshes[mesh_no]->mNumFaces;
}

unsigned int AssimpWrapper::n_tcoord_sets(unsigned int mesh_no){
    bool has_tcoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];
    return tcoords_mask(scene->mMeshes[mesh_no], has_tcoords);
}

std::string AssimpWrapper::texture_path(){
    // get the mateial that is attached to this mesh
    std::string path = NO_TEXTURE_PATH;
    for(int i = 0; i < scene->mNumMaterials; i++) {
        aiMaterial* mat = scene->mMaterials[i];
        std::string path = diffuse_texture_path_on_material(mat);
        if(path != NO_TEXTURE_PATH)
            return path;
    }
    return path;
}

void AssimpWrapper::import_points(unsigned int mesh_no, double* points){
    read_points(scene->mMeshes[mesh_no], points);

}

void AssimpWrapper::import_trilist(unsigned int mesh_no, unsigned int* trilist){
    read_trilist(scene->mMeshes[mesh_no], trilist);
}

void AssimpWrapper::import_tcoords(unsigned int mesh_no, int pindex, double* tcoords){
    read_tcoords(scene->mMeshes[mesh_no], pindex, tcoords);
}

