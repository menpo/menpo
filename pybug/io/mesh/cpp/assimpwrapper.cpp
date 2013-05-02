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
        std::cout << "Something went wrong" << std::endl;
    }
    else {
        //std::cout << "This scene has " << scene->mNumMeshes
        //<< " meshes on it." << std::endl;
        //std::cout << "This scene has " << scene->mNumTextures
        //<< " textures on it." << std::endl;
        std::cout << "This scene has " << scene->mNumMaterials
                  << " materials on it." << std::endl;
        for(int i = 0; i < scene->mNumMeshes; i++)
            mesh_checker(scene->mMeshes[i]);
        //for(int i = 0; i < scene->mNumMaterials; i++)
        //    material_checker(scene->mMaterials[i]);
    }
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

std::string AssimpWrapper::texture_path(unsigned int mesh_no){
    // get the mateial that is attached to this mesh
    unsigned int mat_index = scene->mMeshes[mesh_no]->mMaterialIndex;
    std::cout << mat_index;
    aiMaterial* mat = scene->mMaterials[1];
    return diffuse_texture_path_on_material(mat);
}
