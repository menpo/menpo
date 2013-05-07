#include <iostream>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "assimpwrapper.h"


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

AssimpWrapper::~AssimpWrapper(){
    delete p_scene;
}

AssimpScene* AssimpWrapper::get_scene() {
    return p_scene;
}



AssimpScene::AssimpScene(const aiScene* aiscene) {
    p_scene = aiscene;
    for(unsigned int i = 0; i < p_scene->mNumMeshes; i++) {
        meshes.push_back(new AssimpMesh(p_scene->mMeshes[i], this));
    }
}

unsigned int AssimpScene::n_meshes() {
    return p_scene->mNumMeshes;
}

std::string AssimpScene::texture_path(){
    // get the material that is attached to this mesh
    std::string path = NO_TEXTURE_PATH;
    for(unsigned int i = 0; i < p_scene->mNumMaterials; i++) {
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
    return tcoords_mask(p_mesh, has_tcoords);
}

bool AssimpMesh::has_points(){
    return aiPrimitiveType_POINT & p_mesh->mPrimitiveTypes;
}

bool AssimpMesh::has_lines(){
    return aiPrimitiveType_LINE & p_mesh->mPrimitiveTypes;
}

bool AssimpMesh::has_triangles(){
    return aiPrimitiveType_TRIANGLE & p_mesh->mPrimitiveTypes;
}

bool AssimpMesh::has_polygons(){
    return aiPrimitiveType_POLYGON & p_mesh->mPrimitiveTypes;
}

bool AssimpMesh::is_pointcloud(){
    return (!(has_lines() || has_polygons() || has_triangles()));
}

bool AssimpMesh::is_trimesh(){
    return (!(has_points() || has_lines() || has_polygons()) 
            && has_triangles());
}

void AssimpMesh::points(double* points){
    for(unsigned int i = 0; i < p_mesh->mNumVertices; i++) {
        aiVector3D point = p_mesh->mVertices[i];
        points[3*i] = point.x;
        points[3*i + 1] = point.y;
        points[3*i + 2] = point.z;
    }
}

void AssimpMesh::trilist(unsigned int* trilist){
    // it is YOUR responsibility to ensure this
    // mesh contains only triangles before calling this method.
    for(unsigned int i = 0; i < p_mesh->mNumFaces; i++) {
        aiFace face = p_mesh->mFaces[i];
        trilist[3*i] = face.mIndices[0];
        trilist[3*i + 1] = face.mIndices[1];
        trilist[3*i + 2] = face.mIndices[2];
    }
}

void AssimpMesh::tcoords(int index, double* tcoords){
    /* Reads the (s,t) tcoords, removing the alpha channel component.
     * expects tcoords to be a C contiguous array of size
     * (n_points, 2)
     */
    aiVector3D* tcoord_array = p_mesh->mTextureCoords[index];
    for(unsigned int i = 0; i < p_mesh->mNumVertices; i++) {
        aiVector3D tcoord = tcoord_array[i];
        tcoords[2*i] = tcoord.x;
        tcoords[2*i + 1] = tcoord.y;
    }
}

void AssimpMesh::tcoords_with_alpha(int index, double* tcoords){
    /* Reads the tcoords, keeping the alpha channel component.
     * expects tcoords to be a C contiguous array of size
     * (n_points, 3)
     */
    aiVector3D* tcoord_array = p_mesh->mTextureCoords[index];
    for(unsigned int i = 0; i < p_mesh->mNumVertices; i++) {
        aiVector3D tcoord = tcoord_array[i];
        tcoords[3*i] = tcoord.x;
        tcoords[3*i + 1] = tcoord.y;
        tcoords[3*i + 2] = tcoord.z;
    }
}

// ***** HELPER ROUTINES ***** //

unsigned int tcoords_mask(aiMesh* mesh, bool* has_tcoords) {
    int tcoords_counter = 0;
    for(int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; i++) {
        if(mesh->HasTextureCoords(i)) {
            has_tcoords[i] = true;
            tcoords_counter++;
        }
        else
            has_tcoords[i] = false;
    }
    return tcoords_counter;
}

std::string diffuse_texture_path_on_material(aiMaterial* mat) {
    aiString path;
    //std::cout << mat->mNumProperties << std::endl;
    aiString name;
    mat->Get(AI_MATKEY_NAME, name);
    //std::cout << "Material name is " << name.C_Str() << std::endl;
    //std::cout << "Material path is " << path.C_Str() << std::endl;
    unsigned int texture_count = mat->GetTextureCount(aiTextureType_DIFFUSE);
    if(texture_count == 1) {
        mat->GetTexture(aiTextureType_DIFFUSE, 0, &path, 
                        NULL, NULL, NULL, NULL, NULL);
        const char* c_path = path.C_Str();
        std::string cpp_path(c_path);
        return cpp_path;
    }
    else
        return NO_TEXTURE_PATH;
}

