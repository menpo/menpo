#include <iostream>
#include "assimputils.h"
#include <assimp/scene.h>


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

void mesh_flags(aiMesh* mesh, bool& has_points, bool& has_lines, 
                bool& has_triangles, bool& has_polygons) {
    unsigned int types_enum = mesh->mPrimitiveTypes;
    has_points = aiPrimitiveType_POINT & types_enum;
    has_lines = aiPrimitiveType_LINE & types_enum;
    has_triangles = aiPrimitiveType_TRIANGLE & types_enum;
    has_polygons = aiPrimitiveType_POLYGON & types_enum;
}

void read_points(aiMesh* mesh, double* points) {
    for(int i = 0; i < mesh->mNumVertices; i++) {
        aiVector3D point = mesh->mVertices[i];
        points[3*i] = point.x;
        points[3*i + 1] = point.y;
        points[3*i + 2] = point.z;
    }
}

void read_trilist(aiMesh* mesh, unsigned int* trilist) {
    for(int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        if(face.mNumIndices != 3)
            std::cout << "WARNING: Face " << i << " has " <<
                face.mNumIndices << " indices." << std::endl;
        trilist[3*i] = face.mIndices[0];
        trilist[3*i + 1] = face.mIndices[1];
        trilist[3*i + 2] = face.mIndices[2];
    }
}

void read_tcoords(aiMesh* mesh, int pindex, double* tcoords) {
    aiVector3D* tcoord_array = mesh->mTextureCoords[pindex];
    for(int i = 0; i < mesh->mNumVertices; i++) {
        aiVector3D tcoord = tcoord_array[i];
        tcoords[3*i] = tcoord.x;
        tcoords[3*i + 1] = tcoord.y;
        tcoords[3*i + 2] = tcoord.z;
    }
}

void mesh_checker(aiMesh* mesh) {
    bool has_points, has_lines, has_triangles, has_polygons;
    mesh_flags(mesh, has_points, has_lines, has_triangles, has_polygons);
    if (has_points || has_lines || has_polygons)
        std::cout << "This mesh has problematic data" << std::endl;
}

void mesh_import(aiMesh* mesh, double* points, unsigned int* trilist, 
                 double** tcoords) {
    read_points(mesh, points);
    read_trilist(mesh, trilist);
    bool has_tcoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];
    tcoords_mask(mesh, has_tcoords);
    unsigned int tcoords_index = 0;
    for(int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; i++) {
        if(has_tcoords) {
            read_tcoords(mesh, i, tcoords[tcoords_index]);
            tcoords_index++;
        }
    }
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

