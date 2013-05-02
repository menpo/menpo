#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>

int tcoords_mask(aiMesh* mesh, bool* has_tcoords)
{
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

void counters(aiMesh* mesh, int& n_points, int& n_faces, int& n_tcoords,
        bool& has_points, bool& has_lines, bool& has_triangles,
        bool& has_polygons)
{
    if(mesh->HasFaces())
        n_faces = mesh->mNumFaces;
    else
        n_faces = 0;
    if(mesh->HasPositions())
        n_points = mesh->mNumVertices;
    else
        n_points = 0;
    bool has_tcoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];
    n_tcoords = tcoords_mask(mesh, has_tcoords);
    unsigned int types_enum = mesh->mPrimitiveTypes;
    has_points = aiPrimitiveType_POINT & types_enum;
    has_lines = aiPrimitiveType_LINE & types_enum;
    has_triangles = aiPrimitiveType_TRIANGLE & types_enum;
    has_polygons = aiPrimitiveType_POLYGON & types_enum;
}


void read_points(aiMesh* mesh, double* points)
{
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        aiVector3D point = mesh->mVertices[i];
        points[3*i] = point.x;
        points[3*i + 1] = point.y;
        points[3*i + 2] = point.z;
    }
}

void read_trilist(aiMesh* mesh, unsigned int* trilist)
{
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

void read_tcoords(aiMesh* mesh, int pindex, double* tcoords)
{
    aiVector3D* tcoord_array = mesh->mTextureCoords[pindex];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        aiVector3D tcoord = tcoord_array[i];
        tcoords[3*i] = tcoord.x;
        tcoords[3*i + 1] = tcoord.y;
        tcoords[3*i + 2] = tcoord.z;
    }
}

void mesh_checker(aiMesh* mesh)
{
    int n_points;
    int n_faces;
    int n_tcoords;
    bool has_points, has_lines, has_triangles, has_polygons;
    counters(mesh, n_points, n_faces, n_tcoords, has_points, has_lines,
            has_triangles, has_polygons);
    std::cout << "Mesh has " << n_faces << " faces." << std::endl;
    std::cout << "Mesh has " << n_points << " points." << std::endl;
    std::cout << "Mesh has " << n_tcoords<< " sets of tcoords." << std::endl;
    std::cout << "Includes points?    " << has_points << std::endl;
    std::cout << "Includes lines?     " << has_lines << std::endl;
    std::cout << "Includes triangles? " << has_triangles << std::endl;
    std::cout << "Includes polygons?  " << has_polygons << std::endl;
    std::cout << "Mesh uses material " << mesh->mMaterialIndex << std::endl;
}

void material_checker(aiMaterial* mat)
{
    std::cout << mat->mNumProperties << std::endl;
    aiString name;
    aiString path;
    mat->Get(AI_MATKEY_NAME, name);
    unsigned int texture_count = mat->GetTextureCount(aiTextureType_DIFFUSE);
    if(texture_count == 1)
        mat->GetTexture(aiTextureType_DIFFUSE, 0, &path, NULL, NULL, NULL, NULL, NULL);
    std::cout << "Material name is " << name.C_Str() << std::endl;
    std::cout << "Material path is " << path.C_Str() << std::endl;
}

int main()
{
    //const char* path = "/home/jab08/Dropbox/medicaldata/data_provided/110329103820NH.stl";
    const char* path = "/home/jab08/Dropbox/testData/ioannis_1.obj";
    Assimp::Importer importer;
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
    const aiScene* scene = importer.ReadFile(path,
            aiProcess_RemoveComponent       |
            aiProcess_JoinIdenticalVertices |
            aiProcess_Triangulate           |
            aiProcess_FindDegenerates       |
            aiProcess_SortByPType);
    if(!scene) {
        std::cout << "Something went wrong" << std::endl;
        return 1;
    }
    std::cout << "This scene has " << scene->mNumMeshes << " meshes on it." << std::endl;
    std::cout << "This scene has " << scene->mNumTextures << " textures on it." << std::endl;
    std::cout << "This scene has " << scene->mNumMaterials << " materials on it." << std::endl;
    for(int i = 0; i < scene->mNumMeshes; i++)
        mesh_checker(scene->mMeshes[i]);
    for(int i = 0; i < scene->mNumMaterials; i++)
        material_checker(scene->mMaterials[i]);
    return 0;
}

