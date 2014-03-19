#include <iostream>
#include "assimpwrapper.h"

int main(int argc, char** argv)
{
    if(argc < 2){
        std::cout << "Please provide a path to a file as an argument to this";
        std::cout << " script." << std::endl;
        return 1;
    }
    std::cout << argv[1] << std::endl;
    AssimpImporter importer(argv[1]);
    AssimpScene* scene = importer.get_scene();
    std::cout << "This file contains " << scene->n_meshes() << " meshes." << std::endl;
    std::cout << "The scenes texture path is " << scene->texture_path() << std::endl;
    AssimpMesh* mesh;
    for(unsigned int i = 0; i < scene->n_meshes(); i++){
        mesh  = scene->meshes[i];
        std::cout << "--- MESH " << i << " ---" << std::endl;
        std::cout << "  is a trimesh?: " << mesh->is_trimesh() << std::endl;
        std::cout << "  n_points:      " << mesh->n_points() << std::endl;
        std::cout << "  n_faces:       " << mesh->n_faces() << std::endl;
        std::cout << "  n_tcoord_sets: " << mesh->n_tcoord_sets() << std::endl;
    }
    return 0;
}

