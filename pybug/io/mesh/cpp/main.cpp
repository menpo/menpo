#include <iostream>
#include "assimpwrapper.h"

int main()
{
    //std::string path = "/home/jab08/Dropbox/medicaldata/data_provided/110329103820NH.stl";
    std::string path = "/home/jab08/Dropbox/testData/ioannis_1.obj";
    AssimpWrapper* assimpwrapper = new AssimpWrapper(path);
    unsigned int n_meshes = assimpwrapper->n_meshes();
    std::cout << "This file contains " << n_meshes << " meshes." << std::endl;
    for(int i = 0; i < n_meshes; i++) {
        std::cout << "----- MESH NO. " << i << " -----" << std::endl;
        unsigned int n_points = assimpwrapper->n_points(i);
        unsigned int n_tcoord_sets = assimpwrapper->n_tcoord_sets(i);
        unsigned int n_tris = assimpwrapper->n_tris(i);
        std::string texture_path = assimpwrapper->texture_path(i);
        std::cout << "n_points:      " << n_points << std::endl;
        std::cout << "n_tris:        " << n_tris << std::endl;
        std::cout << "n_tcoord_sets: " << n_tcoord_sets << std::endl;
        std::cout << "texture_path:  " << texture_path << std::endl;
    }
    delete assimpwrapper;
    return 0;
}

