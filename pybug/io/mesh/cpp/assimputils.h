#pragma once

#include <string>
const std::string NO_TEXTURE_PATH = "NO_TEXTURE_PATH";

class aiMesh;
class aiMaterial;

unsigned int tcoords_mask(aiMesh* mesh, bool* has_tcoords);

void mesh_flags(aiMesh* mesh, bool& has_points, bool& has_lines,
                bool& has_triangles, bool& has_polygons);

void read_points(aiMesh* mesh, double* points);

void read_trilist(aiMesh* mesh, unsigned int* trilist);

void read_tcoords(aiMesh* mesh, int pindex, double* tcoords);

void mesh_checker(aiMesh* mesh);

void mesh_import(aiMesh* mesh, double* points, unsigned int* trilist,
                 double** tcoords);

std::string diffuse_texture_path_on_material(aiMaterial* mat);

