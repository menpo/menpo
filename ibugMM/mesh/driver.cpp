#include <iostream> 
#include <vector>
#include "mesh.h"
#include "triangle.h"
#include "vec3.h"


int main()
{
  double coords [] = {0.,0.,0.,
					  1.,0.,0.,
	                  0.,1.,0.,
                      1.,1.,0.};
  unsigned coordsIndex [] = {0,1,2,
                             3,2,1};
  unsigned n_coords = 4;
  unsigned n_triangles = 2;
  Mesh* mesh = new Mesh(coords, n_coords,coordsIndex, n_triangles);
  std::vector<Triangle*>::iterator it;
  for(it = mesh->triangles.begin(); it != mesh->triangles.end(); it++) {
	std::cout << (*it)->area() << std::endl;
	std::cout << (*it)->normal() << std::endl;
  }
  delete mesh;



  return 0;
}
