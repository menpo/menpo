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
	                    0.,1.,0.,
	                    0.,1.,0.,
	                    0.,1.,0.,
                      1.,1.,0.};
  unsigned coordsIndex [] = {0,1,2,
                             2,3,0,
                             4,5,0,
                             0,6,4,
                             6,0,3,
                             1,0,5};
  unsigned n_coords = 7;
  unsigned n_triangles = 6;
  std::cout << "before construct" << std::endl;
  Mesh* mesh = new Mesh(coords, n_coords,coordsIndex, n_triangles);
  std::cout << "can construct!" << std::endl;
  //std::vector<Triangle*>::iterator it;
  //for(it = mesh->triangles.begin(); it != mesh->triangles.end(); it++) {
	//std::cout << (*it)->area() << std::endl;
	//std::cout << (*it)->normal() << std::endl;
  //}
  
  mesh->calculateLaplacianOperator();
  //mesh->verifyAttachements();
  delete mesh;



  return 0;
}
