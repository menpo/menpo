from ibugMM.importer.model import ModelImporterFactory
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)

## ioannis face
ioannis_path_2 = '/home/jab08/Dropbox/testData/ioannis_2.obj'
importer = ModelImporterFactory(ioannis_path_2)
ioannis_2 = importer.generateFace()
ioannis_2.landmarks['nose']  = [10476]
ioannis_2.landmarks['l_eye'] = [40615]
ioannis_2.landmarks['r_eye'] = [40526]
ioannis_2.landmarks['mouth'] = [41366, 28560, 36719, 17657, 13955, 26988, 6327, 8229]

# store the ground truth equivielent positions
ioannis_2_gt = 32294 

phi_n = ioannis_2.calculate_geodesics(ioannis_2.landmarks['nose'])['phi']

mask = phi_n > -100000000
mask[53058] = False
self = ioannis_2
original_vertex_index = np.arange(self.n_coords)
kept_vertices = original_vertex_index[mask]
bool_coord_index_mask = \
  np.in1d(self.coordsIndex, kept_vertices).reshape(self.coordsIndex.shape)
kept_triangles_orig_index = self.coordsIndex[np.all(bool_coord_index_mask, axis = 1)]
# some additional vertices will have to be removed as they no longer 
# form part of a triangle
kept_vertices_orig_index = np.unique(kept_triangles_orig_index)
number_conversion = np.zeros_like(original_vertex_index)
new_vertex_numbering = np.arange(kept_vertices_orig_index.shape[0])
number_conversion[kept_vertices_orig_index] = new_vertex_numbering
new_coord_index = number_conversion[kept_triangles_orig_index].astype(np.uint32)
new_coords = self.coords[kept_vertices_orig_index]
