from ibugMM.importer.model import import_face 
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)
from tvtk.api import tvtk
from tvtk.tools import ivtk


def print_geodesic_patterns(phi_coords, i):
  distances = distance.pdist(phi_coords[:,0:i], 'cityblock')
  plt.figure()
  fig = plt.gcf()
  plt.hist(distances, bins=1000, histtype='stepfilled', range=(0,500))
  fig.savefig('full_' + `i` + '.pdf')
  plt.figure()
  fig = plt.gcf()
  small_distances = distances[np.where(distances < 5)]
  plt.hist(small_distances, bins=1000, histtype='stepfilled', range=(0,5))
  fig.savefig('near0_' + `i` + '.pdf')

def gen_phi_coords(face, landmarks):
  phi_m = face.geodesics_about_vertices(landmarks['mouth'])['phi']
  phi_n = face.geodesics_about_vertices(landmarks['nose'])['phi']
  phi_l = face.geodesics_about_vertices(landmarks['l_eye'])['phi']
  phi_r = face.geodesics_about_vertices(landmarks['r_eye'])['phi']
  # ph_coords s.t. phi_coords[0] is the geodesic vector for the first ordinate
  return np.vstack((phi_m, phi_n, phi_l, phi_r)).T

## ioannis face
ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1.obj'
o_ioannis_1 = import_face(ioannis_path_1)
ioannis_path_2 = '/home/jab08/Dropbox/testData/ioannis_2.obj'
o_ioannis_2 = import_face(ioannis_path_2)

## note, l_eye is THEIR l_eye (right as we look at it)
landmarks_1['nose']  = [46731]
landmarks_1['l_eye'] = [5695]
landmarks_1['r_eye'] = [5495]
landmarks_1['mouth'] = [15461, 18940, 12249, 17473, 36642, 2889, 11560, 10125]
landmarks_1['cheek_freckle'] = [752]
o_ioannis_1.landmarks = landmarks_1

landmarks_2['nose']  = [10476]
landmarks_2['l_eye'] = [40615]
landmarks_2['r_eye'] = [40526]
landmarks_2['mouth'] = [41366, 28560, 36719, 17657, 13955, 26988, 6327, 8229]
landmarks_2['cheek_freckle'] = [32294]
o_ioannis_2.landmarks = landmarks_2
ioannis_1  = o_ioannis_1.new_face_masked_from_lm('nose')
ioannis_2  = o_ioannis_2.new_face_masked_from_lm('nose')
#
## generate the geodesic vectors for face 1 and 2
phi_1 = gen_phi_coords(ioannis_1, ioannis_1.landmarks)
phi_2 = gen_phi_coords(ioannis_2, ioannis_2.landmarks)

# find all distances between the two phi's
distances = distance.cdist(phi_1, phi_2)

# interpolate the phi_vectors to the centre of each triangle
phi_1_tri = np.mean(phi_1[ioannis_1.coords_index], axis=1)
phi_2_tri = np.mean(phi_2[ioannis_2.coords_index], axis=1)

# and also find the coordinates at the centre of each triangle
tri_coords = np.mean(ioannis_2.coords[ioannis_2.coords_index], axis=1)

distances_1_v_to_2_tri = distance.cdist(phi_1, phi_2_tri)

# now find the minimum mapping indicies, both for 2 onto 1 and 1 onto 2
mins_2_to_1 = np.argmin(distances, axis=1)
mins_1_to_2 = np.argmin(distances, axis=0)


# and find the minimum mapping of 2 onto 1 from tri to vert
mins_2_tri_to_1 = np.argmin(distances_1_v_to_2_tri, axis=1)


using_tris = np.where(mins_2_tri_to_1 < mins_2_to_1)[0]
new_c = ioannis_2.coords[mins_2_to_1]
new_c[using_tris] = tri_coords[mins_2_tri_to_1[using_tris]]
##
### and count how many times each vertex is mapped from
##count_1_to_2 = np.bincount(mins_1_to_2)
##count_2_to_1 = np.bincount(mins_2_to_1)
##
### calculate the mapping and build a new face using it
face2_on1 = Face(ioannis_2.coords[mins_2_to_1], ioannis_1.coords_index,
                 texture=ioannis_1.texture, texture_coords=ioannis_1.texture_coords, 
                 texture_coords_index=ioannis_1.texture_coords_index)

face2_on1_inc_tris = Face(new_c, ioannis_1.coords_index,
                          texture=ioannis_1.texture, texture_coords=ioannis_1.texture_coords, 
                          texture_coords_index=ioannis_1.texture_coords_index)

face1_on2 = Face(ioannis_1.coords[mins_1_to_2], ioannis_2.coords_index,
                 texture=ioannis_2.texture, texture_coords=ioannis_2.texture_coords, 
                 texture_coords_index=ioannis_2.texture_coords_index)


#
#
#
def min_tri_and_min_vertex_location(i):
  ioannis_2.view_location_of_vertices(mins_2_to_1[i])
  mlab.figure()
  ioannis_2.view_location_of_triangles(mins_2_tri_to_1[i])


def l2_using_only(indexes):
  if len(indexes) > 1:
    return np.sum(delta_phi[:,indexes]**2,1)
  else:
    return delta_phi[:,indexes]**2

#phi_coords_2 = gen_phi_coords(ioannis_2, landmarks_2)
#for i in range(1,5):
#  print_geodesic_patterns(phi_coords_2, i)


#figure = mlab.gcf()
#mlab.clf()
#figure.scene.disable_render = True
#
#self = ioannis_2
#face_coords = mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
#                                   self.coords[:,2], self.coordsIndex, 
#                                   color=(0.5,0.5,0.5)) 
#figure.scene.disable_render = False
#face_points = face_coords.glyph.glyph_source.glyph_source.output.points.to_array()
#
#
#def picker_callback(picker):
#    """ Picker callback: this get called when on pick events.
#    """
#    if picker.actor in face_coords.actor.actors:
#        # Find which data point corresponds to the point picked:
#        # we have to account for the fact that each data point is
#        # represented by a glyph with several points
#        point_id = picker.point_id/face_points.shape[0]
#        # If the no points have been selected, we have '-1'
#        print point_id
#        if point_id != -1:
#            # Retrieve the coordinnates coorresponding to that data
#            # point
#            x, y, z = self.coords[point_id,0], self.coords[point_id,1], self.coords[point_id,2]
#            # Move the outline to the data point.
#            outline.bounds = (x-0.1, x+0.1,
#                              y-0.1, y+0.1,
#                              z-0.1, z+0.1)
#
#
#picker = figure.on_mouse_pick(picker_callback)
#picker.tolerance = 0.01
#
#
#mlab.show()
