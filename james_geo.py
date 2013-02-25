import numpy as np
from scipy.spatial import distance
from ibugMM.importer.model import import_face
import matplotlib.pyplot as plt
from mayavi import mlab
from ibugMM.mesh.face import Face
from scipy.stats import norm
plt.interactive(True)

def gaussian_weighting(x, std_dev=40):
  return std_dev*np.sqrt(2*np.pi) * norm.pdf(x, scale=std_dev)


james_m_path = '/home/jab08/Dropbox/testData/james_m.obj'
james_n_path = '/home/jab08/Dropbox/testData/james_n.obj'
james_h_path = '/home/jab08/Dropbox/testData/james_h.obj'
print 'a'
james_m = import_face(james_m_path)
print 'b'
james_n = import_face(james_n_path)
print 'c'
james_h = import_face(james_h_path)

james_m.landmarks['mouth']  = [17517, 30064, 24052, 12240, 16405, 12242, 24055, 11702]
james_m.landmarks['nose']   = [32168]
james_m.landmarks['l_eb']   = [5611, 12326, 5718]
james_m.landmarks['bridge'] = [3842]
james_m.landmarks['r_eb']   = [8943, 22538, 17114]
james_m.landmarks['l_eye']  = [21309, 7891, 14204, 21477, 21468, 21459, 14197, 2679]
james_m.landmarks['r_eye']  = [2843, 8025, 14586, 1476, 6160, 11072, 14580, 2847]

james_h.landmarks['mouth']  = [11741, 24637, 4120, 4138, 13895, 3565, 8594, 30832]
james_h.landmarks['nose']   = [31160]
james_h.landmarks['l_eb']   = [40, 291, 21685]
james_h.landmarks['bridge'] = [3809]
james_h.landmarks['r_eb']   = [1473, 12448, 3840]
james_h.landmarks['l_eye']  = [615, 16810, 10588, 21599, 10582, 21600, 21593, 16809]
james_h.landmarks['r_eye']  = [14639, 11017, 27631, 338, 2851, 2847, 22799, 22788]

james_n.landmarks['mouth']  = [1757, 11782, 590, 16397, 16433, 16395, 8809, 11787]
james_n.landmarks['nose']   = [91]
james_n.landmarks['l_eb']   = [21120, 12464, 10827]
james_n.landmarks['bridge'] = [3949]
james_n.landmarks['r_eb']   = [9085, 1532, 9121]
james_n.landmarks['l_eye']  = [21110, 8044, 14299, 10796, 5801, 21228, 14289, 14282]
james_n.landmarks['r_eye']  = [14645, 14667, 11162, 11165, 22191, 22186, 22180, 14656]


james_mid_h = james_h.new_face_masked_from_lm('nose', distance=120, method='exact')
james_mid_n = james_n.new_face_masked_from_lm('nose', distance=120, method='exact')
james_mid_m = james_m.new_face_masked_from_lm('nose', distance=120, method='exact')


def gen_phi_coords(face, method='heat'):
  phi_vectors = np.zeros([face.n_vertices, face.n_landmarks])
  for i,k in enumerate(face.landmarks):
    print i
    print k
    phi_vectors[:,i] = face.geodesics_about_lm(k, method)['phi']
  # ph_coords s.t. phi_coords[0] is the geodesic vector for the first ordinate
  return phi_vectors

def gen_phi_coords_per_lm_bar_mouth(face, method='heat'):
  phi_vectors = np.zeros([face.n_vertices, face.n_landmark_vertices - 7])
  i = 0
  for k in face.landmarks.iterkeys():
    if k != 'mouth':
      for v in face.landmarks[k]:
        phi_vectors[:,i] = face.geodesics_about_vertices([v], method)['phi']
        i += 1
    else:
      phi_vectors[:,i] = face.geodesics_about_lm(k, method)['phi']
      i += 1
  print i
  print face.n_landmark_vertices - 7
  return phi_vectors

def gen_phi_coords_per_lm_vertex(face, method='heat'):
  phi_vectors = np.zeros([face.n_vertices, face.n_landmark_vertices])
  for i,v in enumerate(face.all_landmarks):
      phi_vectors[:,i] = face.geodesics_about_vertices([v], method)['phi']
  # ph_coords s.t. phi_coords[0] is the geodesic vector for the first ordinate
  return phi_vectors

def geodesic_mapping(face_1, face_2, method='exact'):
  phi_1 = gen_phi_coords_per_lm_vertex(face_1, method)
  phi_2 = gen_phi_coords_per_lm_vertex(face_2, method)

  ## find all distances between the two phi's
  distances = distance.cdist(phi_1, phi_2)
  #
  ## interpolate the phi_vectors to the centre of each triangle
  #phi_1_tri = np.mean(phi_1[ioannis_1.tri_index], axis=1)
  #phi_2_tri = np.mean(phi_2[ioannis_2.tri_index], axis=1)
  #
  ## and also find the coordinates at the centre of each triangle
  #tri_coords = np.mean(ioannis_2.coords[ioannis_2.tri_index], axis=1)
  #
  #distances_1_v_to_2_tri = distance.cdist(phi_1, phi_2_tri)
  #
  ## now find the minimum mapping indicies, both for 2 onto 1 and 1 onto 2
  mins_2_to_1 = np.argmin(distances, axis=1)
  mins_1_to_2 = np.argmin(distances, axis=0)
  #
  ## and find the minimum mapping of 2 onto 1 from tri to vert
  #mins_2_tri_to_1 = np.argmin(distances_1_v_to_2_tri, axis=1)
  #
  #using_tris = np.where(mins_2_tri_to_1 < mins_2_to_1)[0]
  #new_c = ioannis_2.coords[mins_2_to_1]
  #new_c[using_tris] = tri_coords[mins_2_tri_to_1[using_tris]]
  ###
  #### and count how many times each vertex is mapped from
  ###count_1_to_2 = np.bincount(mins_1_to_2)
  ###count_2_to_1 = np.bincount(mins_2_to_1)
  ###
  #### calculate the mapping and build a new face using it
  face_2_on_1 = Face(face_2.coords[mins_2_to_1], face_1.tri_index,
                   texture=face_1.texture, texture_coords=face_1.texture_coords,
                   texture_tri_index=face_1.texture_tri_index)
  #
  #face2_on1_inc_tris = Face(new_c, face_1.tri_index,
  #                          texture=face_1.texture, texture_coords=face_1.texture_coords,
  #                          texture_tri_index=face_1.texture_tri_index)
  #
  face_1_on_2 = Face(face_1.coords[mins_1_to_2], face_2.tri_index,
                   texture=face_2.texture, texture_coords=face_2.texture_coords,
                   texture_tri_index=face_2.texture_tri_index)
  return face_1_on_2, face_2_on_1, distances

face_1 = james_mid_h
face_2 = james_mid_n
method='exact'
## ioannis face
ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1.obj'
o_ioannis_1 = import_face(ioannis_path_1)
ioannis_path_2 = '/home/jab08/Dropbox/testData/ioannis_2.obj'
o_ioannis_2 = import_face(ioannis_path_2)

## note, l_eye is THEIR l_eye (right as we look at it)
o_ioannis_1.landmarks['nose']  = [17537]
o_ioannis_1.landmarks['l_eye'] = [5695]
o_ioannis_1.landmarks['r_eye'] = [5495]
o_ioannis_1.landmarks['mouth'] = [15461, 18940, 12249, 17473, 36642, 2889, 11560, 10125]
o_ioannis_1.landmarks['cheek_freckle'] = [752]

o_ioannis_2.landmarks['nose']  = [10476]
o_ioannis_2.landmarks['l_eye'] = [40615]
o_ioannis_2.landmarks['r_eye'] = [40526]
o_ioannis_2.landmarks['mouth'] = [41366, 28560, 36719, 17657, 13955, 26988, 6327, 8229]
o_ioannis_2.landmarks['cheek_freckle'] = [32294]

face_1 = o_ioannis_1.new_face_masked_from_lm('nose', method='exact')
face_2 = o_ioannis_2.new_face_masked_from_lm('nose', method='exact')

phi_1 = gen_phi_coords_per_lm_vertex(face_1, method=method)
phi_2 = gen_phi_coords_per_lm_vertex(face_2, method=method)

# weighting_1 is how influential each value of each phi vector is considered by the first face
weightings_1 = gaussian_weighting(phi_1)
weightings_2 = gaussian_weighting(phi_2)

differences = phi_1[:,np.newaxis,:] - phi_2
# multiply by the weightings (store straight back in differences)
np.multiply(weightings_1[:,np.newaxis,:], differences, out=differences)
# square differences, and sum them
np.square(differences, out=differences)
np.sum(differences, axis=2)
diffs = np.sum(differences, axis=2)
min_2_to_1 = np.argmin(diffs,axis=1)
face_2_on_1 = Face(face_2.coords[min_2_to_1], face_1.tri_index, 
    texture=face_1.texture, texture_coords=face_1.texture_coords, 
    texture_tri_index=face_1.texture_tri_index)
