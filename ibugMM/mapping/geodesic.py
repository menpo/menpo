import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mayavi import mlab
from ibugMM.mesh.face import Face
from scipy.stats import norm
plt.interactive(True)

class GeodesicMapping(object):

  def __init__(self, face_1, face_2, method='exact', signiture='vertices', std_dev = None):
    self.face_1 = face_1
    self.face_2 = face_2
    self.method = method
    self.signiture = signiture
    self.std_dev = std_dev
    self.calculate_signitures()
    self.calculate_mapping()
    self.calculate_mapped_faces()

  def calculate_signitures(self):
    if self.signiture == 'vertices':
      print 'calculating geodesic signiture for all landmark vertices'
      self.gsig_1 = geodesic_signiture_per_lm_vertex(self.face_1, self.method)
      self.gsig_2 = geodesic_signiture_per_lm_vertex(self.face_2, self.method)
    elif self.signiture == 'groups':
      print 'calculating geodesic signiture for each group of landmark vertices'
      self.gsig_1 = geodesic_signiture_per_lm_group(self.face_1, self.method)
      self.gsig_2 = geodesic_signiture_per_lm_group(self.face_2, self.method)
    elif self.signiture == 'vertices_group_mouth':
      print 'calculating geodesic signiture for each landmark vertex except \
          the mouth, which is treated as a group'
      self.gsig_1 = geodesic_signiture_per_lm_group_bar_mouth(self.face_1, self.method)
      self.gsig_2 = geodesic_signiture_per_lm_group_bar_mouth(self.face_2, self.method)
    else:
      raise Exception("I don't understand signiture " + `self.signiture`)

  def calculate_mapping(self):
    if self.std_dev == None:
      print 'no std dev supplied - creating a linear mapping'
      self.min_2_to_1, self.min_1_to_2 = linear_geodesic_mapping(self.gsig_1, self.gsig_2)
    else:
      print 'std dev supplied - creating a weighted mapping with std_dev ' + `self.std_dev`
      self.min_2_to_1, self.min_1_to_2 = weighted_geodesic_mapping(self.gsig_1, self.gsig_2, std_dev=std_dev)

  def calculate_mapped_faces(self):
    self.face_2_on_1 = new_face_from_mapping(self.face_1, self.face_2,
        self.min_2_to_1)
    self.face_1_on_2 = new_face_from_mapping(self.face_2, self.face_1,
        self.min_1_to_2)
    self.face_2_on_1_t_of_2 = new_face_from_mapping(self.face_1, self.face_2,
        self.min_2_to_1, transfer_texture=True)
    self.face_1_on_2_t_of_1 = new_face_from_mapping(self.face_2, self.face_1,
        self.min_1_to_2, transfer_texture=True)


def linear_geodesic_mapping(phi_1, phi_2):
  distances = distance.cdist(phi_1, phi_2)
  min_2_to_1 = np.argmin(distances, axis=1)
  min_1_to_2 = np.argmin(distances, axis=0)
  return min_2_to_1, min_1_to_2

def weighted_geodesic_mapping(phi_1, phi_2, std_dev=40):
  # weighting_1 is how influential each value of each phi vector is considered by the first face
  weightings_1 = gaussian_weighting(phi_1, std_dev)
  weightings_2 = gaussian_weighting(phi_2, std_dev)
  differences = phi_1[:,np.newaxis,:] - phi_2
  # multiply by the first face weightings (store straight back in differences)
  np.multiply(weightings_1[:,np.newaxis,:], differences, out=differences)
  np.square(differences, out=differences)
  diffs = np.sum(differences, axis=2)
  min_2_to_1 = np.argmin(diffs,axis=1)
  # now repeat but for the second weighting
  differences = np.subtract(phi_1[:,np.newaxis,:], phi_2, out=differences)
  np.multiply(weightings_2[np.newaxis,:,:], differences, out=differences)
  diffs = np.sum(differences, axis=2)
  min_1_to_2 = np.argmin(diffs,axis=0)
  return min_2_to_1, min_1_to_2

def gaussian_weighting(x, std_dev):
  return std_dev*np.sqrt(2*np.pi) * norm.pdf(x, scale=std_dev)

def geodesic_signiture_per_lm_group(face, method='exact'):
  phi_vectors = np.zeros([face.n_vertices, face.n_landmarks])
  for i,k in enumerate(face.landmarks):
    print i
    print k
    phi_vectors[:,i] = face.geodesics_about_lm(k, method)['phi']
  # ph_coords s.t. geodesic_signiture[0] is the geodesic vector for the first ordinate
  return phi_vectors

def geodesic_signiture_per_lm_group_bar_mouth(face, method='exact'):
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

def geodesic_signiture_per_lm_vertex(face, method='exact'):
  phi_vectors = np.zeros([face.n_vertices, face.n_landmark_vertices])
  for i,v in enumerate(face.all_landmarks):
      phi_vectors[:,i] = face.geodesics_about_vertices([v], method)['phi']
  # ph_coords s.t. geodesic_signiture[0] is the geodesic vector for the first ordinate
  return phi_vectors

def new_face_from_mapping(face_a, face_b, min_b_to_a, transfer_texture=False):
  if not transfer_texture:
    return Face(face_b.coords[min_b_to_a], face_a.tri_index,
        texture=face_a.texture, texture_coords=face_a.texture_coords,
        texture_tri_index=face_a.texture_tri_index)
  else:
    return Face(face_b.coords[min_b_to_a], face_a.tri_index,
        texture=face_b.texture, texture_coords=face_b.texture_coords_per_vertex[min_b_to_a])


def unique_closest_n(phi, n):
  np.argsort(phi, axis=1)
  ranking = np.argsort(phi, axis=1)
  top_n = ranking[:,:n]
  sort_order = np.lexsort(top_n.T)
  top_n_ordered = top_n[sort_order]
  top_n_diff = np.diff(top_n_ordered, axis=0)
  ui = np.ones(len(top_n), 'bool')
  ui[1:] = (top_n_diff != 0).any(axis=1)
  unique_top_n = top_n_ordered[ui]
  return unique_top_n

#def geodesic_mapping(face_1, face_2, method='exact'):
#  phi_1 = gen_phi_coords_per_lm_vertex(face_1, method)
#  phi_2 = gen_phi_coords_per_lm_vertex(face_2, method)
#
#  ## find all distances between the two phi's
#  distances = distance.cdist(phi_1, phi_2)
#  #
#  ## interpolate the phi_vectors to the centre of each triangle
#  #phi_1_tri = np.mean(phi_1[ioannis_1.tri_index], axis=1)
#  #phi_2_tri = np.mean(phi_2[ioannis_2.tri_index], axis=1)
#  #
#  ## and also find the coordinates at the centre of each triangle
#  #tri_coords = np.mean(ioannis_2.coords[ioannis_2.tri_index], axis=1)
#  #
#  #distances_1_v_to_2_tri = distance.cdist(phi_1, phi_2_tri)
#  #
#  ## now find the minimum mapping indicies, both for 2 onto 1 and 1 onto 2
#  mins_2_to_1 = np.argmin(distances, axis=1)
#  mins_1_to_2 = np.argmin(distances, axis=0)
#  #
#  ## and find the minimum mapping of 2 onto 1 from tri to vert
#  #mins_2_tri_to_1 = np.argmin(distances_1_v_to_2_tri, axis=1)
#  #
#  #using_tris = np.where(mins_2_tri_to_1 < mins_2_to_1)[0]
#  #new_c = ioannis_2.coords[mins_2_to_1]
#  #new_c[using_tris] = tri_coords[mins_2_tri_to_1[using_tris]]
#  ###
#  #### and count how many times each vertex is mapped from
#  ###count_1_to_2 = np.bincount(mins_1_to_2)
#  ###count_2_to_1 = np.bincount(mins_2_to_1)
#  ###
#  #### calculate the mapping and build a new face using it
#  face_2_on_1 = Face(face_2.coords[mins_2_to_1], face_1.tri_index,
#                   texture=face_1.texture, texture_coords=face_1.texture_coords,
#                   texture_tri_index=face_1.texture_tri_index)
#  #
#  #face2_on1_inc_tris = Face(new_c, face_1.tri_index,
#  #                          texture=face_1.texture, texture_coords=face_1.texture_coords,
#  #                          texture_tri_index=face_1.texture_tri_index)
#  #
#  face_1_on_2 = Face(face_1.coords[mins_1_to_2], face_2.tri_index,
#                   texture=face_2.texture, texture_coords=face_2.texture_coords,
#                   texture_tri_index=face_2.texture_tri_index)
#  return face_1_on_2, face_2_on_1, distances
