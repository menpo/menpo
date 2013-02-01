from ibugMM.importer.model import ModelImporterFactory
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)

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
  phi_m = face.calculate_geodesics(landmarks['mouth'])
  phi_n = face.calculate_geodesics(landmarks['nose'])
  phi_l = face.calculate_geodesics(landmarks['l_eye'])
  phi_r = face.calculate_geodesics(landmarks['r_eye'])
  # ph_coords s.t. phi_coords[0] is the geodesic vector for the first ordinate
  return np.vstack((phi_m, phi_n, phi_l, phi_r)).T

## ioannis face
ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1.obj'
importer = ModelImporterFactory(ioannis_path_1)
ioannis_1 = importer.generateFace()
ioannis_path_2 = '/home/jab08/Dropbox/testData/ioannis_2.obj'
importer = ModelImporterFactory(ioannis_path_2)
ioannis_2 = importer.generateFace()
# nose, l_eye, r_eye
ioannis_2.landmarks['nose']  = [10476]
ioannis_2.landmarks['l_eye'] = [40526]
ioannis_2.landmarks['r_eye'] = [40615]
ioannis_2.landmarks['mouth'] = [41366, 28560, 36719, 17657, 13955, 26988, 6327, 8229]
landmarks_1 = {}
landmarks_1['nose']  = [10476]
landmarks_1['l_eye'] = [40526]
landmarks_1['r_eye'] = [40615]
landmarks_1['mouth'] = [41366, 28560, 36719, 17657, 13955, 26988, 6327, 8229]
#phi_coords_2 = gen_phi_coords(ioannis_2, landmarks_2)
#for i in range(1,5):
#  print_geodesic_patterns(phi_coords_2, i)




