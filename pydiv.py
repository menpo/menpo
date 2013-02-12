from ibugMM.importer.model import ModelImporterFactory
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)
from tvtk.api import tvtk
from tvtk.tools import ivtk

ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1.obj'
importer = ModelImporterFactory(ioannis_path_1)
face = importer.generateFace()
face.landmarks['nose']  = [46731]
face.landmarks['l_eye'] = [5695]
face.landmarks['r_eye'] = [5495]
face.landmarks['mouth'] = [15461, 18940, 12249, 17473, 36642, 2889, 11560, 10125]
face.landmarks['cheek_freckle'] = [752]

X = face.geodesics_about_vertices(face.landmarks['nose'])['X']

c_tri = face.coords[face.tri_index]
e_01 = c_tri[:,1,:] - c_tri[:,0,:]
e_12 = c_tri[:,2,:] - c_tri[:,1,:]
e_20 = c_tri[:,0,:] - c_tri[:,2,:]

mag_e_01 = np.sqrt(np.sum(e_01 * e_01, axis=1))
mag_e_12 = np.sqrt(np.sum(e_12 * e_12, axis=1))
mag_e_20 = np.sqrt(np.sum(e_20 * e_20, axis=1))

unit_e_01 = e_01 / (mag_e_01[...,np.newaxis])
unit_e_12 = e_12 / (mag_e_12[...,np.newaxis])
unit_e_20 = e_20 / (mag_e_20[...,np.newaxis])

cos_0 = -1.0 * np.sum(unit_e_01 * unit_e_20, axis=1)
cos_1 = -1.0 * np.sum(unit_e_01 * unit_e_12, axis=1)
cos_2 = -1.0 * np.sum(unit_e_12 * unit_e_20, axis=1)

cross_0 = -1.0 * np.cross(unit_e_01, unit_e_20)
cross_1 = -1.0 * np.cross(unit_e_12, unit_e_01)
cross_2 = -1.0 * np.cross(unit_e_20, unit_e_12)

sin_0 = np.sum(cross_0 * cross_0, axis=1)
sin_1 = np.sum(cross_1 * cross_1, axis=1)
sin_2 = np.sum(cross_2 * cross_2, axis=1)

cot_0 = cos_0/sin_0
cot_1 = cos_1/sin_1
cot_2 = cos_2/sin_2

self = face

def divergence(X):
  X_dot_e_01 = np.sum(X*e_01, axis=1)
  X_dot_e_12 = np.sum(X*e_12, axis=1)
  X_dot_e_20 = np.sum(X*e_20, axis=1)
  div_X_tri = np.zeros_like(self.tri_index, dtype=np.float)  
  div_X_tri[:,0] = (cot_2 * X_dot_e_01) - (cot_1 * X_dot_e_20)
  div_X_tri[:,1] = (cot_0 * X_dot_e_12) - (cot_2 * X_dot_e_01)
  div_X_tri[:,2] = (cot_1 * X_dot_e_20) - (cot_0 * X_dot_e_12)
  return div_X_tri

div_X_tri = divergence(X)
