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
e_02 = c_tri[:,2,:] - c_tri[:,0,:]
e_12 = c_tri[:,2,:] - c_tri[:,1,:]

mod_e_01 = np.sqrt(np.sum(e_01*e_01, axis=1))
mod_e_02 = np.sqrt(np.sum(e_02*e_02, axis=1))
mod_e_12 = np.sqrt(np.sum(e_12*e_12, axis=1))

normed_e_01 = e_01/(mod_e_01[...,np.newaxis])
normed_e_02 = e_02/(mod_e_02[...,np.newaxis])
normed_e_12 = e_12/(mod_e_12[...,np.newaxis])

cos_0 = np.sum(normed_e_01*normed_e_02, axis=1)
cos_1 = -1.0 * np.sum(normed_e_12*normed_e_02, axis=1)
cos_2 = np.sum(normed_e_12*normed_e_02, axis=1)

cross_0 = np.cross(normed_e_01, normed_e_02)
cross_1 = -1.0 * np.cross(normed_e_12, normed_e_02)
cross_2 = np.cross(normed_e_12, normed_e_02)

sin_0 = np.sum(cross_0*cross_0, axis=1)
sin_1 = np.sum(cross_1*cross_1, axis=1)
sin_2 = np.sum(cross_2*cross_2, axis=1)

cot_0 = cos_0/(np.sqrt(1-(cos_0*cos_0)))
cot_1 = cos_1/(np.sqrt(1-(cos_1*cos_1)))
cot_2 = cos_2/(np.sqrt(1-(cos_2*cos_2)))

cot_0_a = cos_0/sin_0
cot_1_a = cos_1/sin_1
cot_2_a = cos_2/sin_2

Xdot_01 = np.sum(X*e_01, axis=1)
Xdot_02 = np.sum(X*e_02, axis=1)
Xdot_12 = np.sum(X*e_12, axis=1)


divX_at_0 =       cot_1*Xdot_02 + cot_2*Xdot_01
divX_at_1 =  -1.0*cot_2*Xdot_02 + cot_0*Xdot_12
divX_at_2 = -1.0*(cot_0*Xdot_12 + cot_1*Xdot_02)
