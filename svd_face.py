from ibugMM.importer.model import import_face
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy
from mayavi import mlab
plt.interactive(True)
from tvtk.api import tvtk
from tvtk.tools import ivtk

ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1_a.obj'
face = import_face(ioannis_path_1)
face.landmarks['nose']  = [13225]
#face.view_geodesic_contours_about_lm('nose')
L_c = face.laplacian()
svd = scipy.linalg.svd(L_c.toarray())

#geo = face.geodesics_about_vertices(face.landmarks['nose'])
#u_t = geo['u_t']
