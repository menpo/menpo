from ibugMM.importer.model import import_face 
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)
from tvtk.api import tvtk
from tvtk.tools import ivtk

bunny = import_face('~/Downloads/bunny.obj')
bunny.landmarks = {}
bunny.landmarks['back'] = [784]
bunny.store_geodesics_for_all_landmarks()
bunny.view_last_geodesic_contours(periodicity=5)

#geo = face.geodesics_about_vertices(face.landmarks['nose'])
#u_t = geo['u_t']
