from pybug.importer.model import import_face
from pybug.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)
from tvtk.api import tvtk
from tvtk.tools import ivtk

sphere = import_face('~/Dropbox/testData/sphere-160k.obj')
sphere.landmarks['x'] = [1]
sphere.store_geodesics_for_all_landmarks()

