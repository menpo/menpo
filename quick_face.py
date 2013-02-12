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

