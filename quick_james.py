from ibugMM.importer.model import import_face
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mayavi import mlab
plt.interactive(True)
from tvtk.api import tvtk
from tvtk.tools import ivtk

james_m_path = '/home/jab08/Dropbox/testData/james_m.obj'
james_n_path = '/home/jab08/Dropbox/testData/james_n.obj'
james_h_path = '/home/jab08/Dropbox/testData/james_h.obj'
james_m = import_face(james_m_path)
james_n = import_face(james_n_path)
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

