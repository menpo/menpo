import numpy as np
from ibugMM.mapping.geodesic import GeodesicMapping
import matplotlib.pyplot as plt
from mayavi import mlab
plt.interactive(True)
from ibugMM.importer.model import import_face

james_n_path = '/home/jab08/Dropbox/testData/james_n_no_mouth.obj'
james_h_path = '/home/jab08/Dropbox/testData/james_h_no_mouth.obj'
james_n = import_face(james_n_path)
james_h = import_face(james_h_path)

james_mid_h = james_h.new_face_masked_from_lms(['nose', 'mouth'], [120.0, 50], method='exact')
james_mid_n = james_n.new_face_masked_from_lms(['nose', 'mouth'], [120.0, 50], method='exact')

geodesic_mapper = GeodesicMapping(james_mid_h, james_mid_n)

