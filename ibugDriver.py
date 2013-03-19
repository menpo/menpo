from pybug.mesh.face     import Face
from pybug.mesh.cppmesh import CppMesh
from pybug.importer.model import ModelImporterFactory
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg 
import matplotlib.pyplot as plt

def hist(A):
  h, bins = np.histogram(A, bins = 50)
  width  = 0.7*(bins[1] - bins[0])
  center = (bins[:-1]+bins[1:])/2
  plt.bar(center, h, align='center', width = width)
  plt.show()

print 'Imports done.'

## ioannis face
objPath = '/home/jab08/testData/ioannis.obj'
oimporter = ModelImporterFactory(objPath)
print 'Importer ready'
testMesh = oimporter.generateFace()
print 'Face generated'
# nose, l_eye, r_eye
nose  = [10476]
l_eye = [40526]
r_eye = [40615]
mouth = [41366, 28560, 36719, 17657, 13955, 26988, 6327, 8229]
testMesh.calculate_geodesics(mouth)


#L_c_mag = np.dot(L_c.T, L_c)
#print 'dotted the L_c'
#b  = (L_c.T).dot(div_X)
#print 'dotted the L_c and the div_X'
#phi2 = linalg.spsolve(L_c_mag, b)
#print 'solved!'

#cI = oFace.coordsIndex[np.where(oFace.coordsIndex==0)[0]]
#c = oFace.coords[np.unique(cI)]
#cI[cI==32179] = 4
#cI[cI==35717] = 5
#cI[cI==32839] = 6

# unit cube 
#c  = np.array([[ 0., 0., 0.],
#               [ 1., 0., 0.],
#               [ 1., 1., 0.],
#               [ 0., 1., 0.],
#               [ 0., 0., 1.],
#               [ 1., 0., 1.],
#               [ 1., 1., 1.],
#               [ 0., 1., 1.]])
#
#cI = np.array([[0, 3, 1],
#               [1, 3, 2],
#               [1, 2, 5],
#               [6, 5, 2],
#               [2, 3, 6],
#               [6, 3, 7],
#               [3, 0, 4],
#               [3, 4, 7],
#               [0, 1, 4],
#               [5, 4, 1],
#               [4, 5, 6],
#               [6, 7, 4]], dtype=np.uint32)
#
#c  = np.array([[ 0., 0., 0.],
#               [ 1., 0., 0.],
#               [ 0., 1., 0.],
#               [-2., 0., 0.],
#               [ 0.,-1., 0.]])
#cI = np.array([[0, 4, 1],
#               [3, 0, 2],
#               [1, 2, 0],
#               [0, 3, 4]], dtype=np.uint32)
#
#print cI
#print c
#testMesh = Face(coords=c,coordsIndex=cI)


#bunnyImp = ModelImporterFactory('~/Dropbox/meshes/elephant-50kv.off')
#testMesh = bunnyImp.generateFace()

#sphereImp = ModelImporterFactory('~/sphere-160k.obj')
#testMesh  = sphereImp.generateFace()
#
#u_0 = np.zeros(testMesh.n_coords)
#u_0[0] = 1
