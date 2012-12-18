from ibugMM.mesh.face     import Face
from ibugMM.mesh.cppmesh import CppMesh
from ibugMM.importer.model import OBJImporter
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def hist(A):
  h, bins = np.histogram(A, bins = 50)
  width  = 0.7*(bins[1] - bins[0])
  center = (bins[:-1]+bins[1:])/2
  plt.bar(center, h, align='center', width = width)
  plt.show()


#print 'Imports done.'
#objPath = '/home/jab08/testData/ioannis_001/exports/ioannis_001_022.obj'
#oimporter = OBJImporter(objPath)
#print 'Importer ready'
#oFace = oimporter.generateFace()
#print 'Face generated'
#u_0 = np.zeros(oFace.n_coords)
#u_0[3000] = 1
#u_t = np.zeros(oFace.n_coords)
#L_c, A = oFace.laplacian_operator()
#print 'Calculated Laplacian'
#A = csc_matrix(A)
#L_c = csc_matrix(L_c)
#t = 0.5
#u_t = spsolve(A - (L_c), u_0)
#print 'Solved for u_t'
#grad_u_t = oFace.gradient(u_t)
#print 'Solved for grad_u_t'
#grad_u_t_mag  = np.sqrt(np.sum(grad_u_t**2, axis=1))
#X = -1.0*grad_u_t/(grad_u_t_mag).reshape([-1,1])
## some of the vectors may have been zero length - ensure this
## is true before and after
#X[grad_u_t_mag == 0] = 0
#print 'Generated X'
#div_X = oFace.divergence(X)
#print 'Generated div_X'
#phi = spsolve(L_c, div_X)
#print 'Generated distances'

#cI = oFace.coordsIndex[np.where(oFace.coordsIndex==0)[0]]
#c = oFace.coords[np.unique(cI)]
#cI[cI==32179] = 4
#cI[cI==35717] = 5
#cI[cI==32839] = 6

c  = np.array([[ 0., 0., 0.],
               [ 1., 0., 0.],
               [ 0., 1., 0.],
               [-1., 0., 0.],
               [ 0.,-1., 0.]])

cI = np.array([[0, 4, 1],
               [3, 0, 2],
               [1, 2, 0],
               [0, 3, 4]], dtype=np.uint32)
print cI
print c

testMesh = Face(coords=c,coordsIndex=cI)
u_0 = np.zeros(testMesh.n_coords)
u_0[0] = 1
u_t = np.zeros(testMesh.n_coords)
L_c, A = testMesh.laplacian_operator()
print 'Calculated Laplacian'
A = csc_matrix(A)
L_c = csc_matrix(L_c)
t = 1.4371
u_t = spsolve(A - (L_c), u_0)
print 'Solved for u_t'
grad_u_t = testMesh.gradient(u_t)
print 'Solved for grad_u_t'
grad_u_t_mag  = np.sqrt(np.sum(grad_u_t**2, axis=1))
X = -1.0*grad_u_t/(grad_u_t_mag).reshape([-1,1])
# some of the vectors may have been zero length - ensure this
# is true before and after
X[grad_u_t_mag == 0] = 0
print 'Generated X'
div_X = testMesh.divergence(X)
print 'Generated div_X'
phi = spsolve(L_c, div_X)
print 'Generated distances'




