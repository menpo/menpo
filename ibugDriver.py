from ibugMM.mesh.face     import Face
from ibugMM.mesh.cppmesh import CppMesh
from ibugMM.importer.model import OBJImporter
import numpy as np

print 'Imports done.'
objPath = '/home/jab08/testData/ioannis_001/exports/ioannis_001_022.obj'
oimporter = OBJImporter(objPath)
print 'Importer ready'
oFace = oimporter.generateFace()
print 'Face generated'
#L_c, A, grad_u, div_grad_u = oFace.geodesic()
print 'All calcuated'
#cI = oFace.coordsIndex[np.where(oFace.coordsIndex==0)[0]]
#c = oFace.coords[np.unique(cI)]
#cI[cI==32179] = 4
#cI[cI==35717] = 5
#cI[cI==32839] = 6
#print cI
#print c

#testMesh = Face(coords=np.zeros_like(c),coordsIndex=cI)
#oFace.laplacian_operator()
#sparse = oFace.construct_sparse_vertex_matrix()




