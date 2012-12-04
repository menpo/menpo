from ibugMM.mesh import cppmesh
import numpy as np

coords = np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
coordsIndex = np.array([[0,1,2],[3,2,1]],dtype=np.uint32)
myMesh = cppmesh.CppMesh(coords, coordsIndex)

