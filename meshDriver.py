from pybug.mesh import face
import numpy as np

coords = np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
coordsIndex = np.array([[0,1,2],[3,2,1]],dtype=np.uint32)
myMesh = face.Face(coords=coords, coordsIndex=coordsIndex)
myMesh.verifyAttachments()


