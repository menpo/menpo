from ibugMM.mesh.face     import Face
from ibugMM.importer.model import OBJImporter
import numpy as np

objPath = '/home/jab08/testData/ioannis_001/exports/ioannis_001_022.obj'
oimporter = OBJImporter(objPath)
oFace = oimporter.generateFace()

