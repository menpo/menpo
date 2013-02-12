import visvis as vv
from ibugMM.importer.model import ModelImporterFactory
import numpy as np

vv.figure()
a = vv.gca()

## ioannis face
ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1.obj'
importer = ModelImporterFactory(ioannis_path_1)
o_ioannis_1 = importer.generateFace()

mesh = vv.wobjects.Mesh(None, o_ioannis_1.coords, faces=o_ioannis_1.coords_index)
app = vv.use()
app.Run()

