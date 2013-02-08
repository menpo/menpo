from ibugMM.importer.model import ModelImporterFactory
from ibugMM.mesh.face import Face
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(True)
from tvtk.tools import ivtk
from tvtk.api import tvtk

## ioannis face
ioannis_path_2 = '/home/jab08/Dropbox/testData/ioannis_2.obj'
importer = ModelImporterFactory(ioannis_path_2)
ioannis = importer.generateFace()

# Create a cone:
cs = tvtk.ConeSource(resolution=100)
mapper = tvtk.PolyDataMapper(input=cs.output)
actor = tvtk.Actor(mapper=mapper)

# Now create the viewer:
#v = ivtk.IVTKWithCrustAndBrowser(size=(600,600))
#v.open()
#v.scene.add_actors(actor)  # or v.scene.add_actor(a)

