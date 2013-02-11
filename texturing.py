from tvtk.api import tvtk
from tvtk.tools import ivtk
from ibugMM.importer.model import ModelImporterFactory
import numpy as np

ioannis_path_1 = '/home/jab08/Dropbox/testData/ioannis_1.obj'
importer = ModelImporterFactory(ioannis_path_1)
o_ioannis_1 = importer.generateFace()

