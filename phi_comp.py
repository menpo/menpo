from ibugMM.importer.model import import_face 
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(True)

sphere = import_face('~/sphere-160k.obj')
geo = sphere.geodesics_about_vertices([0])
plt.hist(geo['phi'], bins=1000)

