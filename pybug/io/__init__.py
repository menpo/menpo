import os
import mesh
#from pyassimp import pyassimp


def smartimport(filepath, **kwargs):
    """ Smart data importer. Chooses an appropriate importer based on the
    file extension of the data file past in. pass keepimporter=True as a kwarg
    if you want the actual importer object attached to the returned face object
    at face.importer.
    """
    keepimporter = kwargs.pop('keepimporter', False)
    ext = os.path.splitext(filepath)[-1]
    if ext == '.off':
        importer = mesh.OFFImporter(filepath, **kwargs)
    elif ext == '.wrl':
        importer = mesh.WRLImporter(filepath, **kwargs)
    elif ext == '.obj':
        importer = mesh.OBJImporter(filepath, **kwargs)
    else:
        raise Exception("I don't understand the file type " + `ext`)
    spatialdata = importer.build()
    if keepimporter:
        print 'attaching the importer at shape.importer'
        spatialdata.importer = importer
    return spatialdata


# def pyassimp_import(filepath):
#     scene = pyassimp.load(filepath)
#     if len(scene.meshes) > 1:
#         print 'warning - multipl'
#
