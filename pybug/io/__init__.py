import os
import mesh

def smartimport(filepath, **kwargs):
    """ Smart data importer. Chooses an appropriate importer based on the
    file extension of the data file past in. pass keepimporter=True as a kwarg
    if you want the actual importer object attached to the returned face object
    at face.importer.
    """
    ext = os.path.splitext(filepath)[-1]
    if ext == '.off':
        importer = mesh.OFFImporter(filepath, **kwargs)
    elif ext == '.wrl':
        importer = mesh.WRLImporter(filepath, **kwargs)
    elif ext == '.obj':
        importer = mesh.OBJImporter(filepath, **kwargs)
    else:
       raise Exception("I don't understand the file type " + `ext`)
       return None
    spatialdata = importer.build()
    if kwargs.get('keepimporter', False):
       print 'attaching the importer at spatialdata.importer'
       spatialdata.importer = importer
    return spatialdata
