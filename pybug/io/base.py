import abc
import os
# from pybug.io import mesh
#
#
# def smartimport(filepath, **kwargs):
#     """ Smart data importer. Chooses an appropriate importer based on the
#     file extension of the data file past in. pass keepimporter=True as a kwarg
#     if you want the actual importer object attached to the returned face object
#     at face.importer.
#     """
#     keepimporter = kwargs.pop('keepimporter', False)
#     ext = os.path.splitext(filepath)[-1]
#     if ext == '.off':
#         importer = mesh.OFFImporter(filepath, **kwargs)
#     elif ext == '.wrl':
#         importer = mesh.WRLImporter(filepath, **kwargs)
#     elif ext == '.obj':
#         importer = mesh.OBJImporter(filepath, **kwargs)
#     else:
#         raise Exception("I don't understand the file type " + `ext`)
#     shape = importer.build()
#     if keepimporter:
#         print 'attaching the importer at shape.importer'
#         shape.importer = importer
#     return shape


class Importer:
    """
    Abstract representation of an Importer. Construction on an importer
    takes a resource path and Imports the data into the Importers internal
    representation. To get an instance of one of our datatypes you access
    the shape method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        self.filepath = os.path.abspath(os.path.expanduser(filepath))
        self.filename = os.path.splitext(os.path.basename(self.filepath))[0]
        self.extension = os.path.splitext(self.filepath)[1]
        self.folder = os.path.dirname(self.filepath)

    @abc.abstractmethod
    def shape(self):
        pass
