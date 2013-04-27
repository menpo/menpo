# whether collections have any role in pybug is not clear.
# for now this module remains here.
# TODO pybug collections - do we need them?
#
# from . import SpatialData
# from .mesh import TriMesh
#
#
# class SpatialDataCollectionError(Exception):
#     pass
#
#
# class ShapeClassError(SpatialDataCollectionError):
#     pass
#
#
# class SpatialDataCollection(object):
#     """ A bag of SpatialData. Provides functionality for
#     - viewing all the data in the set
#     - performing transformations on all pieces of data in the set
#
#     will enforce that all the added elements are instances of SpecialData
#     but that's it.
#     """
#
#     def __init__(self, spatialdataiter):
#         if not all(isinstance(x, SpatialData) for x in spatialdataiter):
#             notsd = [x for x in spatialdataiter
#                      if not isinstance(x, SpatialData)]
#             raise SpatialDataCollectionError('Can only add SpatialData'
#                                              + ' instances (' + str(notsd)
#                                              + ' are not)')
#         self.data = list(spatialdataiter)
#
#     def add_spatialdata(self, shape):
#         """ Adds an instance of shape to the collection
#         """
#         if not isinstance(shape, SpatialData):
#             raise SpatialDataCollectionError('Can only add SpatialData '
#                                              + ' instances')
#         else:
#             self.data.append(shape)
#
#
# class ShapeClass(SpatialDataCollection):
#     """A collection of SpatialData that all have the same
#     landmark configuration (and so can be considered to be of the same shape)
#     """
#
#     def __init__(self, spatialdataiter):
#         SpatialDataCollection.__init__(self, spatialdataiter)
#         try:
#             unique_lm_configs = set(x.landmarks.config for x in self.data)
#         except AttributeError:
#             raise ShapeClassError("All elements of a shape class must have "
#                                   "landmarks attached")
#         if len(unique_lm_configs) != 1:
#             raise ShapeClassError("All elements in shape class must have "
#                                   "landmarks with the same config")
#
#
# class TriMeshShapeClass(ShapeClass):
#     """A shape class that only contains TriMesh instances
#     """
#
#     def __init__(self, trimeshiter):
#         ShapeClass.__init__(self, trimeshiter)
#         if not all(isinstance(x, TriMesh) for x in self.data):
#             raise ShapeClassError("Trying to build a trimesh shape"
#                                               " class with non-trimesh "
#                                               "elements")
