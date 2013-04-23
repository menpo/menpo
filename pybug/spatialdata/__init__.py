

class SpatialData(object):
    """ Abstract representation of a n-dimensional piece of spatial data.
    This could be simply be a set of vectors in an n-dimensional space,
    or a structured surface or mesh. At this level of abstraction we only
    define basic metadata that can be attached to all kinds of spatial
    data.
    """

    def __init__(self):
        pass


from pybug.spatialdata.pointcloud import PointCloud, PointCloud3d
from pybug.spatialdata.mesh import TriMesh, FastTriMesh

