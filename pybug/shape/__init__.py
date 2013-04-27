import abc


class Shape(object):
    """ Abstract representation of shape. All subclasses will have some
     data where semantic meaning can be assigned to the i'th item. Shape
     couples a LandmarkManager to this item of data, meaning,
     all subclasses are landmarkable. Note this does not mean that all
     subclasses need have a spatial meaning (i.e. the 7'th node of a Graph
     can still be landmarked)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.landmarks = LandmarkManager(self)

    @abc.abstractproperty
    def _n_landmarkable_items(self):
        """
        The number of items that can be landmarked on this Shape.
        """
        pass

    @abc.abstractmethod
    def _landmark_at_index(self, index):
        """
        Retrieve the item at the index position

        :param index: Index position
        """
        pass

    @abc.abstractmethod
    def _add_meta_landmark_item(self, item):
        """
        Adds the item as a new meta item, returning the index into the meta
        items (the meta_index). If this Shape is not capable of adding meta
        items, instead return False.
        :param item: The new item to be added
        :return meta_index: The index into the meta landmarks of the added
        item OR False (if meta items cannot be added)
        """
        pass

    @abc.abstractmethod
    def _meta_landmark_at_meta_index(self, index):
        """
        Returns the meta item at the meta index position.
        """
        pass


from pybug.shape.pointcloud import PointCloud
from pybug.shape.mesh import TriMesh, FastTriMesh
from pybug.shape.landmarks import LandmarkManager

