from matplotlib import cm
import numpy as np
from pybug.exceptions import DimensionalityError
from pybug.visualize import LandmarkViewer


class LandmarkManager(object):
    """
    Class for storing and manipulating Landmarks associated with a Shape.
    """

    def __init__(self, shape, label, landmark_dict=None):
        """
        """
        self.landmark_dict = {}
        self.shape = shape
        self.label = label
        if landmark_dict:
            self.add_landmarks(landmark_dict)

    def add_landmarks(self, landmark_dict):
        for key, pointcloud in landmark_dict.iteritems():
            if pointcloud.n_dims == self.shape.n_dims:
                self.landmark_dict[key] = pointcloud
            else:
                raise DimensionalityError("Dimensions of the landmarks must "
                                          "match the dimensions of the "
                                          "parent shape")

    def with_label(self, label):
        return LandmarkManager(self.shape, self.label,
                               {label: self.landmark_dict[label]})

    def without_label(self, label):
        new_dict = dict(self.landmark_dict)
        del new_dict[label]
        return LandmarkManager(self.shape, self.label, new_dict)

    def view(self, include_labels=True, cmap=cm.jet,
             halign='center', valign='bottom', size=6, **kwargs):
        """
        View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.
        """
        shape_viewer = self.shape.view(**kwargs)
        return LandmarkViewer(self.label, self.landmark_dict).view(
            onviewer=shape_viewer, include_labels=include_labels, cmap=cmap,
            halign=halign, valign=valign, size=size, **kwargs)

    @property
    def labels(self):
        return self.landmark_dict.keys()

    @property
    def landmarks(self):
        return self.landmark_dict.values()

    @property
    def all_landmarks(self):
        from pybug.shape import PointCloud

        all_points = [x.points for x in self.landmarks]
        all_points = np.concatenate(all_points, axis=0)
        return PointCloud(all_points)

    @property
    def n_labels(self):
        return len(self.landmark_dict)

    @property
    def n_landmarks(self):
        return sum([x.n_points for x in self.landmark_dict.values()])