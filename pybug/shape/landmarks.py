import numpy as np
from pybug.exceptions import DimensionalityError
from pybug.visualize import PointCloudViewer3d, LabelViewer3d


class LandmarkManager(object):
    """
    Class for storing and manipulating Landmarks associated with a Shape.
    """

    def __init__(self, shape, landmark_dict=None):
        """
        """
        self.landmark_dict = {}
        self.shape = shape
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
        return LandmarkManager(self.shape, {label: self.landmark_dict[label]})

    def without_label(self, label):
        new_dict = dict(self.landmark_dict)
        del new_dict[label]
        return LandmarkManager(self.shape, new_dict)

    def _rebuild(self, landmarks):
        return LandmarkManager(self.shape, landmark_dict=landmarks)

    def view(self, **kwargs):
        """ View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.
        """
        lms = np.array([x.feature for x in self])
        labels = [x.numbered_label for x in self]
        pcviewer = self.shape.view(**kwargs)
        pointviewer = PointCloudViewer3d(lms)
        pointviewer.view(onviewer=pcviewer)
        lmviewer = LabelViewer3d(lms, labels, offset=np.array([0, 16, 0]))
        lmviewer.view(onviewer=pcviewer)
        return lmviewer

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

    @property
    def config(self):
        """A frozen set specifying all the landmarks numbered labels
        """
        return frozenset(x.numbered_label for x in self)