import numpy as np
from pybug.visualization import PointCloudViewer3d, LabelViewer3d


class Landmark(object):
    """ An object representing an annotated feature.
    Only makes sense in the context of a parent Shape instance, and so
    one is required at construction.
    """

    def __init__(self, shape, shape_index, label, label_index):
        self.shape = shape
        self.index = shape_index
        self.label = label
        self.label_index = label_index

    @property
    def feature(self):
        return list(self.shape.points_and_metapoints[self.index])

    @property
    def numbered_label(self):
        return self.label + '_' + str(self.label_index)


class ReferenceLandmark(Landmark):
    """A Landmark that references a point that is a part of a point cloud
    """

    def __init__(self, shape, shape_index, label, label_index):
        Landmark.__init__(self, shape, shape_index,
                          label, label_index)
        if shape_index < 0 or shape_index > self.shape.n_points:
            raise Exception("Reference landmarks have to have an index "
                            + "in the range 0 < i < n_points of the parent "
                            + "shape")


class MetaLandmark(Landmark):
    """A landmark that is totally separate from the parent shape."
    """

    def __init__(self, shape, metapoint, label, label_index):
        pointcloud_index = shape.addmetapoint(metapoint)
        Landmark.__init__(self, shape, pointcloud_index,
                          label, label_index)

    @property
    def metapoint_index(self):
        """ How far into the metapoints part of the array this metapoint is
        """
        return self.index - self.shape.n_points - 1


class LandmarkManager(object):
    """Class for storing and manipulating Landmarks associated with a Shape.
    Landmarks index into the points and metapoints of the associated
    PointCloud. Landmarks which are explicitly given as coordinates would
    be entirely constructed from metapoints, whereas point indexed landmarks
    would be composed entirely of points. This class can handle any arbitrary
    mixture of the two.
    """

    def __init__(self, pointcloud, landmarks=None):
        """ shape - the shape whose these landmarks apply to
        landmarks - an existing list of landmarks to initialize this manager to
        """
        if landmarks is None:
            landmarks = []
        self.pointcloud = pointcloud
        self._data = []
        if landmarks:
            pcs = set(lm.pointcloud for lm in landmarks)
            if len(pcs) != 1:
                raise Exception('Building a LandmarkManager using Landmarks '
                                'with non-compatible pointclouds')
            if landmarks[0].pointcloud is not self.pointcloud:
                raise Exception('Building a LandmarkManager using Landmarks '
                                'with a different shape to self')
            self._data = landmarks
            self._sort_data()

    def __iter__(self):
        self._i = -1
        return self

    def next(self):
        self._i += 1
        if self._i == len(self._data):
            raise StopIteration
        return self._data[self._i]

    def add_reference_landmarks(self, landmark_dict):
        for k, v in landmark_dict.iteritems():
            for i, index in enumerate(v):
                lm = ReferenceLandmark(self.pointcloud, index, k, i)
                self._data.append(lm)
        self._sort_data()

    def _sort_data(self):
        """ Sorts the data by the numbered_label. Ensures that iteration
        over self is always in a consistent order.
        """
        self._data.sort(key=lambda x: x.numbered_label)

    def reference_landmarks(self):
        return self._rebuild([x for x in self._data
                              if isinstance(x, ReferenceLandmark)])

    def meta_landmarks(self):
        return self._rebuild([x for x in self._data
                              if isinstance(x, MetaLandmark)])

    def with_label(self, label):
        return self._rebuild([x for x in self._data
                              if x.label == label])

    def without_label(self, label):
        return self._rebuild([x for x in self._data
                              if x.label != label])

    def _rebuild(self, landmarks):
        return LandmarkManager(self.pointcloud, landmarks=landmarks)

    def view(self, **kwargs):
        """ View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.
        """
        lms = np.array([x.feature for x in self])
        labels = [x.numbered_label for x in self]
        pcviewer = self.pointcloud.view(**kwargs)
        pointviewer = PointCloudViewer3d(lms)
        pointviewer.view(onviewer=pcviewer)
        lmviewer = LabelViewer3d(lms, labels, offset=np.array([0, 16, 0]))
        lmviewer.view(onviewer=pcviewer)
        return lmviewer

    def __len__(self):
        return len(self._data)

    @property
    def n_labels(self):
        return len(set(x.label for x in self))

    @property
    def config(self):
        """A frozen set specifying all the landmarks numbered labels
        """
        return frozenset(x.numbered_label for x in self)