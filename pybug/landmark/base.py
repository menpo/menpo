import numpy as np
import abc
from pybug.exceptions import DimensionalityError
from pybug.visualize import LandmarkViewer


class Landmarkable(object):
    r"""
    Abstract interface for object that can have landmarks attached to them.
    Landmarkable objects have a public dictionary of landmarks which are
    managed by a :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    . This means that different sets of landmarks can be attached to the
    same object. Landmarks can be N-dimensional and are expected to be some
    subclass of :class:`PointCloud <pybug.shape.pointcloud.Pointcloud>`.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.landmarks = {}

    def add_landmark_set(self, label, landmark_dict):
        r"""
        Add a new set of landmarks to the object. These landmarks should
        come in the form of a dictionary. Each key is the semantic
        meaning of the landmark and the landmarks themselves are of some
        subclass of :class:`PointCloud <pybug.shape.pointcloud.Pointcloud>`.
        The provided label becomes the key for the landmark dictionary. It's
        important to note that any subclass of PointCloud can be used,
        it is just assumed that it contains a numpy array of points.

        :param label: The name of the set of landmarks
        :type label: String
        :param landmark_dict: Dictionary of labels and pointclouds
            representing all the landmarks
        :type landmark_dict: Python dictionary. Keys are strings,
            values are :class:`pointclouds <pybug.shape.pointcloud.Pointcloud>`
        :raise:
        """
        if not isinstance(landmark_dict, dict):
            raise ValueError('Landmark set must be of type dictionary')
        else:
            self.landmarks[label] = LandmarkManager(
                self, label, landmark_dict=landmark_dict)

    def get_landmark_set(self, label):
        return self.landmarks[label]


class LandmarkManager(object):
    """
    Class for storing and manipulating Landmarks associated with an object.
    This involves managing the internal dictionary, as well as providing
    convenience functions for operations like viewing.

    :param shape: The parent object that owns these landmarks
    :type shape: :class:`Landmarkable <pybug.landmarks.base.Landmarkable>`
    :param label: Name of landmark set
    :type label: String
    :param landmark_dict: Dictionary of labels and pointclouds representing
        the landmark set
    :type landmark_dict: Python dictionary. Keys are strings,
        values are :class:`pointclouds <pybug.shape.pointcloud.Pointcloud>`
    """

    def __init__(self, shape, label, landmark_dict=None):
        self.landmark_dict = {}
        self.shape = shape
        self.label = label
        if landmark_dict:
            self.add_landmarks(landmark_dict)

    def __iter__(self):
        """
        Iterate over the internal landmark dictionary
        """
        return iter(self.landmark_dict.iteritems())

    def add_landmarks(self, landmark_dict):
        r"""
        Add more landmarks to the current set. Expects a dictionary of
        labels to pointclouds. If the label already exists it will be
        replaced with the new value.

        :param landmark_dict: Dictionary of labels and pointclouds representing
            the landmark set
        :type landmark_dict: Python dictionary. Keys are strings,
            values are :class:`pointclouds <pybug.shape.pointcloud.Pointcloud>`
        :raise:
            :class:`DimensionalityError <pybug.exceptions.DimensionalityError>`
            Raised when the dimensions of the landmarks don't match those of
            the parent shape.
        """
        for key, pointcloud in landmark_dict.iteritems():
            if pointcloud.n_dims == self.shape.n_dims:
                self.landmark_dict[key] = pointcloud
            else:
                raise DimensionalityError("Dimensions of the landmarks must "
                                          "match the dimensions of the "
                                          "parent shape")

    def update_landmarks(self, label, lmarks):
        r"""
        Replace a given label with a new subclass of
        :class:`pointclouds <pybug.shape.pointcloud.Pointcloud>`. If the
        label does not exist, then this becomes a new entry.

        :param label: The semantic meaning of the landmarks being added eg.
            eye
        :type label: String
        :param lmarks: A collection of landmarks
        :type lmarks: :class:`PointCloud <pybug.shape.pointcloud.Pointcloud>`
        """
        self.landmark_dict[label] = lmarks

    def with_label(self, label):
        """
        Return a new LandmarkManager that only contains the given label.
        This is useful for things like only viewing the 'eye' landmarks.

        :param label: The label of the landmarks to view
        :type label: String
        :return: New landmark manager containing only the given label.
        :rtype: :class:`LandmarkManager <pybug.landmarks.base.LandmarkManager>`
        """
        return LandmarkManager(self.shape, self.label,
                               {label: self.landmark_dict[label]})

    def without_label(self, label):
        """
        Return a new LandmarkManager that contains all landmarks except
        those with the given label.
        This is useful for things like filtering out the 'eye' landmarks when
        viewing.

        :param label: The label of the landmarks to exclude
        :type label: String
        :return: New landmark manager excluding the given label.
        :rtype: :class:`LandmarkManager <pybug.landmarks.base.LandmarkManager>`
        """
        new_dict = dict(self.landmark_dict)
        del new_dict[label]
        return LandmarkManager(self.shape, self.label, new_dict)

    def view(self, include_labels=True, **kwargs):
        """
        View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.

        :keyword include_labels: If True, also render the label names next
        to the landmarks.
        """
        shape_viewer = self.shape.view(**kwargs)
        return LandmarkViewer(self.label, self.landmark_dict, self.shape).view(
            onviewer=shape_viewer, include_labels=include_labels, **kwargs)

    @property
    def labels(self):
        """
        All the labels for the landmark set.

        :return: Landmark labels
        :rtype: List of strings
        """
        return self.landmark_dict.keys()

    @property
    def landmarks(self):
        """
        All the landmarks for the set.

        :return: Landmarks
        :rtype: List of
            :class:`PointClouds <pybug.shape.pointcloud.Pointcloud>`
        """
        return self.landmark_dict.values()

    @property
    def all_landmarks(self):
        """
        A new pointcloud that contains all the points within the landmark
        set. Iterates over the dictionary and creates a single PointCloud
        using all the points.

        :return: All the landmarks in the set.
        :rtype: :class:`PointClouds <pybug.shape.pointcloud.Pointcloud>`
        """
        from pybug.shape import PointCloud

        all_points = [x.points for x in self.landmarks]
        all_points = np.concatenate(all_points, axis=0)
        return PointCloud(all_points)

    @property
    def n_labels(self):
        """
        Total number of labels

        :return: Label count
        :rtype: int
        """
        return len(self.landmark_dict)

    @property
    def n_landmarks(self):
        """
        Total number of landmarks (across ALL keys). Iterate over each
        PointCloud and sum the number of points. Therefore, this is the same
         as the number of points of the all_landmarks property.

        :return: Landmark count
        :rtype: int
        """
        return sum([x.n_points for x in self.landmark_dict.values()])