import numpy as np
import abc
from pybug.exceptions import DimensionalityError
from pybug.visualize import LandmarkViewer
from pybug.visualize.base import Viewable


class Landmarkable(object):
    r"""
    Abstract interface for object that can have landmarks attached to them.
    Landmarkable objects have a public dictionary of landmarks which are
    managed by a :class:`pybug.landmark.base.LandmarkManager`. This means that
    different sets of landmarks can be attached to the same object.
    Landmarks can be N-dimensional and are expected to be some
    subclass of :class:`pybug.shape.pointcloud.Pointcloud`.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.landmarks = {}

    def add_landmark_set(self, label, landmark_dict):
        r"""
        Add a new set of landmarks to the object. These landmarks should
        come in the form of a dictionary. Each key is the semantic
        meaning of the landmark and the landmarks themselves are of some
        subclass of :class:`pybug.shape.pointcloud.Pointcloud`.
        The provided label becomes the key for the landmark dictionary. It's
        important to note that any subclass of PointCloud can be used,
        it is just assumed that it contains a numpy array of points.

        Parameters
        ----------
        label : string
            The name of the set of landmarks
        landmark_dict : dictionary (string, :class:`pybug.shape.pointcloud.Pointcloud`)
            Dictionary of labels and pointclouds representing all the landmarks

        Raises
        ------
        ValueError
            If ``landmark_dict`` is not of type dictionary.
        """
        if not isinstance(landmark_dict, dict):
            raise ValueError('Landmark set must be of type dictionary')
        else:
            self.landmarks[label] = LandmarkManager(
                self, label, landmark_dict=landmark_dict)

    def get_landmark_set(self, label):
        return self.landmarks[label]


class LandmarkManager(Viewable):
    """
    Class for storing and manipulating Landmarks associated with an object.
    This involves managing the internal dictionary, as well as providing
    convenience functions for operations like viewing.

    Parameters
    ----------
    shape : :class:`pybug.landmarks.base.Landmarkable`
        The parent object that owns these landmarks
    label : string
        Name of landmark set
    landmark_dict : dictionary (string, :class:`pybug.shape.pointcloud.Pointcloud`), optional
        Dictionary of labels and pointclouds representing all the landmarks

        Default: ``None``
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

        Parameters
        ----------
        landmark_dict : dictionary (string, :class:`pybug.shape.pointcloud.Pointcloud`), optional
            Dictionary of labels and pointclouds representing all the landmarks

        Raises
        ------
        :class:`pybug.exceptions.DimensionalityError`
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
        :class:`pybug.shape.pointcloud.Pointcloud`. If the label does not
        exist, then this becomes a new entry.

        Parameters
        ----------
        :param label : string
            The semantic meaning of the landmarks being added eg. eye
        :param lmarks : :class:`pybug.shape.pointcloud.Pointcloud`
            The landmark pointcloud.
        """
        self.landmark_dict[label] = lmarks

    def with_label(self, label):
        """
        Return a new LandmarkManager that only contains the given label.
        This is useful for things like only viewing the 'eye' landmarks.

        Parameters
        ----------
        :param label : string
            The label of the landmarks to view

        Returns
        -------
        landmark_manager : :class:`LandmarkManager`
            New landmark manager containing only the given label.
        """
        return LandmarkManager(self.shape, self.label,
                               {label: self.landmark_dict[label]})

    def without_label(self, label):
        """
        Return a new LandmarkManager that contains all landmarks except
        those with the given label.
        This is useful for things like filtering out the 'eye' landmarks when
        viewing.

        Parameters
        ----------
        :param label : string
            The label of the landmarks to view

        Returns
        -------
        landmark_manager : :class:`LandmarkManager`
            New landmark manager excluding the given label.
        """
        new_dict = dict(self.landmark_dict)
        del new_dict[label]
        return LandmarkManager(self.shape, self.label, new_dict)

    def _view(self, figure_id=None, new_figure=False, include_labels=True,
              **kwargs):
        """
        View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.

        Parameters
        ----------
        include_labels : bool, optional
            If ``True``, also render the label names next to the landmarks.
        kwargs : dict, optional
            Passed through to the viewer.
        """
        shape_viewer = self.shape.view(figure_id=figure_id,
                                       new_figure=new_figure, **kwargs)
        lmark_viewer = LandmarkViewer(shape_viewer.figure_id, False,
                                      self.label, self.landmark_dict,
                                      self.shape)

        return lmark_viewer.render(include_labels=include_labels, **kwargs)

    @property
    def labels(self):
        """
        All the labels for the landmark set.

        :type: List of strings
        """
        return self.landmark_dict.keys()

    @property
    def landmarks(self):
        """
        All the landmarks for the set.

        :type: List of :class:`pybug.shape.pointcloud.Pointcloud`
        """
        return self.landmark_dict.values()

    @property
    def all_landmarks(self):
        """
        A new pointcloud that contains all the points within the landmark
        set.

        Iterates over the dictionary and creates a single PointCloud using
        all the points.

        :type: :class:`pybug.shape.pointcloud.Pointcloud`
        """
        from pybug.shape import PointCloud

        all_points = [x.points for x in self.landmarks]
        all_points = np.concatenate(all_points, axis=0)
        return PointCloud(all_points)

    @property
    def n_labels(self):
        """
        Total number of labels.

        :type: int
        """
        return len(self.landmark_dict)

    @property
    def n_landmarks(self):
        """
        Total number of landmarks (across ALL keys).

        Iterate over each PointCloud and sum the number of points. Therefore,
        this is the same as the number of points of the all_landmarks property.

        :type: int
        """
        return sum([x.n_points for x in self.landmark_dict.values()])