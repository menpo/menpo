import abc
import numpy as np
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

    @property
    def n_landmark_groups(self):
        r"""
        The number of landmark groups on this object.

        :type: int
        """
        return len(self.landmarks)

    @property
    def _lms(self):
        r"""
        Convenient access to landmarks. If there is only one landmark group
        this returns the manager on that sole group. If not, returns 0.

        Note that this is just an interactive session convenience - never
        rely on this method in functions

        :type: :class:`LandmarkManager` or None
        """
        if self.n_landmark_groups == 1:
            return self.landmarks.values()[0]
        else:
            return None

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
        landmark_dict : dictionary (string,
                                   :class:`pybug.shape.pointcloud.Pointcloud`)
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

    def _all_landmarks_with_group_and_label(self, group=None, label=None):
        r"""
        Returns the point cloud of the landmarks matching the group and
        label arguments.

        Parameters
        ----------
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
             all landmarks in the group are used.

            Default: None

        Returns
        -------
        pc : :class:`pybug.shape.pointcloud.PointCloud`
            The point cloud of the landmarks found.
        """
        if self.n_landmark_groups == 0:
            raise ValueError("Cannot recover landmark pointcloud as there "
                             "are no landmarks on this object")

        if group is None:
            if self.n_landmark_groups > 1:
                raise ValueError("no group was provided and there are "
                                 "multiple groups. Specify a group, "
                                 "e.g. {}".format(self.landmarks.keys()[0]))
            else:
                group = self.landmarks.keys()[0]

        if label is None:
            pc = self.landmarks[group].all_landmarks
        else:
            pc = self.landmarks[group].with_label(label).all_landmarks
        return pc

    def _enforce_ownership_of_all_landmarks(self):
        r"""
        Loops through all landmark groups on self and ensures each has a
        target of self.
        """
        for manager in self.landmarks.iteritems():
            manager.target = self


class LandmarkManager(Viewable):
    """
    Class for storing and manipulating Landmarks associated with an object.
    This involves managing the internal dictionary, as well as providing
    convenience functions for operations like viewing.

    Parameters
    ----------
    target : :class:`pybug.landmarks.base.Landmarkable`
        The parent object that owns these landmarks
    label : string
        Name of landmark set
    landmark_dict : dictionary (string,
                         :class:`pybug.shape.pointcloud.Pointcloud`), optional
        Dictionary of labels and pointclouds representing all the landmarks

        Default: ``None``
    """

    def __init__(self, target, label, landmark_dict=None):
        self.landmark_dict = {}
        self.target = target
        self.label = label
        if landmark_dict:
            self.add_landmarks(landmark_dict)

    def __iter__(self):
        """
        Iterate over the internal landmark dictionary
        """
        return iter(self.landmark_dict.iteritems())

    @property
    def target(self):
        r"""
        The instance of :class:`Landmarkable` that this Landmark manager
        belongs to.

        :type : :class:`Landmarkable`
        """
        return self._target

    @target.setter
    def target(self, target):
        r"""
        Set the ownership of a landmark manager to an instance of
        :class:`Landmarkable`.
        """
        if not isinstance(target, Landmarkable):
            raise ValueError("Trying to set a target that is not "
                             "Landmarkable")
        else:
            self._target = target

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

    def add_landmarks(self, landmark_dict):
        r"""
        Add more landmarks to the current set. Expects a dictionary of
        labels to pointclouds. If the label already exists it will be
        replaced with the new value.

        Parameters
        ----------
        landmark_dict : dictionary (string,
                        :class:`pybug.shape.pointcloud.Pointcloud`), optional
            Dictionary of labels and pointclouds representing all the landmarks

        Raises
        ------
        :class:`pybug.exceptions.DimensionalityError`
            Raised when the dimensions of the landmarks don't match those of
            the parent shape.
        """
        for key, pointcloud in landmark_dict.iteritems():
            if pointcloud.n_dims == self.target.n_dims:
                self.landmark_dict[key] = pointcloud
            else:
                raise DimensionalityError("Dimensions of the landmarks must "
                                          "match the dimensions of the "
                                          "parent shape")

    def update_landmarks(self, label, landmarks):
        r"""
        Replace a given label with a new subclass of
        :class:`pybug.shape.pointcloud.Pointcloud`. If the label does not
        exist, then this becomes a new entry.

        Parameters
        ----------
        :param label : string
            The semantic meaning of the landmarks being added eg. eye
        :param landmarks : :class:`pybug.shape.pointcloud.Pointcloud`
            The landmark pointcloud.
        """
        self.landmark_dict[label] = landmarks

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
        return LandmarkManager(self.target, self.label,
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
        return LandmarkManager(self.target, self.label, new_dict)

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
        target_viewer = self.target.view(figure_id=figure_id,
                                         new_figure=new_figure, **kwargs)
        landmark_viewer = LandmarkViewer(target_viewer.figure_id, False,
                                         self.label, self.landmark_dict,
                                         self.target)

        return landmark_viewer.render(include_labels=include_labels, **kwargs)
