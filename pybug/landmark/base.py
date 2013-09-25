import abc
import copy
from pybug.transform.base import Transformable
from pybug.visualize import LandmarkViewer
from pybug.visualize.base import Viewable
import numpy as np


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
        super(Landmarkable, self).__init__()
        self._landmarks = LandmarkManager(self)

    @property
    def landmarks(self):
        return self._landmarks

    @landmarks.setter
    def landmarks(self, value):
        self._landmarks = copy.deepcopy(value)
        self._landmarks._target = self

    @property
    def n_landmark_groups(self):
        r"""
        The number of landmark groups on this object.

        :type: int
        """
        return self.landmarks.n_groups


class LandmarkManager(Transformable):
    """
    Class for storing and manipulating Landmarks associated with an object.
    This involves managing the internal dictionary, as well as providing
    convenience functions for operations like viewing.

    Parameters
    ----------
    target : :class:`pybug.landmarks.base.Landmarkable`
        The parent object that owns these landmarks
    """

    def __init__(self, target):
        super(LandmarkManager, self).__init__()
        self.__target = target
        self._landmark_groups = {}

    def __iter__(self):
        """
        Iterate over the internal landmark group dictionary
        """
        return iter(self._landmark_groups.iteritems())

    def __setitem__(self, group_label, landmark_group):
        self._landmark_groups[group_label] = copy.deepcopy(landmark_group)
        self._landmark_groups[group_label]._group_label = group_label
        self._landmark_groups[group_label]._target = self._target

    def __getitem__(self, group_label):
        return self._landmark_groups[group_label]

    @property
    def _target(self):
        return self.__target

    @_target.setter
    def _target(self, value):
        self.__target = value
        for group in self._landmark_groups.itervalues():
            group._target = self.__target

    @property
    def n_groups(self):
        """
        Total number of labels.

        :type: int
        """
        return len(self._landmark_groups)

    @property
    def has_landmarks(self):
        """
        Whether the object has landmarks or not

        :type: int
        """
        return self.n_groups == 0

    @property
    def group_labels(self):
        """
        All the labels for the landmark set.

        :type: List of strings
        """
        return self._landmark_groups.keys()

    def update(self, landmark_manager):
        new_landmark_manager = copy.deepcopy(landmark_manager)
        new_landmark_manager._target = self.__target
        self._landmark_groups.update(new_landmark_manager._landmark_groups)

    def _transform(self, transform):
        for group in self._landmark_groups.itervalues():
            group.landmarks._transform(transform)
        return self


class LandmarkGroup(Viewable):
    """

    """

    def __init__(self, target, group_label, pointcloud, labels_to_masks):
        super(LandmarkGroup, self).__init__()

        if not labels_to_masks:
            raise ValueError('Landmark groups are designed for their internal '
                             'state, other than owernship, to be immutable. '
                             'Empty label sets are not permitted.')

        if np.vstack(labels_to_masks.values()).shape[1] != pointcloud.n_points:
            raise ValueError('Each mask must have the same number of points '
                             'as the landmark pointcloud.')

        unlabelled_points = np.sum(labels_to_masks.values(), axis=0) == 0
        if np.any(unlabelled_points):
            raise ValueError('Every point in the landmark pointcloud must be '
                             'labelled. Points {0} were unlabelled.'.format(
                             np.nonzero(unlabelled_points)))

        self._group_label = group_label
        self._target = target
        self._pointcloud = pointcloud
        self._labels_to_masks = labels_to_masks

    def __iter__(self):
        """
        Iterate over the internal label dictionary
        """
        return iter(self._labels_to_masks)

    def __setitem__(self, label, indices):
        """
        Add a new label to the landmark group by adding a new set of indices

        Parameters
        ----------
        label : String
            Label of landmark.
        indices : (K,) ndarray
            Array of indices in to the pointcloud. Each index implies
            membership to the label.
        """
        mask = np.zeros(self._pointcloud.n_points, dtype=np.bool)
        mask[indices] = True
        self._labels_to_masks[label] = mask

    def __getitem__(self, label):
        """
        Get back a sub-pointcloud given a label. Internally only one pointcloud
        is kept, but we wish to have access to the pointcloud specific to a
        particular label.

        Parameters
        ----------
        label : Label of landmark

        Returns
        -------
        pointcloud : :class:`pybug.shape.pointcloud.Pointcloud`
            The pointcloud containing all the landmarks for the given label.
        """
        return self._pointcloud.from_mask(self._labels_to_masks[label])

    @property
    def group_label(self):
        """
        The label of this landmark group.

        :type: String
        """
        return self._group_label

    @property
    def labels(self):
        """
        The list of labels that belong to this group.

        :type: [strings]
        """
        return self._labels_to_masks.keys()

    @property
    def n_labels(self):
        """
        Number of labels in the group.

        :type: int
        """
        return len(self.labels)

    @property
    def landmarks(self):
        """
        The pointcloud representing all the landmarks in the group.

        :type: :class:`pybug.shape.pointcloud.Pointcloud`
        """
        return self._pointcloud

    @property
    def n_landmarks(self):
        """
        The total number of landmarks in the group.

        :type: int
        """
        return self._pointcloud.n_points

    def with_labels(self, labels):
        """
        Returns a new landmark group that contains only the given label.

        Parameters
        ----------
        label : String
            Label to filter on.

        Returns
        -------
        landmark_group : :class:`LandmarkGroup`
            A new landmark group with the same group label but containing only
            the given label.
        """
        # Make it easier to use by accepting a single string as well as a list
        if isinstance(labels, str):
            labels = [labels]
        return self._new_group_with_only_labels(labels)

    def without_labels(self, labels):
        """
        Returns a new landmark group that contains all labels EXCEPT the given
        label.

        Parameters
        ----------
        label : String
            Label to exclude.

        Returns
        -------
        landmark_group : :class:`LandmarkGroup`
            A new landmark group with the same group label but containing all
            labels except the given label.
        """
        # Make it easier to use by accepting a single string as well as a list
        if isinstance(labels, str):
            labels = [labels]
        labels_to_keep = list(set(self.labels).difference(labels))
        return self._new_group_with_only_labels(labels_to_keep)

    def _new_group_with_only_labels(self, labels):
        set_difference = set(labels).difference(self.labels)
        if len(set_difference) > 0:
            raise ValueError('Labels {0} do not exist in the landmark '
                             'group. Available labels are: {1}'.format(
                             list(set_difference), self.labels))

        masks_to_keep = [self._labels_to_masks[l] for l in labels
                         if l in self._labels_to_masks]
        overlap = np.sum(masks_to_keep, axis=0) > 0
        masks_to_keep = [l[overlap] for l in masks_to_keep]

        return LandmarkGroup(self._target, self.group_label,
                             self._pointcloud.from_mask(overlap),
                             dict(zip(labels, masks_to_keep)))

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
        target_viewer = self._target.view(figure_id=figure_id,
                                          new_figure=new_figure, **kwargs)
        landmark_viewer = LandmarkViewer(target_viewer.figure_id, False,
                                         self.group_label, self._pointcloud,
                                         self._labels_to_masks, self._target)

        return landmark_viewer.render(include_labels=include_labels, **kwargs)
