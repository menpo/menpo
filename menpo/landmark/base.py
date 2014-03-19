import abc
import copy

import numpy as np

from menpo.transform.base import Transformable
from menpo.visualize import LandmarkViewer
from menpo.visualize.base import Viewable


class Landmarkable(object):
    r"""
    Abstract interface for object that can have landmarks attached to them.
    Landmarkable objects have a public dictionary of landmarks which are
    managed by a :class:`menpo.landmark.base.LandmarkManager`. This means that
    different sets of landmarks can be attached to the same object.
    Landmarks can be N-dimensional and are expected to be some
    subclass of :class:`menpo.shape.pointcloud.Pointcloud`. These landmarks
    are wrapped inside a :class:`LandmarkGroup` object that performs
    useful tasks like label filtering and viewing.
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


class LandmarkManager(Transformable, Viewable):
    """
    Class for storing and manipulating Landmarks associated with an object.
    This involves managing the internal dictionary, as well as providing
    convenience functions for operations like viewing.

    Parameters
    ----------
    target : :class:`menpo.landmarks.base.Landmarkable`
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

    def __setitem__(self, group_label, value):
        """
        Sets a new landmark group for the given label. This can be set using
        an existing landmark group, or using a PointCloud. Existing landmark
        groups will have their target reset. If a PointCloud is provided then
        all landmarks belong to a single label `all`.

        Parameters
        ----------
        group_label : String
            Label of new group.
        value : LandmarkGroup or PointCloud
            The new landmark group to set.

        Raises
        ------
        DimensionalityError
            If the landmarks and the shape are not of the same dimensionality.
        """
        from menpo.shape import PointCloud
        # firstly, make sure the dim is correct
        if value.n_dims != self._target.n_dims:
            from menpo.exception import DimensionalityError
            raise DimensionalityError(
                "Trying to set {}D landmarks on a "
                "{}D shape".format(value.n_dims, self._target.n_dims))
        if isinstance(value, PointCloud):
            lmark_group = LandmarkGroup(
                None, None, value,
                {'all': np.ones(value.n_points, dtype=np.bool)})
        elif isinstance(value, LandmarkGroup):
            lmark_group = copy.deepcopy(value)
        else:
            raise ValueError('Valid types are PointCloud or LandmarkGroup')

        self._landmark_groups[group_label] = lmark_group
        self._landmark_groups[group_label]._group_label = group_label
        self._landmark_groups[group_label]._target = self._target

    def __getitem__(self, group_label=None):
        """
        Returns the group for the provided label.

        Parameters
        ---------
        group_label : String, optional
            The label of the group. If None is provided, and if there is only
            one group, the unambiguous group will be returned.

            Default: None

        Returns
        -------
        lmark_group : :class:`LandmarkGroup`
            The matching landmark group.
        """
        if group_label is None:
            if self.n_groups == 1:
                group_label = self.group_labels[0]
            else:
                raise ValueError("Cannot use None as a key as there are {} "
                                 "landmark groups".format(self.n_groups))
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
        return self.n_groups != 0

    @property
    def group_labels(self):
        """
        All the labels for the landmark set.

        :type: List of strings
        """
        return self._landmark_groups.keys()

    def update(self, landmark_manager):
        """
        Update the manager with the groups from another manager. This performs
        a deep copy on the other landmark manager and resets it's target.

        Parameters
        ----------
        landmark_manager : :class:`LandmarkManager`
            The landmark manager to copy from.
        """
        new_landmark_manager = copy.deepcopy(landmark_manager)
        new_landmark_manager._target = self.__target
        self._landmark_groups.update(new_landmark_manager._landmark_groups)

    def _transform_inplace(self, transform):
        for group in self._landmark_groups.itervalues():
            group.lms._transform_inplace(transform)
        return self

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        """
        View all landmarks groups on the current manager.

        Parameters
        ----------
        include_labels : bool, optional
            If ``True``, also render the label names next to the landmarks.
        kwargs : dict, optional
            Passed through to the viewer.
        """
        for group in self._landmark_groups.itervalues():
            group._view(figure_id=figure_id, new_figure=new_figure, **kwargs)

    def __str__(self):
        out_string = '{}: n_groups: {}'.format(type(self).__name__,
                                               self.n_groups)
        if self.has_landmarks:
            for label, group in self:
                out_string += '\n'
                out_string += '({}): {}'.format(label, group.__str__())

        return out_string


class LandmarkGroup(Viewable):
    """
    An immutable object that holds a PointCloud (or a subclass) and stores
    labels for each point. These labels are defined via masks on the
    pointcloud. For this reason, the pointcloud is considered to be immutable.

    Parameters
    ----------
    target : :class:`menpo.landmarks.base.Landmarkable`
        The parent object of this landmark group.
    group_label : String
        The label of the group.
    pointcloud : :class:`menpo.shape.pointcloud.PointCloud`
        The pointcloud representing the landmarks.
    labels_to_masks : dict of string to boolean ndarrays
        For each label, the mask that specifies the indices in to the
        pointcloud that belong to the label.
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
        Returns a new landmark group that contains ONLY the specified label.

        Parameters
        ----------
        label : String
            Label to filter on.

        Returns
        -------
        landmark_group : :class:`LandmarkGroup`
            A new landmark group with a single label.
        """
        return self.with_labels(label)

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
    def lms(self):
        """
        The pointcloud representing all the landmarks in the group.

        :type: :class:`menpo.shape.pointcloud.Pointcloud`
        """
        return self._pointcloud

    @property
    def n_landmarks(self):
        """
        The total number of landmarks in the group.

        :type: int
        """
        return self._pointcloud.n_points

    @property
    def n_dims(self):
        """
        The dimensionality of these landmarks.

        :type: int
        """
        return self._pointcloud.n_dims

    def with_labels(self, labels=None):
        """
        Returns a new landmark group that contains only the given labels.

        Parameters
        ----------
        labels : String or List of strings, optional
            Labels that should be kept in the returned landmark group. If
            None is passed, and if there is only one label on this group,
            the label will be substituted automatically.

            Default: None
        Returns
        -------
        landmark_group : :class:`LandmarkGroup`
            A new landmark group with the same group label but containing only
            the given label.
        """
        # make it easier by allowing None when there is only one label
        if labels is None:
            if self.n_labels == 1:
                labels = self.labels
            else:
                raise ValueError("Cannot use None as there are "
                                 "{} labels".format(self.n_labels))
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
        """
        Deal with changing indices when you add and remove points. In this case
        we only deal with building a new dataset that keeps masks.

        Parameters
        ----------
        labels : [String]
            List of strings of the labels to keep

        Returns
        -------
        lmark_group : :class:`LandmarkGroup`
            The new landmark group with only the requested labels.
        """
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

    def __str__(self):
        return '{}: label: {}, n_labels: {}, n_points: {}'.format(
            type(self).__name__, self.group_label, self.n_labels,
            self.n_landmarks)
