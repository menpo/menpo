import abc
from collections import OrderedDict, MutableMapping

import numpy as np

from menpo.base import Copyable
from menpo.transform.base import Transformable
from menpo.visualize import LandmarkViewer
from menpo.visualize.base import Viewable


class Landmarkable(Copyable):
    r"""
    Abstract interface for object that can have landmarks attached to them.
    Landmarkable objects have a public dictionary of landmarks which are
    managed by a :map:`LandmarkManager`. This means that
    different sets of landmarks can be attached to the same object.
    Landmarks can be N-dimensional and are expected to be some
    subclass of :map:`PointCloud`. These landmarks
    are wrapped inside a :map:`LandmarkGroup` object that performs
    useful tasks like label filtering and viewing.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._landmarks = None

    @abc.abstractproperty
    def n_dims(self):
        pass

    @property
    def landmarks(self):
        if self._landmarks is None:
            self._landmarks = LandmarkManager()
        return self._landmarks

    @property
    def has_landmarks(self):
        return self._landmarks is not None

    @landmarks.setter
    def landmarks(self, value):
        # firstly, make sure the dim is correct. Note that the dim can be None
        lm_n_dims = value.n_dims
        if lm_n_dims is not None and lm_n_dims != self.n_dims:
            raise ValueError(
                "Trying to set {}D landmarks on a "
                "{}D object".format(value.n_dims, self.n_dims))
        self._landmarks = value.copy()

    @property
    def n_landmark_groups(self):
        r"""
        The number of landmark groups on this object.

        :type: `int`
        """
        return self.landmarks.n_groups


class LandmarkableViewable(Landmarkable, Viewable):
    r"""
    Mixin for :map:`Landmarkable` and :map:`Viewable` objects. Provides a
    single helper method for viewing Landmarks and slf on the same figure.
    """

    def view_landmarks(self, figure_id=None, new_figure=False,
                       group=None, render_numbering=True,
                       render_legend=True,
                       with_labels=None, without_labels=None,
                       lmark_view_kwargs=None, obj_view_kwargs=None):
        """
        View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.

        Parameters
        ----------
        group : `str`, optional
            If ``None``, show all groups, else show only the provided group.
        render_numbering : `bool`, optional
            If ``True``, also render the label names next to the landmarks.
        render_legend : `bool`, optional
            If ``True``, also render a legend showing the landmark group symbols
            and colours.
        with_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        lmark_view_kwargs : `dict`, optional
            Passed through to the landmark viewer.
        obj_view_kwargs : `dict`, optional
            Passed through to this object's viewer.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        ValueError
            If the landmark manager doesn't contain the provided group label.
        """
        # Can't expand None so need a dict by default, but don't want
        # a mutual default kwarg
        obj_view_kwargs = obj_view_kwargs or {}
        lmark_view_kwargs = lmark_view_kwargs or {}

        self_view = self.view(figure_id=figure_id, new_figure=new_figure,
                              **obj_view_kwargs)
        landmark_view = self.landmarks.view(figure_id=self_view.figure_id,
                                            new_figure=False,
                                            group=group,
                                            targettype=type(self),
                                            render_numbering=render_numbering,
                                            render_legend=render_legend,
                                            with_labels=with_labels,
                                            without_labels=without_labels,
                                            **lmark_view_kwargs)
        return landmark_view


class LandmarkManager(MutableMapping, Transformable, Viewable):
    """
    Class for storing and manipulating Landmarks associated with an object.
    This involves managing the internal dictionary, as well as providing
    convenience functions for operations like viewing.

    A LandmarkManager ensures that all it's Landmarks are of the
    same dimensionality.

    """
    def __init__(self):
        super(LandmarkManager, self).__init__()
        self._landmark_groups = {}

    @property
    def n_dims(self):
        if self.n_groups != 0:
            # Python version independent way of getting the first value
            for v in self._landmark_groups.values():
                return v.n_dims
        else:
            return None

    def copy(self):
        r"""
        Generate an efficient copy of this :map:`LandmarkManager`.

        Returns
        -------

        ``type(self)``
            A copy of this object

        """
        # do a normal copy. The dict will be shallow copied - rectify that here
        new = Copyable.copy(self)
        for k, v in new._landmark_groups.items():
            new._landmark_groups[k] = v.copy()
        return new

    def __iter__(self):
        """
        Iterate over the internal landmark group dictionary
        """
        return iter(self._landmark_groups)

    def __setitem__(self, group, value):
        """
        Sets a new landmark group for the given label. This can be set using
        an existing landmark group, or using a PointCloud. Existing landmark
        groups will have their target reset. If a PointCloud is provided then
        all landmarks belong to a single label `all`.

        Parameters
        ----------
        group : `string`
            Label of new group.

        value : :map:`LandmarkGroup` or :map:`PointCloud`
            The new landmark group to set.

        Raises
        ------
        DimensionalityError
            If the landmarks and the shape are not of the same dimensionality.
        """
        from menpo.shape import PointCloud
        # firstly, make sure the dim is correct
        n_dims = self.n_dims
        if n_dims is not None and value.n_dims != n_dims:
            raise ValueError(
                "Trying to set {}D landmarks on a "
                "{}D LandmarkManager".format(value.n_dims, self.n_dims))
        if isinstance(value, PointCloud):
            # Copy the PointCloud so that we take ownership of the memory
            lmark_group = LandmarkGroup(
                value,
                OrderedDict([('all', np.ones(value.n_points, dtype=np.bool))]))
        elif isinstance(value, LandmarkGroup):
            # Copy the landmark group so that we now own it
            lmark_group = value.copy()
            # check the target is set correctly
            lmark_group._group_label = group
        else:
            raise ValueError('Valid types are PointCloud or LandmarkGroup')

        self._landmark_groups[group] = lmark_group

    def __getitem__(self, group=None):
        """
        Returns the group for the provided label.

        Parameters
        ---------
        group : `string`, optional
            The label of the group. If None is provided, and if there is only
            one group, the unambiguous group will be returned.
        Returns
        -------
        lmark_group : :map:`LandmarkGroup`
            The matching landmark group.
        """
        if group is None:
            if self.n_groups == 1:
                group = self.group_labels[0]
            else:
                raise ValueError("Cannot use None as a key as there are {} "
                                 "landmark groups".format(self.n_groups))
        return self._landmark_groups[group]

    def __delitem__(self, group):
        """
        Delete the group for the provided label.

        Parameters
        ---------
        group : `string`
            The label of the group.
        """
        del self._landmark_groups[group]

    def __len__(self):
        return len(self._landmark_groups)

    @property
    def n_groups(self):
        """
        Total number of labels.

        :type: `int`
        """
        return len(self._landmark_groups)

    @property
    def has_landmarks(self):
        """
        Whether the object has landmarks or not

        :type: `int`
        """
        return self.n_groups != 0

    @property
    def group_labels(self):
        """
        All the labels for the landmark set.

        :type: list of `string`
        """
        return self._landmark_groups.keys()

    def _transform_inplace(self, transform):
        for group in self._landmark_groups.values():
            group.lms._transform_inplace(transform)
        return self

    def view(self, figure_id=None, new_figure=False, group=None, **kwargs):
        """
        View all landmarks groups on the current manager.

        Parameters
        ----------
        group : `str`, optional
            If ``None``, show all groups, else show only the provided group.
        render_numbering : `boolean`, optional
            If ``True``, also render the label names next to the landmarks.
        render_legend : `bool`, optional
            If ``True``, also render the legend.
        with_labels : None or `str` or list of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : None or `str` or list of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        kwargs : `dict`, optional
            Passed through to the viewer.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        ValueError
            If the landmark manager doesn't contain the provided group label.
        """
        if group is None:
            viewers = []
            for g_label in self._landmark_groups:
                v = self._landmark_groups[g_label].view(figure_id=figure_id,
                                                        new_figure=new_figure,
                                                        group=g_label,
                                                        **kwargs)
                viewers.append(v)
            return viewers
        elif group in self._landmark_groups:
            return self._landmark_groups[group].view(
                figure_id=figure_id, new_figure=new_figure,
                group=group, **kwargs)
        else:
            raise ValueError('Unknown label {}'.format(group))

    def view_widget(self, popup=False):
        r"""
        Visualizes the landmark manager object using the
        menpo.visualize.widgets.visualize_shapes widget.

        Parameters
        -----------
        popup : `boolean`, optional
            If enabled, the widget will appear as a popup window.
        """
        from menpo.visualize import visualize_shapes
        visualize_shapes(self, figure_size=(7, 7), popup=popup)

    def __str__(self):
        out_string = '{}: n_groups: {}'.format(type(self).__name__,
                                               self.n_groups)
        if self.has_landmarks:
            for label in self:
                out_string += '\n'
                out_string += '({}): {}'.format(label, self[label].__str__())

        return out_string


class LandmarkGroup(MutableMapping, Copyable, Viewable):
    """
    An immutable object that holds a :map:`PointCloud` (or a subclass) and
    stores labels for each point. These labels are defined via masks on the
    :map:`PointCloud`. For this reason, the :map:`PointCloud` is considered to
    be immutable.

    The labels to masks must be within an `OrderedDict` so that semantic
    ordering can be maintained.

    Parameters
    ----------
    target : :map:`Landmarkable`
        The parent object of this landmark group.
    group : `string`
        The label of the group.
    pointcloud : :map:`PointCloud`
        The pointcloud representing the landmarks.
    labels_to_masks : `OrderedDict` of `string` to `boolean` `ndarrays`
        For each label, the mask that specifies the indices in to the
        pointcloud that belong to the label.
    copy : `boolean`, optional
        If ``True``, a copy of the :map:`PointCloud` is stored on the group.

    Raises
    ------
    ValueError
        If `dict` passed instead of `OrderedDict`
    ValueError
        If no set of label masks is passed.
    ValueError
        If any of the label masks differs in size to the pointcloud.
    ValueError
        If there exists any point in the pointcloud that is not covered
        by a label.
    """

    def __init__(self, pointcloud, labels_to_masks, copy=True):
        super(LandmarkGroup, self).__init__()

        if not labels_to_masks:
            raise ValueError('Landmark groups are designed for their internal '
                             'state, other than owernship, to be immutable. '
                             'Empty label sets are not permitted.')
        if np.vstack(labels_to_masks.values()).shape[1] != pointcloud.n_points:
            raise ValueError('Each mask must have the same number of points '
                             'as the landmark pointcloud.')
        if type(labels_to_masks) is dict:
            raise ValueError('Must provide an OrderedDict to maintain the '
                             'semantic meaning of the labels.')

        # Another sanity check
        self._labels_to_masks = labels_to_masks
        self._verify_all_labels_masked()

        if copy:
            self._pointcloud = pointcloud.copy()
            self._labels_to_masks = OrderedDict([(l, m.copy()) for l, m in
                                                 labels_to_masks.items()])
        else:
            self._pointcloud = pointcloud
            self._labels_to_masks = labels_to_masks

    def copy(self):
        r"""
        Generate an efficient copy of this :map:`LandmarkGroup`.

        Returns
        -------

        ``type(self)``
            A copy of this object

        """
        new = Copyable.copy(self)
        for k, v in new._labels_to_masks.items():
            new._labels_to_masks[k] = v.copy()
        return new

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
        label : `string`
            Label of landmark.

        indices : ``(K,)`` `ndarray`
            Array of indices in to the pointcloud. Each index implies
            membership to the label.
        """
        mask = np.zeros(self._pointcloud.n_points, dtype=np.bool)
        mask[indices] = True
        self._labels_to_masks[label] = mask

    def __getitem__(self, label=None):
        """
        Returns the PointCloud that contains this label represents on the group.
        This will be a subset of the total landmark group PointCloud.

        Parameters
        ----------
        label : `string`
            Label to filter on.

        Returns
        -------
        pcloud : :map:`PointCloud`
            The PointCloud that this label represents. Will be a subset of the
            entire group's landmarks.
        """
        if label is None:
            return self.lms.copy()
        return self._pointcloud.from_mask(self._labels_to_masks[label])

    def __delitem__(self, label):
        """
        Delete the semantic labelling for the provided label.

         .. note::

             You cannot delete a semantic label and leave the landmark group
             partially unlabelled. Landmark groups must contain labels for
             every point.

        Parameters
        ---------
        label : `string`
            The label to remove.

        Raises
        ------
        ValueError
            If deleting the label would leave some points unlabelled
        """
        # Pop the value off, which is akin to deleting it (removes it from the
        # underlying dict). However, we keep it around so we can check if
        # removing it causes an unlabelled point
        value_to_delete = self._labels_to_masks.pop(label)

        try:
            self._verify_all_labels_masked()
        except ValueError as e:
            # Catch the error, restore the value and re-raise the exception!
            self._labels_to_masks[label] = value_to_delete
            raise e

    def __len__(self):
        return len(self._labels_to_masks)

    @property
    def labels(self):
        """
        The list of labels that belong to this group.

        :type: list of `string`
        """
        return self._labels_to_masks.keys()

    @property
    def n_labels(self):
        """
        Number of labels in the group.

        :type: `int`
        """
        return len(self.labels)

    @property
    def lms(self):
        """
        The pointcloud representing all the landmarks in the group.

        :type: :map:`PointCloud`
        """
        return self._pointcloud

    @property
    def n_landmarks(self):
        """
        The total number of landmarks in the group.

        :type: `int`
        """
        return self._pointcloud.n_points

    @property
    def n_dims(self):
        """
        The dimensionality of these landmarks.

        :type: `int`
        """
        return self._pointcloud.n_dims

    def with_labels(self, labels=None):
        """
        Returns a new landmark group that contains only the given labels.

        Parameters
        ----------
        labels : `string` or list of `string`, optional
            Labels that should be kept in the returned landmark group. If
            None is passed, and if there is only one label on this group,
            the label will be substituted automatically.

        Returns
        -------
        landmark_group : :map:`LandmarkGroup`
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
        label : `string`
            Label to exclude.

        Returns
        -------
        landmark_group : :map:`LandmarkGroup`
            A new landmark group with the same group label but containing all
            labels except the given label.
        """
        # Make it easier to use by accepting a single string as well as a list
        if isinstance(labels, str):
            labels = [labels]
        labels_to_keep = list(set(self.labels).difference(labels))
        return self._new_group_with_only_labels(labels_to_keep)

    def _verify_all_labels_masked(self):
        """
        Verify that every point in the pointcloud is associated with a label.
        If any one point is not covered by a label, then raise a
        ``ValueError``.
        """
        unlabelled_points = np.sum(self._labels_to_masks.values(), axis=0) == 0
        if np.any(unlabelled_points):
            raise ValueError('Every point in the landmark pointcloud must be '
                             'labelled. Points {0} were unlabelled.'.format(
                np.nonzero(unlabelled_points)))

    def _new_group_with_only_labels(self, labels):
        """
        Deal with changing indices when you add and remove points. In this case
        we only deal with building a new dataset that keeps masks.

        Parameters
        ----------
        labels : list of `string`
            List of strings of the labels to keep

        Returns
        -------
        lmark_group : :map:`LandmarkGroup`
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

        return LandmarkGroup(self._pointcloud.from_mask(overlap),
                             OrderedDict(zip(labels, masks_to_keep)))

    def tojson(self):
        r"""
        Convert this `LandmarkGroup` to a dictionary JSON representation.

        Returns
        -------
        Dictionary with 'groups' key. Groups contains a landmark label set,
        containing the label, spatial points and connectivity information.
        Suitable or use in the by the `json` standard library package.
        """
        from menpo.shape import TriMesh, PointUndirectedGraph

        groups = []
        for label in self:
            pcloud = self[label]
            if isinstance(pcloud, TriMesh):
                connectivity = [t.tolist()
                                for t in pcloud.as_pointgraph().adjacency_array]
            elif isinstance(pcloud, PointUndirectedGraph):
                connectivity = [t.tolist()
                                for t in pcloud.adjacency_array]
            else:
                connectivity = []
            landmarks = [{'point': p.tolist()} for p in pcloud.points]
            groups.append({'connectivity': connectivity,
                           'label': label,
                           'landmarks': landmarks})
        return {'groups': groups}

    def view(self, figure_id=None, new_figure=False, targettype=None,
              render_numbering=True, render_legend=True, group='group',
              with_labels=None, without_labels=None, image_view=True,
              **kwargs):
        """
        View all landmarks. Kwargs passed in here will be passed through
        to the shapes view method.

        Parameters
        ----------
        targettype : `type`, optional
            Hint for the landmark viewer for the type of the object these
            landmarks are attached to. If ``None``, The landmarks will be
            visualized without special consideration for the type of the
            target. Mainly used for :map:`Image` subclasses.
        render_numbering : `bool`, optional
            If `True`, also render the label numbers next to the landmarks.
        render_legend : `bool`, optional
            If ``True``, also render the legend.
        group : `str`, optional
            The group label to prepend before the semantic labels
        with_labels : None or `str` or list of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : None or `str` or list of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        kwargs : `dict`, optional
            Passed through to the viewer.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        """
        if with_labels is not None and without_labels is not None:
            raise ValueError('You may only pass one of `with_labels` or '
                             '`without_labels`.')
        elif with_labels is not None:
            lmark_group = self.with_labels(with_labels)
        elif without_labels is not None:
            lmark_group = self.without_labels(without_labels)
        else:
            lmark_group = self  # Fall through

        landmark_viewer = LandmarkViewer(figure_id, new_figure,
                                         group, lmark_group._pointcloud,
                                         lmark_group._labels_to_masks)
        return landmark_viewer.render(render_numbering=render_numbering,
                                      render_legend=render_legend,
                                      image_view=image_view, **kwargs)

    def __str__(self):
        return '{}: n_labels: {}, n_points: {}'.format(
            type(self).__name__, self.n_labels, self.n_landmarks)
