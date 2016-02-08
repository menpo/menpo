from collections import OrderedDict, MutableMapping
import fnmatch

import numpy as np

from menpo.base import Copyable
from menpo.transform.base import Transformable
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

    def __init__(self):
        self._landmarks = None

    def n_dims(self):
        """
        The total number of dimensions.

        :type: `int`
        """
        raise NotImplementedError()

    @property
    def landmarks(self):
        """
        The landmarks object.

        :type: :map:`LandmarkManager`
        """
        if self._landmarks is None:
            self._landmarks = LandmarkManager()
        return self._landmarks

    @property
    def has_landmarks(self):
        """
        Whether the object has landmarks.

        :type: `bool`
        """
        return self._landmarks is not None and self.landmarks.n_groups != 0

    @landmarks.setter
    def landmarks(self, value):
        """
        Landmarks setter.

        Parameters
        ----------
        value : :map:`LandmarkManager`
            The landmarks to set.
        """
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


class LandmarkManager(MutableMapping, Transformable):
    """Store for :map:`LandmarkGroup` instances associated with an object

    Every :map:`Landmarkable` instance has an instance of this class available
    at the ``.landmarks`` property.  It is through this class that all access
    to landmarks attached to instances is handled. In general the
    :map:`LandmarkManager` provides a dictionary-like interface for storing
    landmarks. :map:`LandmarkGroup` instances are stored under string keys -
    these keys are refereed to as the **group name**. A special case is
    where there is a single unambiguous :map:`LandmarkGroup` attached to a
    :map:`LandmarkManager` - in this case ``None`` can be used as a key to
    access the sole group.


    Note that all landmarks stored on a :map:`Landmarkable` in it's attached
    :map:`LandmarkManager` are automatically transformed and copied with their
    parent object.
    """
    def __init__(self):
        super(LandmarkManager, self).__init__()
        self._landmark_groups = {}

    @property
    def n_dims(self):
        """
        The total number of dimensions.

        :type: `int`
        """
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
        if group is None:
            raise ValueError('Cannot set using the key `None`. `None` has a '
                             'reserved meaning for landmark groups.')

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

        :type: `list` of `str`
        """
        # Convert to list so that we can index immediately, as keys()
        # is a generator in Python 3
        return list(self._landmark_groups.keys())

    def keys_matching(self, glob_pattern):
        r"""
        Yield only landmark group names (keys) matching a given glob.

        Parameters
        ----------
        glob_pattern : `str`
            A glob pattern e.g. 'frontal_face_*'

        Yields
        ------
        keys: group labels that match the glob pattern
        """
        for key in fnmatch.filter(self.keys(), glob_pattern):
            yield key

    def items_matching(self, glob_pattern):
        r"""
        Yield only items ``(group, LandmarkGroup)`` where the key matches a
        given glob.

        Parameters
        ----------
        glob_pattern : `str`
            A glob pattern e.g. 'frontal_face_*'

        Yields
        ------
        item : ``(group, LandmarkGroup)``
            Tuple of group, LandmarkGroup where the group matches the glob
        """
        for k, v in self.items():
            if fnmatch.fnmatch(k, glob_pattern):
                yield k, v

    def _transform_inplace(self, transform):
        for group in self._landmark_groups.values():
            group.lms._transform_inplace(transform)
        return self

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualizes the landmark manager object using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the landmark managers will have
            the form of plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_landmarks
            visualize_landmarks(self, figure_size=figure_size, style=style,
                                browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        out_string = '{}: n_groups: {}'.format(type(self).__name__,
                                               self.n_groups)
        if self.has_landmarks:
            for label in self:
                out_string += '\n'
                out_string += '({}): {}'.format(label, self[label].__str__())

        return out_string


class LandmarkGroup(MutableMapping, Copyable, Viewable):
    r"""
    An immutable object that holds a :map:`PointCloud` (or a subclass) and
    stores labels for each point. These labels are defined via masks on the
    :map:`PointCloud`. For this reason, the :map:`PointCloud` is considered to
    be immutable.

    The labels to masks must be within an `OrderedDict` so that semantic
    ordering can be maintained.

    Parameters
    ----------
    pointcloud : :map:`PointCloud`
        The pointcloud representing the landmarks.
    labels_to_masks : `ordereddict` {`str` -> `bool ndarray`}
        For each label, the mask that specifies the indices in to the
        pointcloud that belong to the label.
    copy : `bool`, optional
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
        if not isinstance(labels_to_masks, OrderedDict):
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

    @classmethod
    def init_with_all_label(cls, pointcloud, copy=True):
        r"""
        Static constructor to create a :map:`LandmarkGroup` with a single
        default 'all' label that covers all points.

        Parameters
        ----------
        pointcloud : :map:`PointCloud`
            The pointcloud representing the landmarks.
        copy : `boolean`, optional
            If ``True``, a copy of the :map:`PointCloud` is stored on the group.

        Returns
        -------
        lmark_group : :map:`LandmarkGroup`
            Landmark group wrapping the given pointcloud with a single label
            called 'all' that is ``True`` for all points.
        """
        labels_to_masks = OrderedDict(
            [('all', np.ones(pointcloud.n_points, dtype=np.bool))])
        return LandmarkGroup(pointcloud, labels_to_masks, copy=copy)

    @classmethod
    def init_from_indices_mapping(cls, pointcloud, labels_to_indices,
                                  copy=True):
        r"""
        Static constructor to create a :map:`LandmarkGroup` from an ordered
        dictionary that maps a set of indices .

        Parameters
        ----------
        pointcloud : :map:`PointCloud`
            The pointcloud representing the landmarks.
        labels_to_indices : `ordereddict` {`str` -> `int ndarray`}
            For each label, the indices in to the pointcloud that belong to the
            label.
        copy : `boolean`, optional
            If ``True``, a copy of the :map:`PointCloud` is stored on the group.

        Returns
        -------
        lmark_group : :map:`LandmarkGroup`
            Landmark group wrapping the given pointcloud with the given
            semantic labels applied.

        Raises
        ------
        ValueError
            If `dict` passed instead of `OrderedDict`
        ValueError
            If any of the label masks differs in size to the pointcloud.
        ValueError
            If there exists any point in the pointcloud that is not covered
            by a label.
        """
        labels_to_masks = indices_to_masks(labels_to_indices,
                                           pointcloud.n_points)
        return LandmarkGroup(pointcloud, labels_to_masks, copy=copy)

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

        :type: `list` of `str`
        """
        # Convert to list so that we can index immediately, as keys()
        # is a generator in Python 3
        return list(self._labels_to_masks.keys())

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
        """A new landmark group that contains only the certain labels

        Parameters
        ----------
        labels : `str` or `list` of `str`, optional
            Labels that should be kept in the returned landmark group. If
            ``None`` is passed, and if there is only one label on this group,
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
        """A new landmark group that excludes certain labels
        label.

        Parameters
        ----------
        labels : `str` or `list` of `str`
            Labels that should be excluded in the returned landmark group.

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
        # values is a generator in Python 3, so convert to list
        labels_values = list(self._labels_to_masks.values())
        unlabelled_points = np.sum(labels_values, axis=0) == 0
        if np.any(unlabelled_points):
            nonzero = np.nonzero(unlabelled_points)
            raise ValueError(
                'Every point in the landmark pointcloud must be labelled. '
                'Points {0} were unlabelled.'.format(nonzero))

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
        json : ``dict``
            Dictionary conforming to the LJSON v2 specification.
        """
        labels = [{'mask': mask.nonzero()[0].tolist(),
                   'label': label}
                  for label, mask in self._labels_to_masks.items()]

        return {'landmarks': self.lms.tojson(),
                'labels': labels}

    def has_nan_values(self):
        """
        Tests if the LandmarkGroup contains ``nan`` values or
        not. This is particularly useful for annotations with unknown values or
        non-visible landmarks that have been mapped to ``nan`` values.

        Returns
        -------
        has_nan_values : `bool`
            If the LandmarkGroup contains ``nan`` values.
        """
        return self._pointcloud.has_nan_values()

    def _view_2d(self, with_labels=None, without_labels=None, group='group',
                 figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour=None, line_style='-',
                 line_width=1, render_markers=True, marker_style='o',
                 marker_size=5, marker_face_colour=None,
                 marker_edge_colour=None, marker_edge_width=1.,
                 render_numbering=False, numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_legend=True, legend_title='',
                 legend_font_name='sans-serif', legend_font_style='normal',
                 legend_font_size=10, legend_font_weight='normal',
                 legend_marker_scale=None, legend_location=2,
                 legend_bbox_to_anchor=(1.05, 1.), legend_border_axes_pad=None,
                 legend_n_columns=1, legend_horizontal_spacing=None,
                 legend_vertical_spacing=None, legend_border=True,
                 legend_border_padding=None, legend_shadow=False,
                 legend_rounded_corners=False, render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(10, 8)):
        """
        Visualize the landmark group.

        Parameters
        ----------
        with_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        group : `str` or `None`, optional
            The landmark group to be visualized. If ``None`` and there are more
            than one landmark groups, an error is raised.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True``, the x and y axes are flipped.
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``} or
                      ``(3, )`` `ndarray` or ``None``, optional
            The colour of the lines. If ``None``, a different colour will be
            automatically selected for each label.
        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                        ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                        ``1``, ``2``, ``3``, ``4``, ``8``}, optional
            The style of the markers.
        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The face (filling) colour of the markers.
        marker_edge_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The edge colour of the markers.
        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : {``center``, ``right``, ``left``}, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : {``center``, ``top``, ``bottom``,
                                  ``baseline``}, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : {``serif``, ``sans-serif``, ``cursive``,
                             ``fantasy``, ``monospace``}, optional
            The font of the numbers.
        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the numbers.
        numbers_font_weight : {``ultralight``, ``light``, ``normal``,
                               ``regular``, ``book``, ``medium``, ``roman``,
                               ``semibold``, ``demibold``, ``demi``, ``bold``,
                               ``heavy``, ``extra bold``, ``black``}, optional
            The font weight of the numbers.
        numbers_font_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                              or ``(3, )`` `ndarray`, optional
            The font colour of the numbers.
        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : {``serif``, ``sans-serif``, ``cursive``,
                            ``fantasy``, ``monospace``}, optional
            The font of the legend.
        legend_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the legend.
        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : {``ultralight``, ``light``, ``normal``,
                              ``regular``, ``book``, ``medium``, ``roman``,
                              ``semibold``, ``demibold``, ``demi``, ``bold``,
                              ``heavy``, ``extra bold``, ``black``}, optional
            The font weight of the legend.
        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ===
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ===

        legend_bbox_to_anchor : (`float`, `float`), optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                          ``monospace``}, optional
            The font of the axes.
        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : {``ultralight``, ``light``, ``normal``, ``regular``,
                            ``book``, ``medium``, ``roman``, ``semibold``,
                            ``demibold``, ``demi``, ``bold``, ``heavy``,
                            ``extra bold``, ``black``}, optional
            The font weight of the axes.
        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the LandmarkGroup as a percentage of the
            LandmarkGroup's width. If `tuple` or `list`, then it defines the axis
            limits. If ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the LandmarkGroup as a percentage of the
            LandmarkGroup's height. If `tuple` or `list`, then it defines the
            axis limits. If ``None``, then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        """
        from menpo.visualize import LandmarkViewer2d
        if with_labels is not None and without_labels is not None:
            raise ValueError('You may only pass one of `with_labels` or '
                             '`without_labels`.')
        elif with_labels is not None:
            lmark_group = self.with_labels(with_labels)
        elif without_labels is not None:
            lmark_group = self.without_labels(without_labels)
        else:
            lmark_group = self  # Fall through
        landmark_viewer = LandmarkViewer2d(figure_id, new_figure,
                                           group, lmark_group._pointcloud,
                                           lmark_group._labels_to_masks)
        return landmark_viewer.render(
            image_view=image_view, render_lines=render_lines,
            line_colour=line_colour, line_style=line_style,
            line_width=line_width, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
            legend_font_weight=legend_font_weight,
            legend_marker_scale=legend_marker_scale,
            legend_location=legend_location,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_border_axes_pad=legend_border_axes_pad,
            legend_n_columns=legend_n_columns,
            legend_horizontal_spacing=legend_horizontal_spacing,
            legend_vertical_spacing=legend_vertical_spacing,
            legend_border=legend_border,
            legend_border_padding=legend_border_padding,
            legend_shadow=legend_shadow,
            legend_rounded_corners=legend_rounded_corners,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def _view_3d(self, figure_id=None, new_figure=False, **kwargs):
        try:
            from menpo3d.visualize import LandmarkViewer3d
            return LandmarkViewer3d(figure_id, new_figure,
                                    self._pointcloud, self).render(**kwargs)
        except ImportError:
            from menpo.visualize import Menpo3dMissingError
            raise Menpo3dMissingError()

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualizes the landmark group object using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the landmark managers will have
            the form of plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_landmarkgroups
            visualize_landmarkgroups(self, figure_size=figure_size, style=style,
                                     browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        return '{}: n_labels: {}, n_points: {}'.format(
            type(self).__name__, self.n_labels, self.n_landmarks)


def indices_to_masks(labels_to_indices, n_points):
    r"""
    Take a dictionary of labels to indices and convert it to a dictionary
    that maps labels to masks. This dictionary is the correct format for
    constructing a :map:`LandmarkGroup`.

    Parameters
    ----------
    labels_to_indices : `ordereddict` {`str` -> `int ndarray`}
        For each label, the indices in to the pointcloud that belong to the
        label.
    n_points : `int`
        Number of points in the pointcloud that is being masked.
    """
    if not isinstance(labels_to_indices, OrderedDict):
        raise ValueError('Must provide an OrderedDict to maintain the '
                         'semantic meaning of the labels.')

    masks = OrderedDict()
    for label in labels_to_indices:
        indices = labels_to_indices[label]
        mask = np.zeros(n_points, dtype=np.bool)
        mask[indices] = True
        masks[label] = mask
    return masks
