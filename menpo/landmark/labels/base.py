from collections import OrderedDict
import numpy as np

from menpo.landmark.base import LandmarkGroup
from menpo.landmark.exceptions import LabellingError


def _connectivity_from_array(array, close_loop=False):
    r"""
    Build the connectivity over a given array. For example, given ::

        array = [(0, 3, 1, 2 )]

    Generate the connectivity of ::

        [(0, 3), (3, 1), (1, 2)]

    If ``close_loop`` is true, add an extra connection from the last point to
    the first.
    """
    # zip is a generator - need a list in this case
    conn = list(zip(array, array[1:]))
    if close_loop:
        conn.append((array[-1], array[0]))
    return np.asarray(conn)


def _connectivity_from_range(range_tuple, close_loop=False):
    r"""
    Build the connectivity over a range. For example, given ::

        range_array = np.arange(3)

    Generate the connectivity of ::

        [(0, 1), (1, 2), (2, 3)]

    If ``close_loop`` is true, add an extra connection from the last point to
    the first.
    """
    return _connectivity_from_array(
        np.arange(*range_tuple), close_loop=close_loop)


def _mask_from_range(range_tuple, n_points):
    r"""
    Generate a mask over the range. The mask will be true inside the range.
    """
    mask = np.zeros(n_points, dtype=np.bool)
    range_slice = slice(*range_tuple)
    mask[range_slice] = True
    return mask


def _build_labelling_error_msg(group, n_expected_points,
                               n_actual_points):
    return '{} mark-up expects exactly {} ' \
           'points. However, the given landmark group ' \
           'has {} points'.format(group, n_expected_points,
                                  n_actual_points)


def _validate_input(landmark_group, n_expected_points, group):
    r"""
    Ensure that the input matches the number of expected points.

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        Landmark group to validate
    n_expected_points : `int`
        Number of expected points
    group : `str`
        Group label for error message

    Raises
    ------
    LabellingError
        If the number of expected points doesn't match the number of given
        points
    """
    n_points = landmark_group.lms.n_points
    if n_points != n_expected_points:
        raise LabellingError(_build_labelling_error_msg(group,
                                                        n_expected_points,
                                                        n_points))


def _relabel_group_from_dict(pointcloud, labels_to_ranges):
    """
    Label the given pointcloud according to the given ordered dictionary
    of labels to ranges. This assumes that you can semantically label the group
    by using ranges in to the existing points e.g ::

        labels_to_ranges = {'jaw': (0, 17, False)}

    The third element of the range tuple is whether the range is a closed loop
    or not. For example, for an eye landmark this would be ``True``, as you
    do want to create a closed loop for an eye.

    Parameters
    ----------
    pointcloud : :map:`PointCloud`
        The pointcloud to apply semantic labels to.
    labels_to_ranges : `OrderedDict`
        Ordered dictionary of string labels to range tuples.

    Returns
    -------
    landmark_group: :map:`LandmarkGroup`
        New landmark group

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given pointcloud contains less than ``n_expected_points``
        points.
    """
    from menpo.shape import PointUndirectedGraph

    n_points = pointcloud.n_points
    masks = OrderedDict()
    adjacency_lists = []
    for label, tup in labels_to_ranges.items():
        range_tuple = tup[:-1]
        close_loop = tup[-1]
        adjacency_lists.append(_connectivity_from_range(
            range_tuple, close_loop=close_loop))
        masks[label] = _mask_from_range(range_tuple, n_points)
    adjacency_array = np.vstack(adjacency_lists)

    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph.init_from_edges(pointcloud.points,
                                             adjacency_array), masks)

    return new_landmark_group


def labeller(landmarkable, group, label_func):
    """
    Re-label an existing landmark group on a :map:`Landmarkable` object with a
    new label set.

    Parameters
    ----------
    landmarkable: :map:`Landmarkable`
        :map:`Landmarkable` that will have it's :map:`LandmarkManager`
        augmented with a new :map:`LandmarkGroup`
    group: `str`
        The group label of the existing landmark group that should be
        re-labelled. A copy of this group will be attached to it's landmark
        manager with new labels. The group label of this new group and the
        labels it will have is determined by ``label_func``
    label_func: `func`  -> `(str, LandmarkGroup)`
        A labelling function taken from this module, Takes as input a
        :map:`LandmarkGroup` and returns a tuple of
        (new group label, new LandmarkGroup with semantic labels applied).

    Returns
    -------
    landmarkable : :map:`Landmarkable`
        Augmented ``landmarkable`` (this is just for convenience,
        the object will actually be modified in place)
    """
    new_group, lmark_group = label_func(landmarkable.landmarks[group])
    landmarkable.landmarks[new_group] = lmark_group
    return landmarkable
