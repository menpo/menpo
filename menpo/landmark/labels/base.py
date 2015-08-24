from collections import OrderedDict
from functools import wraps
import numpy as np

from menpo.base import name_of_callable
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


def _validate_input(pcloud, n_expected_points, group):
    r"""
    Ensure that the input matches the number of expected points.

    Parameters
    ----------
    pcloud : :map:`PointCloud`
        Input Pointcloud to validate
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
    n_actual_points = pcloud.n_points
    if n_actual_points != n_expected_points:
        msg = '{} mark-up expects exactly {} ' \
              'points. However, the given landmark group ' \
              'has {} points'.format(group, n_expected_points, n_actual_points)
        raise LabellingError(msg)


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


def _labeller(group_label=None):
    r"""
    Decorator for labelling functions. Labelling functions should return
    a template pointcloud and a mapping dictionary (from labels to indices).
    This decorator then takes that output and returns the correctly labelled
    object depending on if a landmark group, pointcloud or numpy array is
    passed. See the docstrings of the labelling methods where this
    has been made clear.

    Note that we duck type the group label (usually just the name of the
    labelling function) on to the function itself for the labeller method
    below.
    """
    def decorator(labelling_method):
        # Shadowing parent scope variables inside a nested function
        # kills the scope of the parent variable, so we need a unique alias
        # for the group name
        gl = (group_label if group_label is not None
              else name_of_callable(labelling_method))
        # Duck type group label onto method itself
        labelling_method.group_label = gl

        @wraps(labelling_method)
        def wrapper(x):
            from menpo.shape import PointCloud
            # Accepts LandmarkGroup, PointCloud or ndarray
            if isinstance(x, np.ndarray):
                x = PointCloud(x, copy=False)

            # Call the actual labelling method to get the template
            # and dictionary mapping labels to indices
            template, mapping = labelling_method()
            n_expected_points = template.n_points

            if isinstance(x, PointCloud):
                _validate_input(x, n_expected_points, gl)
                return template.from_vector(x.as_vector())
            if isinstance(x, LandmarkGroup):
                _validate_input(x.lms, n_expected_points, gl)
                return LandmarkGroup.init_from_indices_mapping(
                    template.from_vector(x.lms.as_vector()), mapping)
        return wrapper
    return decorator


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
    new_group = label_func(landmarkable.landmarks[group])
    landmarkable.landmarks[label_func.group_label] = new_group
    return landmarkable
