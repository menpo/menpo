from collections import OrderedDict
from functools import wraps
import numpy as np

from menpo.base import name_of_callable
from menpo.landmark.base import LandmarkGroup
from menpo.landmark.exceptions import LabellingError


def connectivity_from_array(array, close_loop=False):
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


def connectivity_from_range(range_tuple, close_loop=False):
    r"""
    Build the connectivity over a range. For example, given ::

        range_array = np.arange(3)

    Generate the connectivity of ::

        [(0, 1), (1, 2), (2, 3)]

    If ``close_loop`` is true, add an extra connection from the last point to
    the first.
    """
    return connectivity_from_array(
        np.arange(*range_tuple), close_loop=close_loop)


def validate_input(pcloud, n_expected_points):
    r"""
    Ensure that the input matches the number of expected points.

    Parameters
    ----------
    pcloud : :map:`PointCloud`
        Input Pointcloud to validate
    n_expected_points : `int`
        Number of expected points

    Raises
    ------
    LabellingError
        If the number of expected points doesn't match the number of given
        points
    """
    n_actual_points = pcloud.n_points
    if n_actual_points != n_expected_points:
        msg = 'Label expects exactly {} ' \
              'points. However, the given landmark group ' \
              'has {} points'.format(n_expected_points, n_actual_points)
        raise LabellingError(msg)


def pcloud_and_lgroup_from_ranges(pointcloud, labels_to_ranges):
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
    labels_to_ranges : `ordereddict` {`str` -> (`int`, `int`, `bool`)}
        Ordered dictionary of string labels to range tuples.

    Returns
    -------
    new_pcloud : :map:`PointCloud`
        New pointcloud with specific connectivity information applied.
    mapping : `ordereddict` {`str` -> `int ndarray`}
        For each label, the indices in to the pointcloud that belong to the
        label.
    """
    from menpo.shape import PointUndirectedGraph

    mapping = OrderedDict()
    all_connectivity = []
    for label, tup in labels_to_ranges.items():
        range_tuple = tup[:-1]
        close_loop = tup[-1]

        connectivity = connectivity_from_range(range_tuple,
                                               close_loop=close_loop)
        all_connectivity.append(connectivity)
        mapping[label] = np.arange(*range_tuple)
    all_connectivity = np.vstack(all_connectivity)

    new_pcloud = PointUndirectedGraph.init_from_edges(pointcloud.points,
                                                      all_connectivity)

    return new_pcloud, mapping


_labeller_docs = r"""
    Parameters
    ----------
    x : :map:`LandmarkGroup` or :map:`PointCloud` or `ndarray`
        The input landmark group, pointcloud or array to label. If a pointcloud
        is passed, then only the connectivity information is propagated to
        the pointcloud (a subclass of :map:`PointCloud` may be returned).
    return_mapping : `bool`, optional
        Only applicable if a :map:`PointCloud` or `ndarray` is passed. Returns
        the mapping dictionary which maps labels to indices into the resulting
        :map:`PointCloud` (which is then used to for building a
        :map:`LandmarkGroup`. This parameter is only provided for internal
        use so that other labellers can piggyback off one another.

    Returns
    -------
    x_labelled : :map:`LandmarkGroup` or :map:`PointCloud`
        If a :map:`LandmarkGroup` was passed, a :map:`LandmarkGroup` is
        returned. This landmark group will contain specific labels and
        these labels may refer to sub-pointclouds with specific connectivity
        information.

        If a :map:`PointCloud` was passed, a :map:`PointCloud` is returned. Only
        the connectivity information is propagated to the pointcloud
        (a subclass of :map:`PointCloud` may be returned).
    mapping_dict : `ordereddict` {`str` -> `int ndarray`}, optional
        Only returned if ``return_mapping==True``. Used for building
        :map:`LandmarkGroup`.

    Raises
    ------
    : :map:`LabellingError`
        If the given landmark group/pointcloud contains less than the
        expected number of points.
"""


def labeller_func(group_label=None):
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
        # Set up the global docs
        labelling_method.__doc__ += _labeller_docs

        @wraps(labelling_method)
        def wrapper(x, return_mapping=False):
            from menpo.shape import PointCloud
            # Accepts LandmarkGroup, PointCloud or ndarray
            if isinstance(x, np.ndarray):
                x = PointCloud(x, copy=False)

            if isinstance(x, PointCloud):
                new_pcloud, mapping = labelling_method(x)
                # This parameter is only provided for internal use so that
                # other labellers can piggyback off one another
                if return_mapping:
                    return new_pcloud, mapping
                else:
                    return new_pcloud
            if isinstance(x, LandmarkGroup):
                new_pcloud, mapping = labelling_method(x.lms)
                return LandmarkGroup.init_from_indices_mapping(new_pcloud, 
                                                               mapping)
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
