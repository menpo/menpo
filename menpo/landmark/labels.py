from collections import OrderedDict
import numpy as np
from menpo.landmark.base import LandmarkGroup
from menpo.landmark.exceptions import LabellingError


def _connectivity_from_range(range_tuple, close_loop=False):
    r"""
    Build the connectivity over a range. For example, given ::

        range_array = np.arange(3)

    Generate the connectivity of ::

        [(0, 1), (1, 2), (2, 3)]

    If ``close_loop`` is true, add an extra connection from the last point to
    the first.
    """
    range_array = np.arange(*range_tuple)
    conn = zip(range_array, range_array[1:])
    if close_loop:
        conn.append((range_array[-1], range_array[0]))
    return np.asarray(conn)


def _mask_from_range(range_tuple, n_points):
    r"""
    Generate a mask over the range. The mask will be true inside the range.
    """
    mask = np.zeros(n_points, dtype=np.bool)
    range_slice = slice(*range_tuple)
    mask[range_slice] = True
    return mask


def _build_labelling_error_msg(group_label, n_expected_points,
                               n_actual_points):
    return '{} mark-up expects exactly {} ' \
           'points. However, the given landmark group ' \
           'has {} points'.format(group_label, n_expected_points,
                                  n_actual_points)


def _validate_input(landmark_group, n_expected_points, group_label):
    r"""
    Ensure that the input matches the number of expected points.

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        Landmark group to validate
    n_expected_points : `int`
        Number of expected points
    group_label : `str`
        Group label for error message

    Raises
    ------
    LabellingError
        If the number of expected points doesn't match the number of given
        points
    """
    n_points = landmark_group.lms.n_points
    if n_points != n_expected_points:
        raise LabellingError(_build_labelling_error_msg(group_label,
                                                        n_expected_points,
                                                        n_points))


def _relabel_group_from_dict(pointcloud, labels_to_ranges):
    """
    Label the given pointcloud according to the given ordered dictionary
    of labels to ranges. This assumes that you can semantically label the group
    by using ranges in to the existing points e.g ::

        labels_to_ranges = {'chin': (0, 17, False)}

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
    from ..shape import PointGraph

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
        PointGraph(pointcloud.points, adjacency_array), masks)

    return new_landmark_group


def imm_58_points(landmark_group):
    """
    Apply the 58 point semantic labels from the
    IMM dataset to the landmarks in the given landmark group.

    The group label will be 'imm_58_points'.

    The semantic labels applied are as follows:

      - chin
      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'imm_58_points'
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 58 points

    References
    -----------
    .. [1] http://www2.imm.dtu.dk/~aam/
    """
    group_label = 'imm_58_points'
    _validate_input(landmark_group, 58, group_label)
    labels = OrderedDict([
        ('chin', (0, 13, False)),
        ('leye', (13, 21, True)),
        ('reye', (21, 29, True)),
        ('leyebrow', (29, 34, False)),
        ('reyebrow', (34, 39, False)),
        ('mouth', (39, 47, True)),
        ('nose', (47, 58, False))
    ])
    return group_label, _relabel_group_from_dict(landmark_group.lms, labels)


def ibug_68_points(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmark group.

    The group label will be 'ibug_68_points'.

    The semantic labels applied are as follows:

      - chin
      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_68_points'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_68_points'
    _validate_input(landmark_group, 68, group_label)
    labels = OrderedDict([
        ('chin', (0, 17, False)),
        ('leye', (36, 42, True)),
        ('reye', (42, 48, True)),
        ('leyebrow', (17, 22, False)),
        ('reyebrow', (22, 27, False)),
        ('mouth', (48, 68, True)),
        ('nose', (27, 36, False))
    ])
    return group_label, _relabel_group_from_dict(landmark_group.lms, labels)


def ibug_68_trimesh(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmarks in
    the given landmark group.

    The group label will be 'ibug_68_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_68_trimesh'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import TriMesh

    group_label = 'ibug_68_trimesh'
    n_expected_points = 68
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group_label)

    tri_list = np.array([[47, 29, 28], [44, 43, 23], [38, 20, 21], [47, 28,42],
                        [49, 61, 60], [40, 41, 37], [37, 19, 20], [28, 40, 39],
                        [38, 21, 39], [36,  1, 	0], [48, 59,  4], [49, 60, 48],
                        [67, 59, 60], [13, 53, 14], [61, 51, 62], [57,  8,  7],
                        [52, 51, 33], [61, 67, 60], [52, 63, 51], [66, 56, 57],
                        [35, 30, 29], [53, 52, 35], [37, 36, 17], [18, 37, 17],
                        [37, 38, 40], [38, 37, 20], [19, 37, 18], [38, 39, 40],
                        [28, 29, 40], [41, 36, 37], [27, 39, 21], [41, 31,  1],
                        [30, 32, 31], [33, 51, 50], [33, 30, 34], [31, 40, 29],
                        [36,  0, 17], [31,  2,  1], [31, 41, 40], [ 1, 36, 41],
                        [31, 49,  2], [ 2, 49,  3], [60, 59, 48], [ 3, 49, 48],
                        [31, 32, 50], [48,  4,  3], [59,  5,  4], [58, 67, 66],
                        [ 5, 59, 58], [58, 59, 67], [ 7,  6, 58], [66, 57, 58],
                        [13, 54, 53], [ 7, 58, 57], [ 6,  5, 58], [50, 61, 49],
                        [62, 67, 61], [31, 50, 49], [32, 33, 50], [30, 33, 32],
                        [34, 52, 33], [35, 52, 34], [53, 63, 52], [62, 63, 65],
                        [62, 51, 63], [66, 65, 56], [63, 53, 64], [62, 66, 67],
                        [62, 65, 66], [57, 56,  9], [65, 63, 64], [ 8, 57,  9],
                        [ 9, 56, 10], [10, 56, 11], [11, 56, 55], [11, 55, 12],
                        [56, 65, 55], [55, 64, 54], [55, 65, 64], [55, 54, 12],
                        [64, 53, 54], [12, 54, 13], [45, 46, 44], [35, 34, 30],
                        [14, 53, 35], [15, 46, 45], [27, 28, 39], [27, 42, 28],
                        [35, 29, 47], [30, 31, 29], [15, 35, 46], [15, 14, 35],
                        [43, 22, 23], [27, 21, 22], [24, 44, 23], [44, 47, 43],
                        [43, 47, 42], [46, 35, 47], [26, 45, 44], [46, 47, 44],
                        [25, 44, 24], [25, 26, 44], [16, 15, 45], [16, 45, 26],
                        [22, 42, 43], [50, 51, 61], [27, 22, 42]])
    new_landmark_group = LandmarkGroup(
        TriMesh(landmark_group.lms.points, tri_list, copy=False),
        OrderedDict([('tri', np.ones(n_points, dtype=np.bool))]))

    return group_label, new_landmark_group


def ibug_68_closed_mouth(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmarks in
    the given landmark group - but ignore the 3 points that are coincident for
    a closed mouth. Therefore, there only 65 points are returned.

    The group label will be 'ibug_68_closed_mouth'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_68_closed_mouth'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 65 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_68_closed_mouth'
    _validate_input(landmark_group, 68, group_label)
    labels = OrderedDict([
        ('chin', (0, 17, False)),
        ('leye', (36, 42, True)),
        ('reye', (42, 48, True)),
        ('leyebrow', (17, 22, False)),
        ('reyebrow', (22, 27, False)),
        ('mouth', (48, 65, True)),  # Ignore 3 coincident points (last 3)
        ('nose', (27, 36, False))
    ])

    new_landmarks = landmark_group.lms.copy()
    # Ignore 3 coincident points
    new_landmarks.points = new_landmarks.points[:-3]
    lg = _relabel_group_from_dict(new_landmarks, labels)
    return group_label, lg


def ibug_66_points(landmark_group):
    """
    Apply the ibug's "standard" 66 point semantic labels (based on the
    original semantic labels of multiPIE but ignoring the 2 points
    describing the inner mouth corners) to the landmark group.

    The group label will be 'ibug_66_points'.

    The semantic labels applied are as follows:

      - chin
      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_66_points'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_66_points'
    _validate_input(landmark_group, 68, group_label)
    labels = OrderedDict([
        ('chin', (0, 17, False)),
        ('leye', (36, 42, True)),
        ('reye', (42, 48, True)),
        ('leyebrow', (17, 22, False)),
        ('reyebrow', (22, 27, False)),
        ('mouth', (48, 66, True)),
        ('nose', (27, 36, False))
    ])

    new_landmarks = landmark_group.lms.copy()
    # Ignore the two inner mouth points
    ind = np.hstack((np.arange(60),
                     np.arange(61, 64),
                     np.arange(65, 68)))
    new_landmarks.points = new_landmarks.points[ind]
    lg = _relabel_group_from_dict(new_landmarks, labels)
    return group_label, lg


def ibug_51_points(landmark_group):
    """
    Apply the ibug's "standard" 51 point semantic labels (based on the
    original semantic labels of multiPIE but removing the annotations
    corresponding to the chin region) to the landmark group.

    The group label will be 'ibug_51_points'.

    The semantic labels applied are as follows:

      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_51_points'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_51_points'
    _validate_input(landmark_group, 68, group_label)
    labels = OrderedDict([
        ('leye', (19, 25, True)),
        ('reye', (25, 31, True)),
        ('leyebrow', (0, 5, False)),
        ('reyebrow', (5, 10, False)),
        ('mouth', (31, 51, True)),
        ('nose', (10, 19, False))
    ])

    new_landmarks = landmark_group.lms.copy()
    # Ignore the chin region
    ind = np.arange(17, 68)
    new_landmarks.points = new_landmarks.points[ind]
    lg = _relabel_group_from_dict(new_landmarks, labels)
    return group_label, lg


def ibug_49_points(landmark_group):
    """
    Apply the ibug's "standard" 49 point semantic labels (based on the
    original semantic labels of multiPIE but removing the annotations
    corresponding to the chin region and the 2 describing the inner mouth
    corners) to the landmark group.

    The group label will be 'ibug_49_points'.

    The semantic labels applied are as follows:

      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_49_points'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_49_points'
    _validate_input(landmark_group, 68, group_label)
    labels = OrderedDict([
        ('leye', (19, 25, True)),
        ('reye', (25, 31, True)),
        ('leyebrow', (0, 5, False)),
        ('reyebrow', (5, 10, False)),
        ('mouth', (31, 49, True)),
        ('nose', (10, 19, False))
    ])

    new_landmarks = landmark_group.lms.copy()
    # Ignore the chin region and two inner mouth points
    ind = np.hstack((np.arange(17, 60),
                     np.arange(61, 64),
                     np.arange(65, 68)))
    new_landmarks.points = new_landmarks.points[ind]
    lg = _relabel_group_from_dict(new_landmarks, labels)
    return group_label, lg


def _build_upper_eyelid():
    top_indices = np.arange(0, 7)
    middle_indices = np.arange(12, 17)
    upper_eyelid_indices = np.hstack((top_indices, middle_indices))

    upper_eyelid_connectivity = zip(top_indices, top_indices[1:])
    upper_eyelid_connectivity += [(0, 12)]
    upper_eyelid_connectivity += zip(middle_indices, middle_indices[1:])
    upper_eyelid_connectivity += [(16, 6)]

    return upper_eyelid_indices, upper_eyelid_connectivity


def ibug_open_eye_points(landmark_group):
    """
    Apply the ibug's "standard" open eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be 'ibug_open_eye_points'.

    The semantic labels applied are as follows:

      - upper eyelid
      - lower eyelid
      - iris
      - pupil
      - sclera

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_open_eye_points'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 38 points
    """
    from ..shape import PointGraph

    group_label = 'ibug_open_eye_points'
    n_expected_points = 38
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group_label)

    upper_el_indices, upper_el_connectivity = _build_upper_eyelid()

    iris_range = (22, 30)
    pupil_range = (30, 38)
    sclera_top = np.arange(12, 17)
    sclera_bottom = np.arange(17, 22)
    sclera_indices = np.hstack((0, sclera_top, 6, sclera_bottom))
    lower_el_top = np.arange(17, 22)
    lower_el_bottom = np.arange(7, 12)
    lower_el_indices = np.hstack((6, lower_el_top, 0, lower_el_bottom))

    iris_connectivity = _connectivity_from_range(iris_range, close_loop=True)
    pupil_connectivity = _connectivity_from_range(pupil_range, close_loop=True)

    sclera_connectivity = zip(sclera_top, sclera_top[1:])
    sclera_connectivity += [(0, 21)]
    sclera_connectivity += zip(sclera_bottom, sclera_bottom[1:])
    sclera_connectivity += [(6, 17)]

    lower_el_connectivity = zip(lower_el_top, lower_el_top[1:])
    lower_el_connectivity += [(6, 7)]
    lower_el_connectivity += zip(lower_el_bottom, lower_el_bottom[1:])
    lower_el_connectivity += [(11, 0)]

    total_connectivity = np.asarray(upper_el_connectivity +
                                    lower_el_connectivity +
                                    iris_connectivity.tolist() +
                                    pupil_connectivity.tolist() +
                                    sclera_connectivity)
    new_landmark_group = LandmarkGroup(
        PointGraph(landmark_group.lms.points, total_connectivity),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['upper_eyelid'] = upper_el_indices
    new_landmark_group['lower_eyelid'] = lower_el_indices
    new_landmark_group['pupil'] = np.arange(*pupil_range)
    new_landmark_group['iris'] = np.arange(*iris_range)
    new_landmark_group['sclera'] = sclera_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group_label, new_landmark_group


def ibug_close_eye_points(landmark_group):
    """
    Apply the ibug's "standard" close eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be 'ibug_close_eye_points'.

    The semantic labels applied are as follows:

      - upper eyelid
      - lower eyelid

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_close_eye_points'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 17 points
    """
    from ..shape import PointGraph

    group_label = 'ibug_close_eye_points'
    n_expected_points = 17
    n_points = landmark_group.lms.n_points
    _validate_input(landmark_group, n_expected_points, group_label)

    upper_indices, upper_connectivity = _build_upper_eyelid()

    middle_indices = np.arange(12, 17)
    bottom_indices = np.arange(6, 12)
    lower_indices = np.hstack((bottom_indices, 0, middle_indices))
    lower_connectivity = zip(bottom_indices, bottom_indices[1:])
    lower_connectivity += [(0, 12)]
    lower_connectivity += zip(middle_indices, middle_indices[1:])
    lower_connectivity += [(11, 0)]

    total_connectivity = np.asarray(upper_connectivity + lower_connectivity)
    new_landmark_group = LandmarkGroup(
        PointGraph(landmark_group.lms.points, total_connectivity),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['upper_eyelid'] = upper_indices
    new_landmark_group['lower_eyelid'] = lower_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group_label, new_landmark_group


def ibug_open_eye_trimesh(landmark_group):
    """
    Apply the ibug's "standard" open eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be 'ibug_open_eye_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_open_eye_trimesh'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 38 points
    """
    from menpo.shape import TriMesh

    group_label = 'ibug_open_eye_trimesh'
    n_expected_points = 38
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group_label)

    tri_list = np.array([[29, 36, 28], [22, 13, 23], [12,  1,  2],
                         [29, 30, 37], [13,  3, 14], [13, 12,  2],
                         [19,  8,  9], [25, 33, 24], [36, 37, 33],
                         [24, 32, 31], [33, 37, 31], [35, 34, 27],
                         [35, 36, 33], [ 3, 13,  2], [14, 24, 23],
                         [33, 32, 24], [15, 25, 14], [25, 26, 34],
                         [22, 30, 29], [31, 37, 30], [24, 31, 23],
                         [32, 33, 31], [22, 12, 13], [ 0,  1, 12],
                         [14, 23, 13], [31, 30, 23], [28, 19, 20],
                         [21, 11,  0], [12, 21,  0], [20, 11, 21],
                         [20, 10, 11], [21, 29, 20], [21, 12, 22],
                         [30, 22, 23], [29, 21, 22], [27, 19, 28],
                         [29, 37, 36], [29, 28, 20], [36, 35, 28],
                         [20, 19, 10], [10, 19,  9], [28, 35, 27],
                         [19, 19,  8], [17, 16,  6], [18,  7,  8],
                         [25, 34, 33], [18, 27, 17], [18, 19, 27],
                         [18, 17,  7], [27, 26, 17], [17,  6,  7],
                         [14, 25, 24], [34, 35, 33], [17, 26, 16],
                         [27, 34, 26], [ 3, 15, 14], [15, 26, 25],
                         [ 4, 15,  3], [16, 26, 15], [16,  4,  5],
                         [16, 15,  4], [16,  5,  6], [8, 18, 19]])

    new_landmark_group = LandmarkGroup(
        TriMesh(landmark_group.lms.points, tri_list, copy=False),
        OrderedDict([('tri', np.ones(n_points, dtype=np.bool))]))

    return group_label, new_landmark_group


def ibug_close_eye_trimesh(landmark_group):
    """
    Apply the ibug's "standard" close eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be 'ibug_close_eye_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_close_eye_trimesh'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 38 points
    """
    from menpo.shape import TriMesh

    group_label = 'ibug_close_eye_trimesh'
    n_expected_points = 17
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group_label)

    tri_list = np.array([[10, 11, 13], [ 3, 13,  2], [ 4, 14,  3],
                         [15,  5, 16], [12, 11,  0], [13, 14, 10],
                         [13, 12,  2], [14, 13,  3], [ 0,  1, 12],
                         [ 2, 12,  1], [13, 11, 12], [ 9, 10, 14],
                         [15,  9, 14], [ 7,  8, 15], [ 5,  6, 16],
                         [15, 14,  4], [ 7, 15, 16], [ 8,  9, 15],
                         [15,  4,  5], [16,  6,  7]])

    new_landmark_group = LandmarkGroup(
        TriMesh(landmark_group.lms.points, tri_list, copy=False),
        OrderedDict([('tri', np.ones(n_points, dtype=np.bool))]))

    return group_label, new_landmark_group


def ibug_tongue(landmark_group):
    """
    Apply the ibug's "standard" tongue semantic labels to the landmarks in the
    given landmark group.

    The group label will be 'ibug_tongue'.

    The semantic labels applied are as follows:

      - outline
      - bisector

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group_label : `str`
        The group label: 'ibug_tongue'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 19 points
    """
    group_label = 'ibug_tongue'
    _validate_input(landmark_group, 19, group_label)
    labels = OrderedDict([
        ('outline', (0, 13, False)),
        ('bisector', (13, 19, False))
    ])
    return group_label, _relabel_group_from_dict(landmark_group.lms, labels)


def labeller(landmarkable, group_label, label_func):
    """
    Takes a landmarkable object and a group label indicating which
    set of landmarks should have semantic meaning attached to them.
    The labelling function will add a new landmark group to each object that
    have been semantically annotated.

    Parameters
    ----------
    landmarkable: :class:`menpo.landmark.base.Landmarkable`
        Landmarkable object
    group_label: string
        The group label of the landmark group to apply semantic labels to.
    label_func: func
        A labelling function taken from this module. `func` should take a
        :class:`menpo.landmark.base.LandmarkGroup` and
        return a tuple of (group label, new LandmarkGroup with semantic labels
        applied.)

    Returns
    -------
    landmarkable : :class:`menpo.landmark.base.Landmarkable`
        landmarkable with label (this is just for convenience,
        the object will actually be modified in place)
    """
    group_label, group = label_func(landmarkable.landmarks[group_label])
    landmarkable.landmarks[group_label] = group
    return landmarkable
