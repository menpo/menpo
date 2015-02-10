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
    conn = zip(array, array[1:])
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
        PointUndirectedGraph(pointcloud.points, adjacency_array), masks)

    return new_landmark_group


def ibug_face_68(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmark group.

    The group label will be ``ibug_face_68``.

    The semantic labels applied are as follows:

      - jaw
      - left_eyebrow
      - right_eyebrow
      - nose
      - left_eye
      - right_eye
      - mouth

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_face_68``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_face_68'
    n_points = 68
    _validate_input(landmark_group, 68, group)

    jaw_indices = np.arange(0, 17)
    lbrow_indices = np.arange(17, 22)
    rbrow_indices = np.arange(22, 27)
    upper_nose_indices = np.arange(27, 31)
    lower_nose_indices = np.arange(31, 36)
    leye_indices = np.arange(36, 42)
    reye_indices = np.arange(42, 48)
    outer_mouth_indices = np.arange(48, 60)
    inner_mouth_indices = np.arange(60, 68)

    jaw_connectivity = _connectivity_from_array(jaw_indices)
    lbrow_connectivity = _connectivity_from_array(lbrow_indices)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices)
    nose_connectivity = np.vstack([
        _connectivity_from_array(upper_nose_indices),
        _connectivity_from_array(lower_nose_indices)])
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    mouth_connectivity = np.vstack([
        _connectivity_from_array(outer_mouth_indices, close_loop=True),
        _connectivity_from_array(inner_mouth_indices, close_loop=True)])

    total_conn = np.vstack([
        jaw_connectivity, lbrow_connectivity, rbrow_connectivity,
        nose_connectivity, leye_connectivity, reye_connectivity,
        mouth_connectivity
    ])

    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points, total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['jaw'] = jaw_indices
    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['nose'] = np.hstack((upper_nose_indices,
                                            lower_nose_indices))
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['mouth'] = np.hstack((outer_mouth_indices,
                                             inner_mouth_indices))

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def ibug_face_66(landmark_group):
    """
    Apply the ibug's "standard" 66 point semantic labels (based on the
    original semantic labels of multiPIE but ignoring the 2 points
    describing the inner mouth corners) to the landmark group.

    The group label will be ``ibug_face_66``.

    The semantic labels applied are as follows:

      - jaw
      - left_eyebrow
      - right_eyebrow
      - nose
      - left_eye
      - right_eye
      - mouth

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_face_66``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_face_66'
    n_points = 66
    _validate_input(landmark_group, 68, group)

    jaw_indices = np.arange(0, 17)
    lbrow_indices = np.arange(17, 22)
    rbrow_indices = np.arange(22, 27)
    upper_nose_indices = np.arange(27, 31)
    lower_nose_indices = np.arange(31, 36)
    leye_indices = np.arange(36, 42)
    reye_indices = np.arange(42, 48)
    outer_mouth_indices = np.arange(48, 60)
    inner_mouth_indices = np.hstack((48, np.arange(60, 63),
                                     54, np.arange(63, 66)))

    jaw_connectivity = _connectivity_from_array(jaw_indices)
    lbrow_connectivity = _connectivity_from_array(lbrow_indices)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices)
    nose_connectivity = np.vstack([
        _connectivity_from_array(upper_nose_indices),
        _connectivity_from_array(lower_nose_indices)])
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    mouth_connectivity = np.vstack([
        _connectivity_from_array(outer_mouth_indices, close_loop=True),
        _connectivity_from_array(inner_mouth_indices, close_loop=True)])

    total_conn = np.vstack([
        jaw_connectivity, lbrow_connectivity, rbrow_connectivity,
        nose_connectivity, leye_connectivity, reye_connectivity,
        mouth_connectivity])

    # Ignore the two inner mouth points
    ind = np.hstack((np.arange(60), np.arange(61, 64), np.arange(65, 68)))
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind],
                             total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['jaw'] = jaw_indices
    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['nose'] = np.hstack([upper_nose_indices,
                                            lower_nose_indices])
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['mouth'] = np.hstack([outer_mouth_indices,
                                             inner_mouth_indices])

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def ibug_face_51(landmark_group):
    """
    Apply the ibug's "standard" 51 point semantic labels (based on the
    original semantic labels of multiPIE but removing the annotations
    corresponding to the jaw region) to the landmark group.

    The group label will be ``ibug_face_51``.

    The semantic labels applied are as follows:

      - left_eyebrow
      - right_eyebrow
      - nose
      - left_eye
      - right_eye
      - mouth

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_face_51``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_face_51'
    n_points = 51
    _validate_input(landmark_group, 68, group)

    lbrow_indices = np.arange(0, 5)
    rbrow_indices = np.arange(5, 10)
    upper_nose_indices = np.arange(10, 14)
    lower_nose_indices = np.arange(14, 19)
    leye_indices = np.arange(19, 25)
    reye_indices = np.arange(25, 31)
    outer_mouth_indices = np.arange(31, 43)
    inner_mouth_indices = np.arange(43, 51)

    lbrow_connectivity = _connectivity_from_array(lbrow_indices)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices)
    nose_connectivity = np.vstack([
        _connectivity_from_array(upper_nose_indices),
        _connectivity_from_array(lower_nose_indices)])
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    mouth_connectivity = np.vstack([
        _connectivity_from_array(outer_mouth_indices, close_loop=True),
        _connectivity_from_array(inner_mouth_indices, close_loop=True)])

    total_conn = np.vstack([
        lbrow_connectivity, rbrow_connectivity, nose_connectivity,
        leye_connectivity, reye_connectivity, mouth_connectivity])

    # Ignore the two inner mouth points
    ind = np.arange(17, 68)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind],
                             total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['nose'] = np.hstack([upper_nose_indices,
                                            lower_nose_indices])
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['mouth'] = np.hstack([outer_mouth_indices,
                                             inner_mouth_indices])

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def ibug_face_49(landmark_group):
    """
    Apply the ibug's "standard" 49 point semantic labels (based on the
    original semantic labels of multiPIE but removing the annotations
    corresponding to the jaw region and the 2 describing the inner mouth
    corners) to the landmark group.

    The group label will be ``ibug_face_49``.

    The semantic labels applied are as follows:

      - left_eyebrow
      - right_eyebrow
      - nose
      - left_eye
      - right_eye
      - mouth

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_face_49``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_face_49'
    n_points = 49
    _validate_input(landmark_group, 68, group)

    lbrow_indices = np.arange(0, 5)
    rbrow_indices = np.arange(5, 10)
    upper_nose_indices = np.arange(10, 14)
    lower_nose_indices = np.arange(14, 19)
    leye_indices = np.arange(19, 25)
    reye_indices = np.arange(25, 31)
    outer_mouth_indices = np.arange(31, 43)
    inner_mouth_indices = np.hstack((31, np.arange(43, 46),
                                     37, np.arange(46, 49)))

    lbrow_connectivity = _connectivity_from_array(lbrow_indices)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices)
    nose_connectivity = np.vstack([
        _connectivity_from_array(upper_nose_indices),
        _connectivity_from_array(lower_nose_indices)])
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    mouth_connectivity = np.vstack([
        _connectivity_from_array(outer_mouth_indices, close_loop=True),
        _connectivity_from_array(inner_mouth_indices, close_loop=True)])

    total_conn = np.vstack([
        lbrow_connectivity, rbrow_connectivity, nose_connectivity,
        leye_connectivity, reye_connectivity, mouth_connectivity])

    # Ignore the two inner mouth points
    ind = np.hstack((np.arange(17, 60), np.arange(61, 64), np.arange(65, 68)))
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind],
                             total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['nose'] = np.hstack([upper_nose_indices,
                                            lower_nose_indices])
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['mouth'] = np.hstack([outer_mouth_indices,
                                             inner_mouth_indices])

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def ibug_face_68_trimesh(landmark_group):
    """
    Apply the ibug's "standard" 68 point triangulation to the landmarks in
    the given landmark group.

    The group label will be ``ibug_face_68_trimesh``.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_face_68_trimesh``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import TriMesh

    group = 'ibug_face_68_trimesh'
    n_expected_points = 68
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

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

    return group, new_landmark_group


def ibug_face_65_closed_mouth(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmarks in
    the given landmark group - but ignore the 3 points that are coincident for
    a closed mouth. Therefore, there only 65 points are returned.

    The group label will be ``ibug_face_65_closed_mouth``.

    The semantic labels applied are as follows:

      - jaw
      - left_eyebrow
      - right_eyebrow
      - nose
      - left_eye
      - right_eye
      - mouth

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_face_65_closed_mouth``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_face_65_closed_mouth'
    n_points = 65
    _validate_input(landmark_group, 68, group)

    jaw_indices = np.arange(0, 17)
    lbrow_indices = np.arange(17, 22)
    rbrow_indices = np.arange(22, 27)
    upper_nose_indices = np.arange(27, 31)
    lower_nose_indices = np.arange(31, 36)
    leye_indices = np.arange(36, 42)
    reye_indices = np.arange(42, 48)
    outer_mouth_indices = np.arange(48, 60)
    inner_mouth_indices = np.arange(60, 65)

    jaw_connectivity = _connectivity_from_array(jaw_indices)
    lbrow_connectivity = _connectivity_from_array(lbrow_indices)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices)
    nose_connectivity = np.vstack([
        _connectivity_from_array(upper_nose_indices),
        _connectivity_from_array(lower_nose_indices)])
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    mouth_connectivity = np.vstack([
        _connectivity_from_array(outer_mouth_indices, close_loop=True),
        _connectivity_from_array(inner_mouth_indices)])

    total_conn = np.vstack([
        jaw_connectivity, lbrow_connectivity, rbrow_connectivity,
        nose_connectivity, leye_connectivity, reye_connectivity,
        mouth_connectivity])

    # Ignore the two inner mouth points
    ind = np.arange(65)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind],
                             total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['jaw'] = jaw_indices
    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['nose'] = np.hstack([upper_nose_indices,
                                            lower_nose_indices])
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['mouth'] = np.hstack([outer_mouth_indices,
                                             inner_mouth_indices])
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def imm_face(landmark_group):
    """
    Apply the 58 point semantic labels from the IMM dataset to the
    landmarks in the given landmark group.

    The group label will be ``imm_face``.

    The semantic labels applied are as follows:

      - jaw
      - left_eye
      - right_eye
      - left_eyebrow
      - right_eyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``imm_face``
    landmark_group : :map:`LandmarkGroup`
        New landmark group

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 58 points

    References
    ----------
    .. [1] http://www2.imm.dtu.dk/~aam/
    """
    group = 'imm_face'
    _validate_input(landmark_group, 58, group)
    labels = OrderedDict([
        ('jaw', (0, 13, False)),
        ('left_eye', (13, 21, True)),
        ('right_eye', (21, 29, True)),
        ('left _eyebrow', (29, 34, False)),
        ('right_eyebrow', (34, 39, False)),
        ('mouth', (39, 47, True)),
        ('nose', (47, 58, False))
    ])
    return group, _relabel_group_from_dict(landmark_group.lms, labels)


def lfpw_face(landmark_group):
    """
    Apply the 29 point semantic labels from the LFPW dataset to the
    landmarks in the given landmark group.

    The group label will be ``lfpw_face``.

    The semantic labels applied are as follows:

      - chin
      - left_eye
      - right_eye
      - left_eyebrow
      - right_eyebrow
      - mouth
      - nose

    Parameters
    ----------
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``lfpw_face``
    landmark_group: :map:`LandmarkGroup`
        New landmark group

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 29 points

    References
    ----------
    .. [1] http://homes.cs.washington.edu/~neeraj/databases/lfpw/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'lfpw_face'
    n_points = 29
    _validate_input(landmark_group, 29, group)

    chin_indices = np.array([28])
    outer_leye_indices = np.array([8, 12, 10, 13])
    pupil_leye_indices = np.array([16])
    outer_reye_indices = np.array([11, 14, 9, 15])
    pupil_reye_indices = np.array([17])
    lbrow_indices = np.array([0, 4, 2, 5])
    rbrow_indices = np.array([3, 6, 1, 7])
    outer_mouth_indices = np.array([22, 24, 23, 27])
    inner_mouth_indices = np.array([22, 25, 23, 26])
    nose_indices = np.array([18, 20, 19, 21])

    # TODO: Not sure this makes a lot of sense...
    chin_connectivity = _connectivity_from_array(chin_indices, close_loop=True)
    leye_connectivity = _connectivity_from_array(outer_leye_indices,
                                                 close_loop=True)
    reye_connectivity = _connectivity_from_array(outer_reye_indices,
                                                 close_loop=True)
    lbrow_connectivity = _connectivity_from_array(lbrow_indices,
                                                  close_loop=True)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices,
                                                  close_loop=True)
    mouth_connectivity = np.vstack(
        (_connectivity_from_array(outer_mouth_indices, close_loop=True),
         _connectivity_from_array(inner_mouth_indices)))
    nose_connectivity = _connectivity_from_array(nose_indices, close_loop=True)

    total_conn = np.vstack(
        (chin_connectivity, leye_connectivity, reye_connectivity,
         lbrow_connectivity, rbrow_connectivity, mouth_connectivity,
         nose_connectivity))

    # Ignore the two inner mouth points
    ind = np.arange(29)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind],
                             total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['chin'] = chin_indices
    new_landmark_group['left_eye'] = np.hstack((outer_leye_indices,
                                                pupil_leye_indices))
    new_landmark_group['right_eye'] = np.hstack((outer_reye_indices,
                                                 pupil_reye_indices))
    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['mouth'] = np.hstack((outer_mouth_indices,
                                             inner_mouth_indices))
    new_landmark_group['nose'] = nose_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def _build_upper_eyelid():
    top_indices = np.arange(0, 7)
    middle_indices = np.arange(12, 17)
    upper_eyelid_indices = np.hstack((top_indices, middle_indices))

    upper_eyelid_connectivity = zip(top_indices, top_indices[1:])
    upper_eyelid_connectivity += [(0, 12)]
    upper_eyelid_connectivity += zip(middle_indices, middle_indices[1:])
    upper_eyelid_connectivity += [(16, 6)]

    return upper_eyelid_indices, upper_eyelid_connectivity


def ibug_open_eye(landmark_group):
    """
    Apply the ibug's "standard" open eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be ``ibug_open_eye``.

    The semantic labels applied are as follows:

      - upper_eyelid
      - lower_eyelid
      - iris
      - pupil
      - sclera

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_open_eye``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 38 points
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_open_eye'
    n_expected_points = 38
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

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

    total_conn = np.asarray(upper_el_connectivity +
                            lower_el_connectivity +
                            iris_connectivity.tolist() +
                            pupil_connectivity.tolist() +
                            sclera_connectivity)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points, total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['upper_eyelid'] = upper_el_indices
    new_landmark_group['lower_eyelid'] = lower_el_indices
    new_landmark_group['pupil'] = np.arange(*pupil_range)
    new_landmark_group['iris'] = np.arange(*iris_range)
    new_landmark_group['sclera'] = sclera_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def ibug_close_eye_points(landmark_group):
    """
    Apply the ibug's "standard" close eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be ``ibug_close_eye``.

    The semantic labels applied are as follows:

      - upper_eyelid
      - lower_eyelid

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_close_eye``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 17 points
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_close_eye'
    n_expected_points = 17
    n_points = landmark_group.lms.n_points
    _validate_input(landmark_group, n_expected_points, group)

    upper_indices, upper_connectivity = _build_upper_eyelid()

    middle_indices = np.arange(12, 17)
    bottom_indices = np.arange(6, 12)
    lower_indices = np.hstack((bottom_indices, 0, middle_indices))
    lower_connectivity = zip(bottom_indices, bottom_indices[1:])
    lower_connectivity += [(0, 12)]
    lower_connectivity += zip(middle_indices, middle_indices[1:])
    lower_connectivity += [(11, 0)]

    total_conn = np.asarray(upper_connectivity + lower_connectivity)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points, total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['upper_eyelid'] = upper_indices
    new_landmark_group['lower_eyelid'] = lower_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def ibug_open_eye_trimesh(landmark_group):
    """
    Apply the ibug's "standard" open eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be ``ibug_open_eye_trimesh``.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_open_eye_trimesh``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 38 points
    """
    from menpo.shape import TriMesh

    group = 'ibug_open_eye_trimesh'
    n_expected_points = 38
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

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

    return group, new_landmark_group


def ibug_close_eye_trimesh(landmark_group):
    """
    Apply the ibug's "standard" close eye semantic labels to the
    landmarks in the given landmark group.

    The group label will be ``ibug_close_eye_trimesh``.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_close_eye_trimesh``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 38 points
    """
    from menpo.shape import TriMesh

    group = 'ibug_close_eye_trimesh'
    n_expected_points = 17
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

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

    return group, new_landmark_group


def ibug_tongue(landmark_group):
    """
    Apply the ibug's "standard" tongue semantic labels to the landmarks in the
    given landmark group.

    The group label will be ``ibug_tongue``.

    The semantic labels applied are as follows:

      - outline
      - bisector

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_tongue``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 19 points
    """
    group = 'ibug_tongue'
    _validate_input(landmark_group, 19, group)
    labels = OrderedDict([
        ('outline', (0, 13, False)),
        ('bisector', (13, 19, False))
    ])
    return group, _relabel_group_from_dict(landmark_group.lms, labels)


def ibug_hand(landmark_group):
    """
    Apply the ibug's "standard" 39 point semantic labels to the landmark group.

    The group label will be ``ibug_hand``.

    The semantic labels applied are as follows:

      - thumb
      - index
      - middle
      - ring
      - pinky
      - palm

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``ibug_hand``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 39 points
    """
    from menpo.shape import PointUndirectedGraph

    group = 'ibug_hand'
    n_points = landmark_group.lms.n_points
    _validate_input(landmark_group, 39, group)

    thumb_indices = np.arange(0, 5)
    index_indices = np.arange(5, 12)
    middle_indices = np.arange(12, 19)
    ring_indices = np.arange(19, 26)
    pinky_indices = np.arange(26, 33)
    palm_indices = np.hstack((np.array([32, 25, 18, 11, 33, 34, 4]),
                              np.arange(35, 39)))

    thumb_connectivity = _connectivity_from_array(thumb_indices,
                                                  close_loop=False)
    index_connectivity = _connectivity_from_array(index_indices,
                                                  close_loop=False)
    middle_connectivity = _connectivity_from_array(middle_indices,
                                                   close_loop=False)
    ring_connectivity = _connectivity_from_array(ring_indices,
                                                 close_loop=False)
    pinky_connectivity = _connectivity_from_array(pinky_indices,
                                                  close_loop=False)
    palm_connectivity = _connectivity_from_array(palm_indices,
                                                 close_loop=True)

    total_conn = np.vstack((thumb_connectivity, index_connectivity,
                            middle_connectivity, ring_connectivity,
                            pinky_connectivity, palm_connectivity))

    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points, total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['thumb'] = thumb_indices
    new_landmark_group['index'] = index_indices
    new_landmark_group['middle'] = middle_indices
    new_landmark_group['ring'] = ring_indices
    new_landmark_group['pinky'] = pinky_indices
    new_landmark_group['palm'] = palm_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def stickmen_pose(landmark_group):
    """
    Apply the stickmen "standard" 12 point semantic labels to the landmark
    group.

    The group label will be ``stickmen_pose``.

    The semantic labels applied are as follows:

      - torso
      - right_upper_arm
      - left_upper_arm
      - right_lower_arm
      - left_lower_arm
      - head

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``stickmen_pose``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 12 points

    References
    ----------
    .. [1] http://www.robots.ox.ac.uk/~vgg/data/stickmen/
    """
    group = 'stickmen_pose'
    _validate_input(landmark_group, 12, group)
    labels = OrderedDict([
        ('torso', (0, 2, False)),
        ('right_upper arm', (2, 4, False)),
        ('left_upper arm', (4, 6, False)),
        ('right_lower_arm', (6, 8, False)),
        ('left_lower_arm', (8, 10, False)),
        ('head', (10, 12, False))
    ])
    return group, _relabel_group_from_dict(landmark_group.lms, labels)


def lsp_pose(landmark_group):
    """
    Apply the lsp "standard" 14 point semantic labels to the landmark
    group.

    The group label will be ``lsp_pose``.

    The semantic labels applied are as follows:

      - left_leg
      - right_leg
      - left_arm
      - right_arm
      - head

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``lsp_pose``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 14 points

    References
    ----------
    .. [1] http://www.comp.leeds.ac.uk/mat4saj/lsp.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'lsp_pose'
    n_points = landmark_group.lms.n_points
    _validate_input(landmark_group, 14, group)

    left_leg_indices = np.arange(0, 3)
    right_leg_indices = np.arange(3, 6)
    left_arm_indices = np.arange(6, 9)
    right_arm_indices = np.arange(9, 12)
    head_indices = np.arange(12, 14)

    left_leg_connectivity = _connectivity_from_array(left_leg_indices)
    right_leg_connectivity = _connectivity_from_array(right_leg_indices)
    left_arm_connectivity = _connectivity_from_array(left_arm_indices)
    right_arm_connectivity = _connectivity_from_array(right_arm_indices)
    head_connectivity = _connectivity_from_array(head_indices)

    total_conn = np.vstack([left_leg_connectivity,
                            right_leg_connectivity,
                            left_arm_connectivity,
                            right_arm_connectivity,
                            head_connectivity])

    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points, total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['left_leg'] = left_leg_indices
    new_landmark_group['right_leg'] = right_leg_indices
    new_landmark_group['left_arm'] = left_arm_indices
    new_landmark_group['right_arm'] = right_arm_indices
    new_landmark_group['head'] = head_indices

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def flic_pose(landmark_group):
    """
    Apply the flic "standard" 11 point semantic labels to the landmark
    group.

    The group label will be ``flic_pose``.

    The semantic labels applied are as follows:

      - left_arm
      - right_arm
      - hips
      - face

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``flic_pose``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 11 points

    References
    ----------
    .. [1] http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC
    """
    group = 'flic_pose'
    _validate_input(landmark_group, 11, group)
    labels = OrderedDict([
        ('left_arm', (0, 3, False)),
        ('right_arm', (3, 6, False)),
        ('hips', (6, 8, False)),
        ('face', (8, 11, True))])

    return group, _relabel_group_from_dict(landmark_group.lms, labels)


def streetscene_car_view_0(landmark_group):
    """
    Apply the 8 point semantic labels of the view 0  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_0``.

    The semantic labels applied are as follows:

      - front
      - bonnet
      - windshield

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``streetscene_car_view_0``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_0'
    n_points = 8
    _validate_input(landmark_group, 20, group)

    front_indices = np.array([0, 1, 3, 2])
    bonnet_indices = np.array([2, 3, 5, 4])
    windshield_indices = np.array([4, 5, 7, 6])

    front_connectivity = _connectivity_from_array(front_indices,
                                                  close_loop=True)
    bonnet_connectivity = _connectivity_from_array(bonnet_indices,
                                                   close_loop=True)
    windshield_connectivity = _connectivity_from_array(windshield_indices,
                                                       close_loop=True)

    total_conn = np.vstack((front_connectivity, bonnet_connectivity,
                            windshield_connectivity))

    ind = np.arange(8)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['front'] = front_indices
    new_landmark_group['bonnet'] = bonnet_indices
    new_landmark_group['windshield'] = windshield_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_1(landmark_group):
    """
    Apply the 14 point semantic labels of the view 1  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_1``.

    The semantic labels applied are as follows:

      - front
      - bonnet
      - windshield
      - left_side

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``streetscene_car_view_1``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_1'
    n_points = 14
    _validate_input(landmark_group, 20, group)

    front_indices = np.array([0, 1, 3, 2])
    bonnet_indices = np.array([2, 3, 5, 4])
    windshield_indices = np.array([4, 5, 7, 6])
    left_side_indices = np.array([0, 2, 4, 6, 8, 9, 10, 11, 13, 12])

    front_connectivity = _connectivity_from_array(front_indices,
                                                  close_loop=True)
    bonnet_connectivity = _connectivity_from_array(bonnet_indices,
                                                   close_loop=True)
    windshield_connectivity = _connectivity_from_array(windshield_indices,
                                                       close_loop=True)
    left_side_connectivity = _connectivity_from_array(left_side_indices,
                                                      close_loop=True)

    total_conn = np.vstack((front_connectivity, bonnet_connectivity,
                            windshield_connectivity, left_side_connectivity))

    ind = np.hstack((np.arange(9), np.array([10, 12, 14, 16, 18])))
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['front'] = front_indices
    new_landmark_group['bonnet'] = bonnet_indices
    new_landmark_group['windshield'] = windshield_indices
    new_landmark_group['left_side'] = left_side_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_2(landmark_group):
    """
    Apply the 10 point semantic labels of the view 2  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_2``.

    The semantic labels applied are as follows:

      - left_side

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: 'streetscene_car_view_2'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_2'
    n_points = 10
    _validate_input(landmark_group, 20, group)

    left_side_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])

    left_side_connectivity = _connectivity_from_array(left_side_indices,
                                                      close_loop=True)

    total_conn = left_side_connectivity

    ind = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['left_side'] = left_side_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_3(landmark_group):
    """
    Apply the 14 point semantic labels of the view 3  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_2``.

    The semantic labels applied are as follows:

      - left_side
      - rear windshield
      - trunk
      - rear

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``streetscene_car_view_3``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_3'
    n_points = 14
    _validate_input(landmark_group, 20, group)

    left_side_indices = np.array([0, 1, 2, 3, 4, 6, 8, 10, 13, 12])
    rear_windshield_indices = np.array([4, 5, 7, 6])
    trunk_indices = np.array([6, 7, 9, 8])
    rear_indices = np.array([8, 9, 11, 10])

    left_side_connectivity = _connectivity_from_array(left_side_indices,
                                                      close_loop=True)
    rear_windshield_connectivity = _connectivity_from_array(
        rear_windshield_indices, close_loop=True)
    trunk_connectivity = _connectivity_from_array(trunk_indices,
                                                  close_loop=True)
    rear_connectivity = _connectivity_from_array(rear_indices, close_loop=True)

    total_conn = np.vstack((left_side_connectivity,
                            rear_windshield_connectivity,
                            trunk_connectivity, rear_connectivity))

    ind = np.array([0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18])
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['left_side'] = left_side_indices
    new_landmark_group['rear_windshield'] = rear_windshield_indices
    new_landmark_group['trunk'] = trunk_indices
    new_landmark_group['rear'] = rear_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_4(landmark_group):
    """
    Apply the 14 point semantic labels of the view 4  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_4``.

    The semantic labels applied are as follows:

      - front
      - bonnet
      - windshield
      - right_side

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: 'streetscene_car_view_4'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_4'
    n_points = 14
    _validate_input(landmark_group, 20, group)

    front_indices = np.array([0, 1, 3, 2])
    bonnet_indices = np.array([2, 3, 5, 4])
    windshield_indices = np.array([4, 5, 7, 6])
    right_side_indices = np.array([8, 9, 10, 11, 13, 12, 1, 3, 5, 7])

    front_connectivity = _connectivity_from_array(front_indices,
                                                  close_loop=True)
    bonnet_connectivity = _connectivity_from_array(bonnet_indices,
                                                   close_loop=True)
    windshield_connectivity = _connectivity_from_array(windshield_indices,
                                                       close_loop=True)
    right_side_connectivity = _connectivity_from_array(right_side_indices,
                                                       close_loop=True)

    total_conn = np.vstack((front_connectivity, bonnet_connectivity,
                            windshield_connectivity,
                            right_side_connectivity))

    ind = np.hstack((np.arange(8), np.array([9, 11, 13, 15, 17, 19])))
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['front'] = front_indices
    new_landmark_group['bonnet'] = bonnet_indices
    new_landmark_group['windshield'] = windshield_indices
    new_landmark_group['right_side'] = right_side_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_5(landmark_group):
    """
    Apply the 10 point semantic labels of the view 5 of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_5``.

    The semantic labels applied are as follows:

      - right_side

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``streetscene_car_view_5``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_5'
    n_points = 10
    _validate_input(landmark_group, 20, group)

    right_side_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])

    right_side_connectivity = _connectivity_from_array(right_side_indices,
                                                       close_loop=True)

    total_conn = right_side_connectivity

    ind = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['right_side'] = right_side_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_6(landmark_group):
    """
    Apply the 14 point semantic labels of the view 6  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_6``.

    The semantic labels applied are as follows:

      - right_side
      - rear_windshield
      - trunk
      - rear

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``streetscene_car_view_3``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_6'
    n_points = 14
    _validate_input(landmark_group, 20, group)

    right_side_indices = np.array([0, 1, 2, 3, 5, 7, 9, 11, 13, 12])
    rear_windshield_indices = np.array([4, 5, 7, 6])
    trunk_indices = np.array([6, 7, 9, 8])
    rear_indices = np.array([8, 9, 11, 10])

    right_side_connectivity = _connectivity_from_array(right_side_indices,
                                                       close_loop=True)
    rear_windshield_connectivity = _connectivity_from_array(
        rear_windshield_indices, close_loop=True)
    trunk_connectivity = _connectivity_from_array(trunk_indices,
                                                  close_loop=True)
    rear_connectivity = _connectivity_from_array(rear_indices, close_loop=True)

    total_conn = np.vstack((right_side_connectivity,
                            rear_windshield_connectivity,
                            trunk_connectivity, rear_connectivity))

    ind = np.array([1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19])
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['right_side'] = right_side_indices
    new_landmark_group['rear_windshield'] = rear_windshield_indices
    new_landmark_group['trunk'] = trunk_indices
    new_landmark_group['rear'] = rear_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def streetscene_car_view_7(landmark_group):
    """
    Apply the 8 point semantic labels of the view 0  of the MIT Street Scene
    Car dataset to the landmark group.

    The group label will be ``streetscene_car_view_7``.

    The semantic labels applied are as follows:

      - rear_windshield
      - trunk
      - rear

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``streetscene_car_view_7``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 20 points

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'streetscene_car_view_7'
    n_points = 8
    _validate_input(landmark_group, 20, group)

    rear_windshield_indices = np.array([0, 1, 3, 2])
    trunk_indices = np.array([2, 3, 5, 4])
    rear_indices = np.array([4, 5, 7, 6])

    rear_windshield_connectivity = _connectivity_from_array(
        rear_windshield_indices, close_loop=True)
    trunk_connectivity = _connectivity_from_array(trunk_indices,
                                                  close_loop=True)
    rear_connectivity = _connectivity_from_array(rear_indices, close_loop=True)

    total_conn = np.vstack((rear_windshield_connectivity,
                            trunk_connectivity, rear_connectivity))

    ind = np.arange(8, 16)
    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points[ind], total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['rear_windshield'] = rear_windshield_indices
    new_landmark_group['trunk'] = trunk_indices
    new_landmark_group['rear'] = rear_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def bu3dfe_83(landmark_group):
    """
    Apply the BU-3DFE (Binghamton University 3D Facial Expression)
    Database 83 point facial annotation markup to this landmark group.


    The group label will be ``bu3dfe_83``.

    The semantic labels applied are as follows:

      - right_eye
      - left_eye
      - right_eyebrow
      - left_eyebrow
      - right_nose
      - left_nose
      - nostrils
      - outer_mouth
      - inner_mouth
      - jaw

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``bu3dfe_83``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 83 points

    References
    ----------
    .. [1] http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'bu3dfe_83'
    n_points = 83
    _validate_input(landmark_group, n_points, group)

    reye_indices = np.arange(0, 8)
    leye_indices = np.arange(8, 16)
    rbrow_indices = np.arange(16, 26)
    lbrow_indices = np.arange(26, 36)
    rnose_indicies = np.arange(36, 39)
    lnose_indicies = np.arange(39, 42)
    nostril_indices = np.arange(42, 48)
    outermouth_indices = np.arange(48, 60)
    innermouth_indices = np.arange(60, 68)
    jaw_indices = np.arange(68, 83)

    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices,
                                                  close_loop=True)
    lbrow_connectivity = _connectivity_from_array(lbrow_indices,
                                                  close_loop=True)
    rnose_connectivity = _connectivity_from_array(rnose_indicies)
    nostril_connectivity = _connectivity_from_array(nostril_indices)
    lnose_connectivity = _connectivity_from_array(lnose_indicies)
    outermouth_connectivity = _connectivity_from_array(outermouth_indices,
                                                       close_loop=True)
    innermouth_connectivity = _connectivity_from_array(innermouth_indices,
                                                       close_loop=True)
    jaw_connectivity = _connectivity_from_array(jaw_indices)

    total_conn = np.vstack([
        reye_connectivity, leye_connectivity,
        rbrow_connectivity, lbrow_connectivity,
        rnose_connectivity, nostril_connectivity, lnose_connectivity,
        outermouth_connectivity, innermouth_connectivity,
        jaw_connectivity
    ])

    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph(landmark_group.lms.points, total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_nose'] = rnose_indicies
    new_landmark_group['left_nose'] = lnose_indicies
    new_landmark_group['nostrils'] = nostril_indices
    new_landmark_group['outer_mouth'] = outermouth_indices
    new_landmark_group['inner_mouth'] = innermouth_indices
    new_landmark_group['jaw'] = jaw_indices

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


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
