from collections import OrderedDict
import numpy as np

from menpo.landmark.base import LandmarkGroup
from menpo.landmark.labels.base import (
    _validate_input, _connectivity_from_array, _relabel_group_from_dict,
    _connectivity_from_range, _labeller)


@_labeller()
def face_ibug_68_to_face_ibug_68():
    """
    Apply the IBUG 68 point semantic labels (based on the
    original semantic labels of multiPIE and 300W).

    The group label will be ``ibug_face_68``.

    The semantic labels are as follows:

      - jaw
      - left_eyebrow
      - right_eyebrow
      - nose
      - left_eye
      - right_eye
      - mouth

    Parameters
    ----------
    x : :map:`LandmarkGroup` or :map:`PointCloud` or `ndarray`
        The input landmark group, pointcloud or array to label. If a pointcloud
        is passed, then only the connectivity information is propagated to
        the pointcloud (a subclass of :map:`PointCloud` may be returned).

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

    Raises
    ------
    :map:`LabellingError`
        If the given landmark group/pointcloud contains less than 68 points.

    References
    ----------
    .. [1] http://www.multipie.org/
    .. [2] http://ibug.doc.ic.ac.uk/resources/300-W/
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 68

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
    leye_connectivity = _connectivity_from_array(leye_indices,
                                                 close_loop=True)
    reye_connectivity = _connectivity_from_array(reye_indices,
                                                 close_loop=True)
    mouth_connectivity = np.vstack([
        _connectivity_from_array(outer_mouth_indices, close_loop=True),
        _connectivity_from_array(inner_mouth_indices, close_loop=True)])

    all_connectivity = np.vstack([
        jaw_connectivity, lbrow_connectivity, rbrow_connectivity,
        nose_connectivity, leye_connectivity, reye_connectivity,
        mouth_connectivity
    ])

    template = PointUndirectedGraph.init_from_edges(
        np.zeros([n_expected_points, 2]), all_connectivity)

    mapping = OrderedDict()
    mapping['jaw'] = jaw_indices
    mapping['left_eyebrow'] = lbrow_indices
    mapping['right_eyebrow'] = rbrow_indices
    mapping['nose'] = np.hstack((upper_nose_indices,
                                       lower_nose_indices))
    mapping['left_eye'] = leye_indices
    mapping['right_eye'] = reye_indices
    mapping['mouth'] = np.hstack((outer_mouth_indices,
                                        inner_mouth_indices))

    return template, mapping


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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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

    tri_list = np.array([[47, 29, 28], [44, 43, 23], [38, 20, 21],
                         [47, 28, 42], [49, 61, 60], [40, 41, 37],
                         [37, 19, 20], [28, 40, 39], [38, 21, 39],
                         [36,  1,  0], [48, 59,  4], [49, 60, 48],
                         [67, 59, 60], [13, 53, 14], [61, 51, 62],
                         [57,  8,  7], [52, 51, 33], [61, 67, 60],
                         [52, 63, 51], [66, 56, 57], [35, 30, 29],
                         [53, 52, 35], [37, 36, 17], [18, 37, 17],
                         [37, 38, 40], [38, 37, 20], [19, 37, 18],
                         [38, 39, 40], [28, 29, 40], [41, 36, 37],
                         [27, 39, 21], [41, 31,  1], [30, 32, 31],
                         [33, 51, 50], [33, 30, 34], [31, 40, 29],
                         [36,  0, 17], [31,  2,  1], [31, 41, 40],
                         [ 1, 36, 41], [31, 49,  2], [ 2, 49,  3],
                         [60, 59, 48], [ 3, 49, 48], [31, 32, 50],
                         [48,  4,  3], [59,  5,  4], [58, 67, 66],
                         [ 5, 59, 58], [58, 59, 67], [ 7,  6, 58],
                         [66, 57, 58], [13, 54, 53], [ 7, 58, 57],
                         [ 6,  5, 58], [50, 61, 49], [62, 67, 61],
                         [31, 50, 49], [32, 33, 50], [30, 33, 32],
                         [34, 52, 33], [35, 52, 34], [53, 63, 52],
                         [62, 63, 65], [62, 51, 63], [66, 65, 56],
                         [63, 53, 64], [62, 66, 67], [62, 65, 66],
                         [57, 56,  9], [65, 63, 64], [ 8, 57,  9],
                         [ 9, 56, 10], [10, 56, 11], [11, 56, 55],
                         [11, 55, 12], [56, 65, 55], [55, 64, 54],
                         [55, 65, 64], [55, 54, 12], [64, 53, 54],
                         [12, 54, 13], [45, 46, 44], [35, 34, 30],
                         [14, 53, 35], [15, 46, 45], [27, 28, 39],
                         [27, 42, 28], [35, 29, 47], [30, 31, 29],
                         [15, 35, 46], [15, 14, 35], [43, 22, 23],
                         [27, 21, 22], [24, 44, 23], [44, 47, 43],
                         [43, 47, 42], [46, 35, 47], [26, 45, 44],
                         [46, 47, 44], [25, 44, 24], [25, 26, 44],
                         [16, 15, 45], [16, 45, 26], [22, 42, 43],
                         [50, 51, 61], [27, 22, 42]])
    new_landmark_group = LandmarkGroup(
        TriMesh(landmark_group.lms.points, tri_list, copy=False),
        OrderedDict([('tri', np.ones(n_points, dtype=np.bool))]))

    return group, new_landmark_group


def ibug_face_66_trimesh(landmark_group):
    """
    Apply the ibug's "standard" 66 point triangulation to the landmarks in
    the given landmark group.

    The group label will be 'ibug_face_66_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: 'ibug_face_66_trimesh'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 66 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import TriMesh

    # apply ibug_face_66
    _, landmark_group = ibug_face_66(landmark_group)

    group = 'ibug_face_66_trimesh'
    n_expected_points = 66
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

    tri_list = np.array([[47, 29, 28], [44, 43, 23], [38, 20, 21],
                         [47, 28, 42], [40, 41, 37], [51, 62, 61],
                         [37, 19, 20], [28, 40, 39], [38, 21, 39],
                         [36,  1,  0],  [48, 59, 4], [49, 60, 48],
                         [13, 53, 14], [60, 51, 61], [51, 51, 62],
                         [52, 51, 33], [49, 50, 60], [57,  7,  8],
                         [64, 56, 57], [35, 30, 29], [52, 62, 53],
                         [53, 52, 35], [37, 36, 17], [18, 37, 17],
                         [37, 38, 40], [38, 37, 20], [19, 37, 18],
                         [38, 39, 40], [28, 29, 40], [41, 36, 37],
                         [27, 39, 21], [41, 31,  1], [30, 32, 31],
                         [33, 51, 50], [33, 30, 34], [31, 40, 29],
                         [36,  0, 17], [31,  2,  1], [31, 41, 40],
                         [ 1, 36, 41], [31, 49,  2], [ 2, 49,  3],
                         [ 3, 49, 48], [31, 32, 50], [62, 53, 54],
                         [48,  4,  3], [59,  5,  4], [58, 65, 64],
                         [ 5, 59, 58], [58, 59, 65], [ 7,  6, 58],
                         [64, 57, 58], [13, 54, 53], [ 7, 58, 57],
                         [ 6,  5, 58], [63, 55, 54], [65, 59, 48],
                         [31, 50, 49], [32, 33, 50], [30, 33, 32],
                         [34, 52, 33], [35, 52, 34], [48, 60, 65],
                         [64, 63, 56], [60, 65, 61], [65, 64, 61],
                         [57, 56,  9], [ 8, 57,  9], [64, 63, 61],
                         [ 9, 56, 10], [10, 56, 11], [11, 56, 55],
                         [11, 55, 12], [56, 63, 55], [51, 52, 62],
                         [55, 54, 12], [63, 54, 62], [61, 62, 63],
                         [12, 54, 13], [45, 46, 44], [35, 34, 30],
                         [14, 53, 35], [15, 46, 45], [27, 28, 39],
                         [27, 42, 28], [35, 29, 47], [30, 31, 29],
                         [15, 35, 46], [15, 14, 35], [43, 22, 23],
                         [27, 21, 22], [24, 44, 23], [44, 47, 43],
                         [43, 47, 42], [46, 35, 47], [26, 45, 44],
                         [46, 47, 44], [25, 44, 24], [25, 26, 44],
                         [16, 15, 45], [16, 45, 26], [22, 42, 43],
                         [50, 60, 51], [27, 22, 42]])
    new_landmark_group = LandmarkGroup(
        TriMesh(landmark_group.lms.points, tri_list, copy=False),
        OrderedDict([('tri', np.ones(n_points, dtype=np.bool))]))

    return group, new_landmark_group


def ibug_face_51_trimesh(landmark_group):
    """
    Apply the ibug's "standard" 51 point triangulation to the landmarks in
    the given landmark group.

    The group label will be 'ibug_face_51_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: 'ibug_face_51_trimesh'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 51 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import TriMesh

    # apply ibug_face_51
    _, landmark_group = ibug_face_51(landmark_group)

    group = 'ibug_face_51_trimesh'
    n_expected_points = 51
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

    tri_list = np.array([[30, 12, 11], [27, 26,  6], [21,  3,  4],
                         [30, 11, 25], [32, 44, 43], [23, 24, 20],
                         [20,  2,  3], [11, 23, 22], [21,  4, 22],
                         [32, 43, 31], [50, 42, 43], [44, 34, 45],
                         [35, 34, 16], [44, 50, 43], [35, 46, 34],
                         [49, 39, 40], [18, 13, 12], [36, 35, 18],
                         [20, 19,  0], [ 1, 20,  0], [20, 21, 23],
                         [21, 20,  3], [ 2, 20,  1], [21, 22, 23],
                         [11, 12, 23], [24, 19, 20], [10, 22,  4],
                         [13, 15, 14], [16, 34, 33], [16, 13, 17],
                         [14, 23, 12], [14, 24, 23], [43, 42, 31],
                         [14, 15, 33], [41, 50, 49], [41, 42, 50],
                         [49, 40, 41], [33, 44, 32], [45, 50, 44],
                         [14, 33, 32], [15, 16, 33], [13, 16, 15],
                         [17, 35, 16], [18, 35, 17], [36, 46, 35],
                         [45, 46, 48], [45, 34, 46], [49, 48, 39],
                         [46, 36, 47], [45, 49, 50], [45, 48, 49],
                         [48, 46, 47], [39, 48, 38], [38, 47, 37],
                         [38, 48, 47], [47, 36, 37], [28, 29, 27],
                         [18, 17, 13], [10, 11, 22], [10, 25, 11],
                         [18, 12, 30], [13, 14, 12], [26,  5,  6],
                         [10,  4,  5], [ 7, 27,  6], [27, 30, 26],
                         [26, 30, 25], [29, 18, 30], [ 9, 28, 27],
                         [29, 30, 27], [ 8, 27,  7], [ 8,  9, 27],
                         [ 5, 25, 26], [33, 34, 44], [10,  5, 25]])
    new_landmark_group = LandmarkGroup(
        TriMesh(landmark_group.lms.points, tri_list, copy=False),
        OrderedDict([('tri', np.ones(n_points, dtype=np.bool))]))

    return group, new_landmark_group


def ibug_face_49_trimesh(landmark_group):
    """
    Apply the ibug's "standard" 49 point triangulation to the landmarks in
    the given landmark group.

    The group label will be 'ibug_face_49_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: 'ibug_face_49_trimesh'
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 49 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    from menpo.shape import TriMesh

    # apply ibug_face_49
    _, landmark_group = ibug_face_49(landmark_group)

    group = 'ibug_face_49_trimesh'
    n_expected_points = 49
    n_points = landmark_group.lms.n_points

    _validate_input(landmark_group, n_expected_points, group)

    tri_list = np.array([[30, 12, 11], [27, 26,  6], [21,  3,  4],
                         [30, 11, 25], [23, 24, 20], [34, 45, 44],
                         [20,  2,  3], [11, 23, 22], [21,  4, 22],
                         [32, 43, 31], [43, 34, 44], [34, 34, 45],
                         [35, 34, 16], [32, 33, 43], [47, 39, 40],
                         [18, 13, 12], [35, 45, 36], [36, 35, 18],
                         [20, 19,  0], [ 1, 20,  0], [20, 21, 23],
                         [21, 20,  3], [ 2, 20,  1], [21, 22, 23],
                         [11, 12, 23], [24, 19, 20], [10, 22,  4],
                         [13, 15, 14], [16, 34, 33], [16, 13, 17],
                         [14, 23, 12], [14, 24, 23], [14, 15, 33],
                         [45, 36, 37], [41, 48, 47], [41, 42, 48],
                         [47, 40, 41], [46, 38, 37], [48, 42, 31],
                         [14, 33, 32], [15, 16, 33], [13, 16, 15],
                         [17, 35, 16], [18, 35, 17], [31, 43, 48],
                         [47, 46, 39], [43, 48, 44], [48, 47, 44],
                         [47, 46, 44], [39, 46, 38], [46, 37, 45],
                         [28, 29, 27], [18, 17, 13], [10, 11, 22],
                         [10, 25, 11], [18, 12, 30], [13, 14, 12],
                         [26,  5,  6], [10,  4,  5], [ 7, 27,  6],
                         [27, 30, 26], [26, 30, 25], [29, 18, 30],
                         [ 9, 28, 27], [29, 30, 27], [ 8, 27,  7],
                         [ 8,  9, 27], [ 5, 25, 26], [33, 43, 34],
                         [10,  5, 25], [34, 35, 45], [44, 45, 46]])
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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points,
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points,
                                             total_conn))

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
