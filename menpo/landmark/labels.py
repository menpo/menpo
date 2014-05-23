import numpy as np
from menpo.landmark.base import LandmarkGroup
from menpo.landmark.exceptions import LabellingError


def imm_58_points(landmark_group):
    """
    Apply the 58 point semantic labels from the
    IMM dataset to the landmarks in the given landmark group.

    The label applied to this new manager will be 'imm_58_points'.

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
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group with group label 'imm_58_points'

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 58 points

    References
    -----------
    .. [1] http://www2.imm.dtu.dk/~aam/
    """
    group_label = 'imm_58_points'
    n_points = landmark_group.lms.n_points

    if n_points != 58:
        raise LabellingError("{0} mark-up expects exactly 58 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label,
        landmark_group.lms.copy(),
        {'all': np.ones(n_points, dtype=np.bool)})

    new_landmark_group['chin'] = np.arange(13)
    new_landmark_group['leye'] = np.arange(13, 21)
    new_landmark_group['reye'] = np.arange(21, 29)
    new_landmark_group['leyebrow'] = np.arange(29, 34)
    new_landmark_group['reyebrow'] = np.arange(34, 39)
    new_landmark_group['mouth'] = np.arange(39, 47)
    new_landmark_group['nose'] = np.arange(47, 58)

    return new_landmark_group


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
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group with group label 'ibug_68_points'. The pointcloud
        is also copied.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    # TODO: This should probably be some sort of graph that maintains the
    # connectivity defined by ibug (and thus not a PointCloud)
    group_label = 'ibug_68_points'
    n_points = landmark_group.lms.n_points

    if n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label,
        landmark_group.lms.copy(),
        {'all': np.ones(n_points, dtype=np.bool)})

    new_landmark_group['chin'] = np.arange(17)
    new_landmark_group['leye'] = np.arange(36, 42)
    new_landmark_group['reye'] = np.arange(42, 48)
    new_landmark_group['leyebrow'] = np.arange(17, 22)
    new_landmark_group['reyebrow'] = np.arange(22, 27)
    new_landmark_group['mouth'] = np.arange(48, 68)
    new_landmark_group['nose'] = np.arange(27, 36)

    return new_landmark_group


def ibug_68_contour(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmarks in
    the given landmark manager.

    The label applied to this new group will be 'ibug_68_contour'.

    The semantic labels applied are as follows:

      - contour

    Parameters
    ----------
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group with label 'ibug_68_contour'

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark set contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    # TODO: This should probably be some sort of graph that maintains the
    # connectivity defined by ibug (and thus not a PointCloud)
    group_label = 'ibug_68_contour'
    n_points = landmark_group.lms.n_points

    if n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    new_landmarks = landmark_group.lms.copy()
    ind = np.hstack((np.arange(17), np.arange(16, 21),
                     np.arange(21, 26), np.arange(1)))
    new_landmarks.points = new_landmarks.points[ind]
    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label, new_landmarks,
        {'all': np.ones(new_landmarks.n_points, dtype=np.bool)})

    return new_landmark_group


def ibug_68_trimesh(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmarks in
    the given landmark group.

    The label applied to this new group will be 'ibug_68_trimesh'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark manager with label 'ibug_68_trimesh'

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
    n_points = landmark_group.lms.n_points

    if n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

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
        landmark_group._target, group_label,
        TriMesh(landmark_group.lms.points, tri_list),
        {'all': np.ones(n_points, dtype=np.bool)})

    return new_landmark_group

# TODO: ibug_68_all? imports points, contour and trimesh?


def ibug_68_closed_mouth(landmark_group):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of multiPIE) to the landmarks in
    the given landmark group - but ignore the 3 points that are coincident for
    a closed mouth. Therefore, there only 65 points are returned.

    The label applied to this new group will be 'ibug_68_closed_mouth'.

    The semantic labels applied are as follows:

      - tri

    Parameters
    ----------
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark manager with label 'ibug_68_closed_mouth'

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 65 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_68_closed_mouth'
    n_points = landmark_group.lms.n_points

    if landmark_group.lms.n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    # Ignore the 3 coincident points (the last 3 points)
    new_landmarks = landmark_group.lms.copy()
    new_landmarks.points = new_landmarks.points[:-3]
    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label, new_landmarks,
        {'all': np.ones(new_landmarks.n_points, dtype=np.bool)})

    new_landmark_group['chin'] = np.arange(17)
    new_landmark_group['leye'] = np.arange(36, 42)
    new_landmark_group['reye'] = np.arange(42, 48)
    new_landmark_group['leyebrow'] = np.arange(17, 22)
    new_landmark_group['reyebrow'] = np.arange(22, 27)
    new_landmark_group['mouth'] = np.arange(48, 65)
    new_landmark_group['nose'] = np.arange(27, 36)

    return new_landmark_group


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
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group with group label 'ibug_66_points'. The pointcloud
        is also copied.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_66_points'
    n_points = landmark_group.lms.n_points

    if landmark_group.lms.n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    # Ignore the 2 inner mouth corners
    new_landmarks = landmark_group.lms.copy()
    ind = np.hstack((np.arange(0, 60), np.arange(61, 64),
                     np.arange(65, 68)))
    new_landmarks.points = new_landmarks.points[ind]
    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label, new_landmarks,
        {'all': np.ones(new_landmarks.n_points, dtype=np.bool)})

    new_landmark_group['chin'] = np.arange(17)
    new_landmark_group['leye'] = np.arange(36, 42)
    new_landmark_group['reye'] = np.arange(42, 48)
    new_landmark_group['leyebrow'] = np.arange(17, 22)
    new_landmark_group['reyebrow'] = np.arange(22, 27)
    new_landmark_group['mouth'] = np.arange(48, 66)
    new_landmark_group['nose'] = np.arange(27, 36)

    return new_landmark_group


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
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group with group label 'ibug_51_points'. The pointcloud
        is also copied.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_51_points'
    n_points = landmark_group.lms.n_points

    if landmark_group.lms.n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    # Ignore the chin region
    new_landmarks = landmark_group.lms.copy()
    ind = np.arange(17, 68)
    new_landmarks.points = new_landmarks.points[ind]
    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label, new_landmarks,
        {'all': np.ones(new_landmarks.n_points, dtype=np.bool)})

    new_landmark_group['leye'] = np.arange(19, 25)
    new_landmark_group['reye'] = np.arange(25, 31)
    new_landmark_group['leyebrow'] = np.arange(0, 5)
    new_landmark_group['reyebrow'] = np.arange(5, 10)
    new_landmark_group['mouth'] = np.arange(31, 51)
    new_landmark_group['nose'] = np.arange(10, 19)

    return new_landmark_group


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
    landmark_group: :class:`menpo.landmark.base.LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    landmark_group : :class:`menpo.landmark.base.LandmarkGroup`
        New landmark group with group label 'ibug_49_points'. The pointcloud
        is also copied.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 68 points

    References
    ----------
    .. [1] http://www.multipie.org/
    """
    group_label = 'ibug_49_points'
    n_points = landmark_group.lms.n_points

    if landmark_group.lms.n_points != 68:
        raise LabellingError("{0} mark-up expects exactly 68 "
                             "points. However, the given landmark group only "
                             "has {1} points".format(group_label, n_points))

    # Ignore the chin region and the 2 inner mouth corners
    new_landmarks = landmark_group.lms.copy()
    ind = np.hstack((np.arange(17, 60), np.arange(61, 64),
                     np.arange(65, 68)))
    new_landmarks.points = new_landmarks.points[ind]
    new_landmark_group = LandmarkGroup(
        landmark_group._target, group_label, new_landmarks,
        {'all': np.ones(new_landmarks.n_points, dtype=np.bool)})

    new_landmark_group['leye'] = np.arange(19, 25)
    new_landmark_group['reye'] = np.arange(25, 31)
    new_landmark_group['leyebrow'] = np.arange(0, 5)
    new_landmark_group['reyebrow'] = np.arange(5, 10)
    new_landmark_group['mouth'] = np.arange(31, 49)
    new_landmark_group['nose'] = np.arange(10, 19)

    return new_landmark_group


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
        return a new LandmarkGroup with semantic labels applied.

    Returns
    -------
    landmarkable : :class:`menpo.landmark.base.Landmarkable`
        landmarkable with label (this is just for convenience,
        the object will actually be modified in place)
    """
    group = label_func(landmarkable.landmarks[group_label])
    landmarkable.landmarks[group.group_label] = group
    return landmarkable
