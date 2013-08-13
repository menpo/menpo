import numpy as np
from pybug.landmark import LandmarkManager
from pybug.landmark.exceptions import LabellingError


def imm_58_points(landmarks):
    """
    Apply the 58 point semantic labels from the
    `IMM dataset <http://www2.imm.dtu.dk/~aam/>`_ to the landmarks in the
    given landmark manager. The label applied to this new manager will be
    'imm_58_points'.

    The semantic labels applied are as follows:

      - chin
      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    :param landmarks: The landmarks to apply semantic labels to.
    :type landmarks:
        :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    :return: New landmark manager with label 'imm_58_points'
    :raises:
        :class:`LabellingError <pybug.landmark.exceptions.LabellingError>`
        if the given landmark set contains less than 58 points
    :rtype: :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    """
    # TODO: This should probably be some sort of graph that maintains the
    # connectivity defined in the IMM dataset (and not thus not a PointCloud)
    from pybug.shape import PointCloud

    label = 'imm_58_points'
    points = landmarks.all_landmarks.points
    try:
        landmark_dict = {'chin': PointCloud(points[:13]),
                         'leye': PointCloud(points[13:21]),
                         'reye': PointCloud(points[21:29]),
                         'leyebrow': PointCloud(points[29:34]),
                         'reyebrow': PointCloud(points[34:39]),
                         'mouth': PointCloud(points[39:47]),
                         'nose': PointCloud(points[47:])}
    except IndexError as e:
        raise LabellingError("IMM's 58 points mark-up expects at least 58 "
                             "points. However, {0}".format(e.message))

    return LandmarkManager(landmarks.shape, label, landmark_dict)

def ibug_68_points(landmarks):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of http://www.multipie.org/) to the landmarks in
    the given landmark manager. The label applied to this new manager will be
    'ibug_68_points'.

    The semantic labels applied are as follows:

      - chin
      - leye
      - reye
      - leyebrow
      - reyebrow
      - mouth
      - nose

    :param landmarks: The landmarks to apply semantic labels to.
    :type landmarks:
        :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    :return: New landmark manager with label 'ibug_68_points'
    :raises:
        :class:`LabellingError <pybug.landmark.exceptions.LabellingError>`
        if the given landmark set contains less than 68 points
    :rtype: :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    """
    # TODO: might have to rethink what's the v=bes way of implementing this
    # functions
    # TODO: This should probably be some sort of graph that maintains the
    # connectivity defined by ibug (and not thus not a PointCloud)
    from pybug.shape import PointCloud

    label = 'ibug_68_points'
    points = landmarks.all_landmarks.points
    try:
        landmark_dict = {'chin': PointCloud(points[:17]),
                         'leye': PointCloud(points[36:42]),
                         'reye': PointCloud(points[42:48]),
                         'leyebrow': PointCloud(points[17:22]),
                         'reyebrow': PointCloud(points[22:27]),
                         'mouth': PointCloud(points[48:]),
                         'nose': PointCloud(points[27:36])}
    except IndexError as e:
        raise LabellingError("ibug's 68 points mark-up expects at least 68 "
                             "points. However, {0}".format(e.message))

    return LandmarkManager(landmarks.shape, label, landmark_dict)


def ibug_68_contour(landmarks):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of http://www.multipie.org/) to the landmarks in
    the given landmark manager. The label applied to this new manager will be
    'ibug_68_points'.

    The semantic labels applied are as follows:

      - contour

    :param landmarks: The landmarks to apply semantic labels to.
    :type landmarks:
        :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    :return: New landmark manager with label 'ibug_68_points'
    :raises:
        :class:`LabellingError <pybug.landmark.exceptions.LabellingError>`
        if the given landmark set contains less than 68 points
    :rtype: :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    """
    # TODO: This should probably be some sort of graph that maintains the
    # connectivity defined by ibug (and not thus not a PointCloud)
    from pybug.shape import PointCloud

    label = 'ibug_68_contour'
    points = landmarks.all_landmarks.points
    try:
        landmark_dict = {'contour': PointCloud(np.concatenate([points[:17],
                                                               points[
                                                               26:21:-1],
                                                               points[
                                                               21:16:-1],
                                                               points[:1]]))}
    except IndexError as e:
        raise LabellingError("ibug's 68 points mark-up expects at least 68 "
                             "points. However, {0}".format(e.message))

    return LandmarkManager(landmarks.shape, label, landmark_dict)


def ibug_68_trimesh(landmarks):
    """
    Apply the ibug's "standard" 68 point semantic labels (based on the
    original semantic labels of http://www.multipie.org/) to the landmarks in
    the given landmark manager. The label applied to this new manager will be
    'ibug_68_points'.

    The semantic labels applied are as follows:

      - contour

    :param landmarks: The landmarks to apply semantic labels to.
    :type landmarks:
        :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    :return: New landmark manager with label 'ibug_68_points'
    :raises:
        :class:`LabellingError <pybug.landmark.exceptions.LabellingError>`
        if the given landmark set contains less than 68 points
    :rtype: :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>`
    """

    from pybug.shape import TriMesh

    label = 'ibug_68_trimesh'
    points = landmarks.all_landmarks.points
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
    try:
        landmark_dict = {'tri': TriMesh(points, tri_list)}
    except IndexError as e:
        raise LabellingError("ibug's 68 points mark-up expects at least 68 "
                             "points. However, {0}".format(e.message))

    return LandmarkManager(landmarks.shape, label, landmark_dict)

# TODO: ibug_68_all? imports points, contour and trimesh?


def labeller(landmarkables, label, label_func):
    """
    Takes a list of landmarkable objects and a label indicating which set of
    landmarks should have semantic meaning attached to them. The labelling
    function will add a new set of landmarks to each object that have been
    semantically annotated. For example, the different components of the face
    will be separated in to separate pointclouds and given a label that
    identifies their semantic meaning.

    :param landmarkables: List of landmarkable objects
    :type landmarkables:
        [:class:`Landmarkable <pybug.landmark.base.Landmarkable>`]
    :param label: The label of the landmark set to apply semantic labels to
    :type label: String
    :param label_func: A labelling function taken from this module
    :type label_func: Function that takes a
        :class:`LandmarkManager <pybug.landmark.base.LandmarkManager>` and
        returns a new LandmarkManager with semantic labels applied.
    :return: The list of modified landmarkables (this is just for convenience,
        the list will actually be modified in place)
    """
    landmarks = [label_func(l.get_landmark_set(label))
                 for l in landmarkables]

    for (obj, lmarks) in zip(landmarkables, landmarks):
        obj.add_landmark_set(lmarks.label, lmarks.landmark_dict)

    return landmarkables