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
        raise LabellingError("IMM dataset expects at least 58 points. "
                              "However, {0}".format(e.message))

    return LandmarkManager(landmarks.shape, label, landmark_dict)


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