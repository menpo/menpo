from collections import OrderedDict
import numpy as np
from menpo.landmark.base import LandmarkGroup

from menpo.landmark.labels.base import (
    _validate_input, _connectivity_from_array, _relabel_group_from_dict)


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

    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points,
                                             total_conn))

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


def human36M_pose_32(landmark_group):
    """
    Apply the human3.6M "standard" 32 point semantic labels to the landmark
    group.

    The group label will be ``human36M_pose_32``.

    The semantic labels applied are as follows:

      - pelvis
      - right_leg
      - left_leg
      - spine
      - head
      - left_arm
      - left_hand
      - right_arm
      - right_hand
      - torso

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``human36M_pose_32``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 32 points

    References
    ----------
    .. [1] http://vision.imar.ro/human3.6m/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'human36M_pose_32'
    _validate_input(landmark_group, 32, group)

    pelvis_indices = np.array([1, 0, 6])
    right_leg_indices = np.array(range(1, 6))
    left_leg_indices = np.array(range(6, 11))
    spine_indices = np.array([11, 12, 13])
    head_indices = np.array([13, 14, 15])
    left_arm_indices = np.array([16, 17, 18, 19, 23])
    left_hand_indices = np.array([20, 21, 22])
    right_arm_indices = np.array([24, 25, 26, 27, 29, 31])
    right_hand_indices = np.array([28, 29, 30])
    torso_indices = np.array([0, 1, 25, 13, 17, 6])

    pelvis_connectivity = _connectivity_from_array(pelvis_indices)
    right_leg_connectivity = _connectivity_from_array(right_leg_indices)
    left_leg_connectivity = _connectivity_from_array(left_leg_indices)
    spine_connectivity = _connectivity_from_array(spine_indices)
    head_connectivity = _connectivity_from_array(head_indices)
    left_arm_connectivity = _connectivity_from_array(left_arm_indices)
    left_hand_connectivity = _connectivity_from_array(left_hand_indices)
    right_arm_connectivity = _connectivity_from_array(right_arm_indices)
    right_hand_connectivity = _connectivity_from_array(right_hand_indices)
    torso_connectivity = _connectivity_from_array(torso_indices,
                                                  close_loop=True)

    total_conn = np.vstack([
        pelvis_connectivity, right_leg_connectivity, left_leg_connectivity,
        spine_connectivity, head_connectivity, left_arm_connectivity,
        left_hand_connectivity, right_arm_connectivity,
        right_hand_connectivity, torso_connectivity])

    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points,
                                             total_conn))

    new_landmark_group['pelvis'] = pelvis_indices
    new_landmark_group['right_leg'] = right_leg_indices
    new_landmark_group['left_leg'] = left_leg_indices
    new_landmark_group['spine'] = spine_indices
    new_landmark_group['head'] = head_indices
    new_landmark_group['left_arm'] = left_arm_indices
    new_landmark_group['left_hand'] = left_hand_indices
    new_landmark_group['right_arm'] = right_arm_indices
    new_landmark_group['right_hand'] = right_hand_indices
    new_landmark_group['torso'] = torso_indices

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group


def human36M_pose_17(landmark_group):
    """
    Apply the human3.6M "standard" 17 point semantic labels (based on the
    original semantic labels of Human3.6 but removing the annotations
    corresponding to duplicate points, soles and palms) to the landmark group.

    The group label will be ``human36M_pose_17``.

    The semantic labels applied are as follows:

      - pelvis
      - right_leg
      - left_leg
      - spine
      - head
      - left_arm
      - right_arm
      - torso

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``human36M_pose_17``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    error : :map:`LabellingError`
        If the given landmark group contains less than 32 points

    References
    ----------
    .. [1] http://vision.imar.ro/human3.6m/
    """
    from menpo.shape import PointUndirectedGraph

    group = 'human36M_pose_17'
    _validate_input(landmark_group, 32, group)

    pelvis_indices = np.array([1, 0, 4])
    right_leg_indices = np.arange(1, 4)
    left_leg_indices = np.arange(4, 7)
    spine_indices = np.array([0, 7, 8])
    head_indices = np.array([8, 9, 10])
    left_arm_indices = np.array([8, 11, 12, 13])
    right_arm_indices = np.array([8, 14, 15, 16])
    torso_indices = np.array([0, 1, 14, 8, 11, 4])

    pelvis_connectivity = _connectivity_from_array(pelvis_indices)
    right_leg_connectivity = _connectivity_from_array(right_leg_indices)
    left_leg_connectivity = _connectivity_from_array(left_leg_indices)
    spine_connectivity = _connectivity_from_array(spine_indices)
    head_connectivity = _connectivity_from_array(head_indices)
    left_arm_connectivity = _connectivity_from_array(left_arm_indices)
    right_arm_connectivity = _connectivity_from_array(right_arm_indices)
    torso_connectivity = _connectivity_from_array(torso_indices,
                                                  close_loop=True)

    total_conn = np.vstack([
        pelvis_connectivity, right_leg_connectivity, left_leg_connectivity,
        spine_connectivity, head_connectivity, left_arm_connectivity,
        right_arm_connectivity, torso_connectivity])

    # Ignore duplicate points, sole and palms
    ind = np.hstack((np.arange(0, 4), np.arange(6, 9), np.arange(12, 16),
                     np.arange(17, 20), np.arange(25, 28)))
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

    new_landmark_group['pelvis'] = pelvis_indices
    new_landmark_group['right_leg'] = right_leg_indices
    new_landmark_group['left_leg'] = left_leg_indices
    new_landmark_group['spine'] = spine_indices
    new_landmark_group['head'] = head_indices
    new_landmark_group['left_arm'] = left_arm_indices
    new_landmark_group['right_arm'] = right_arm_indices
    new_landmark_group['torso'] = torso_indices

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group
