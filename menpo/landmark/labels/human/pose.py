from collections import OrderedDict
import numpy as np

from ..base import (labeller_func, validate_input, connectivity_from_array,
                    pcloud_and_lgroup_from_ranges)


@labeller_func(group_label='pose_stickmen_12')
def pose_stickmen_12_to_pose_stickmen_12(pcloud):
    r"""
    Apply the 'stickmen' 12-point semantic labels.

    The semantic labels applied are as follows:

      - torso
      - right_upper_arm
      - left_upper_arm
      - right_lower_arm
      - left_lower_arm
      - head

    References
    ----------
    .. [1] http://www.robots.ox.ac.uk/~vgg/data/stickmen/
    """
    n_expected_points = 12
    validate_input(pcloud, n_expected_points)

    labels = OrderedDict([
        ('torso', (0, 2, False)),
        ('right_upper arm', (2, 4, False)),
        ('left_upper arm', (4, 6, False)),
        ('right_lower_arm', (6, 8, False)),
        ('left_lower_arm', (8, 10, False)),
        ('head', (10, 12, False))
    ])
    return pcloud_and_lgroup_from_ranges(pcloud, labels)


@labeller_func(group_label='pose_lsp_14')
def pose_lsp_14_to_pose_lsp_14(pcloud):
    r"""
    Apply the lsp 14-point semantic labels.

    The semantic labels applied are as follows:

      - left_leg
      - right_leg
      - left_arm
      - right_arm
      - head

    References
    ----------
    .. [1] http://www.comp.leeds.ac.uk/mat4saj/lsp.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 14
    validate_input(pcloud, n_expected_points)

    left_leg_indices = np.arange(0, 3)
    right_leg_indices = np.arange(3, 6)
    left_arm_indices = np.arange(6, 9)
    right_arm_indices = np.arange(9, 12)
    head_indices = np.arange(12, 14)

    left_leg_connectivity = connectivity_from_array(left_leg_indices)
    right_leg_connectivity = connectivity_from_array(right_leg_indices)
    left_arm_connectivity = connectivity_from_array(left_arm_indices)
    right_arm_connectivity = connectivity_from_array(right_arm_indices)
    head_connectivity = connectivity_from_array(head_indices)

    all_connectivity = np.vstack([
        left_leg_connectivity, right_leg_connectivity,
        left_arm_connectivity, right_arm_connectivity,
        head_connectivity
    ])

    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points,
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['left_leg'] = left_leg_indices
    mapping['right_leg'] = right_leg_indices
    mapping['left_arm'] = left_arm_indices
    mapping['right_arm'] = right_arm_indices
    mapping['head'] = head_indices

    return new_pcloud, mapping


@labeller_func(group_label='pose_flic_11')
def pose_flic_11_to_pose_flic_11(pcloud):
    r"""
    Apply the flic 11-point semantic labels.

    The semantic labels applied are as follows:

      - left_arm
      - right_arm
      - hips
      - face

    References
    ----------
    .. [1] http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC
    """
    n_expected_points = 11
    validate_input(pcloud, n_expected_points)

    labels = OrderedDict([
        ('left_arm', (0, 3, False)),
        ('right_arm', (3, 6, False)),
        ('hips', (6, 8, False)),
        ('face', (8, 11, True))])

    return pcloud_and_lgroup_from_ranges(pcloud, labels)


@labeller_func(group_label='pose_human36M_32')
def pose_human36M_32_to_pose_human36M_32(pcloud):
    r"""
    Apply the human3.6M 32-point semantic labels.

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

    References
    ----------
    .. [1] http://vision.imar.ro/human3.6m/
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 32
    validate_input(pcloud, n_expected_points)

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

    pelvis_connectivity = connectivity_from_array(pelvis_indices)
    right_leg_connectivity = connectivity_from_array(right_leg_indices)
    left_leg_connectivity = connectivity_from_array(left_leg_indices)
    spine_connectivity = connectivity_from_array(spine_indices)
    head_connectivity = connectivity_from_array(head_indices)
    left_arm_connectivity = connectivity_from_array(left_arm_indices)
    left_hand_connectivity = connectivity_from_array(left_hand_indices)
    right_arm_connectivity = connectivity_from_array(right_arm_indices)
    right_hand_connectivity = connectivity_from_array(right_hand_indices)
    torso_connectivity = connectivity_from_array(torso_indices,
                                                 close_loop=True)

    all_connectivity = np.vstack([
        pelvis_connectivity, right_leg_connectivity, left_leg_connectivity,
        spine_connectivity, head_connectivity, left_arm_connectivity,
        left_hand_connectivity, right_arm_connectivity,
        right_hand_connectivity, torso_connectivity
    ])

    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points,
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['pelvis'] = pelvis_indices
    mapping['right_leg'] = right_leg_indices
    mapping['left_leg'] = left_leg_indices
    mapping['spine'] = spine_indices
    mapping['head'] = head_indices
    mapping['left_arm'] = left_arm_indices
    mapping['left_hand'] = left_hand_indices
    mapping['right_arm'] = right_arm_indices
    mapping['right_hand'] = right_hand_indices
    mapping['torso'] = torso_indices

    return new_pcloud, mapping


@labeller_func(group_label='pose_human36M_17')
def pose_human36M_32_to_pose_human36M_17(pcloud):
    r"""
    Apply the human3.6M 17-point semantic labels (based on the
    original semantic labels of Human3.6 but removing the annotations
    corresponding to duplicate points, soles and palms), originally 32-points.

    The semantic labels applied are as follows:

      - pelvis
      - right_leg
      - left_leg
      - spine
      - head
      - left_arm
      - right_arm
      - torso

    References
    ----------
    .. [1] http://vision.imar.ro/human3.6m/
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 32
    validate_input(pcloud, n_expected_points)

    pelvis_indices = np.array([1, 0, 4])
    right_leg_indices = np.arange(1, 4)
    left_leg_indices = np.arange(4, 7)
    spine_indices = np.array([0, 7, 8])
    head_indices = np.array([8, 9, 10])
    left_arm_indices = np.array([8, 11, 12, 13])
    right_arm_indices = np.array([8, 14, 15, 16])
    torso_indices = np.array([0, 1, 14, 8, 11, 4])

    pelvis_connectivity = connectivity_from_array(pelvis_indices)
    right_leg_connectivity = connectivity_from_array(right_leg_indices)
    left_leg_connectivity = connectivity_from_array(left_leg_indices)
    spine_connectivity = connectivity_from_array(spine_indices)
    head_connectivity = connectivity_from_array(head_indices)
    left_arm_connectivity = connectivity_from_array(left_arm_indices)
    right_arm_connectivity = connectivity_from_array(right_arm_indices)
    torso_connectivity = connectivity_from_array(torso_indices,
                                                 close_loop=True)

    all_connectivity = np.vstack([
        pelvis_connectivity, right_leg_connectivity, left_leg_connectivity,
        spine_connectivity, head_connectivity, left_arm_connectivity,
        right_arm_connectivity, torso_connectivity
    ])

    # Ignore duplicate points, sole and palms
    ind = np.hstack([np.arange(0, 4), np.arange(6, 9), np.arange(12, 16),
                     np.arange(17, 20), np.arange(25, 28)])
    new_pcloud = PointUndirectedGraph.init_from_edges(
        pcloud.points[ind], all_connectivity)

    mapping = OrderedDict()
    mapping['pelvis'] = pelvis_indices
    mapping['right_leg'] = right_leg_indices
    mapping['left_leg'] = left_leg_indices
    mapping['spine'] = spine_indices
    mapping['head'] = head_indices
    mapping['left_arm'] = left_arm_indices
    mapping['right_arm'] = right_arm_indices
    mapping['torso'] = torso_indices

    return new_pcloud, mapping
