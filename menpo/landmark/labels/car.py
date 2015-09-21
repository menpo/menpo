from collections import OrderedDict
import numpy as np

from .base import labeller_func, validate_input, connectivity_from_array


@labeller_func(group_label='car_streetscene_view_0_8')
def car_streetscene_20_to_car_streetscene_view_0_8(pcloud):
    r"""
    Apply the 8-point semantic labels of "view 0" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - front
      - bonnet
      - windshield

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    front_indices = np.array([0, 1, 3, 2])
    bonnet_indices = np.array([2, 3, 5, 4])
    windshield_indices = np.array([4, 5, 7, 6])

    front_connectivity = connectivity_from_array(front_indices,
                                                 close_loop=True)
    bonnet_connectivity = connectivity_from_array(bonnet_indices,
                                                  close_loop=True)
    windshield_connectivity = connectivity_from_array(windshield_indices,
                                                      close_loop=True)

    all_connectivity = np.vstack([front_connectivity, bonnet_connectivity,
                                  windshield_connectivity])

    ind = np.arange(8)
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['front'] = front_indices
    mapping['bonnet'] = bonnet_indices
    mapping['windshield'] = windshield_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_1_14')
def car_streetscene_20_to_car_streetscene_view_1_14(pcloud):
    """
    Apply the 14-point semantic labels of "view 1" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - front
      - bonnet
      - windshield
      - left_side

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    front_indices = np.array([0, 1, 3, 2])
    bonnet_indices = np.array([2, 3, 5, 4])
    windshield_indices = np.array([4, 5, 7, 6])
    left_side_indices = np.array([0, 2, 4, 6, 8, 9, 10, 11, 13, 12])

    front_connectivity = connectivity_from_array(front_indices,
                                                 close_loop=True)
    bonnet_connectivity = connectivity_from_array(bonnet_indices,
                                                  close_loop=True)
    windshield_connectivity = connectivity_from_array(windshield_indices,
                                                      close_loop=True)
    left_side_connectivity = connectivity_from_array(left_side_indices,
                                                     close_loop=True)

    all_connectivity = np.vstack([
        front_connectivity, bonnet_connectivity, windshield_connectivity,
        left_side_connectivity
    ])

    ind = np.hstack((np.arange(9), np.array([10, 12, 14, 16, 18])))
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['front'] = front_indices
    mapping['bonnet'] = bonnet_indices
    mapping['windshield'] = windshield_indices
    mapping['left_side'] = left_side_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_2_10')
def car_streetscene_20_to_car_streetscene_view_2_10(pcloud):
    r"""
    Apply the 10-point semantic labels of "view 2" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - left_side

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    left_side_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])

    left_side_connectivity = connectivity_from_array(left_side_indices,
                                                     close_loop=True)

    all_connectivity = left_side_connectivity

    ind = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['left_side'] = left_side_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_3_14')
def car_streetscene_20_to_car_streetscene_view_3_14(pcloud):
    r"""
    Apply the 14-point semantic labels of "view 3" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - left_side
      - rear windshield
      - trunk
      - rear

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    left_side_indices = np.array([0, 1, 2, 3, 4, 6, 8, 10, 13, 12])
    rear_windshield_indices = np.array([4, 5, 7, 6])
    trunk_indices = np.array([6, 7, 9, 8])
    rear_indices = np.array([8, 9, 11, 10])

    left_side_connectivity = connectivity_from_array(left_side_indices,
                                                     close_loop=True)
    rear_windshield_connectivity = connectivity_from_array(
        rear_windshield_indices, close_loop=True)
    trunk_connectivity = connectivity_from_array(trunk_indices, close_loop=True)
    rear_connectivity = connectivity_from_array(rear_indices, close_loop=True)

    all_connectivity = np.vstack([
        left_side_connectivity, rear_windshield_connectivity,
        trunk_connectivity, rear_connectivity
    ])

    ind = np.array([0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18])
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['left_side'] = left_side_indices
    mapping['rear_windshield'] = rear_windshield_indices
    mapping['trunk'] = trunk_indices
    mapping['rear'] = rear_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_4_14')
def car_streetscene_20_to_car_streetscene_view_4_14(pcloud):
    r"""
    Apply the 14-point semantic labels of "view 4" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - front
      - bonnet
      - windshield
      - right_side

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    front_indices = np.array([0, 1, 3, 2])
    bonnet_indices = np.array([2, 3, 5, 4])
    windshield_indices = np.array([4, 5, 7, 6])
    right_side_indices = np.array([8, 9, 10, 11, 13, 12, 1, 3, 5, 7])

    front_connectivity = connectivity_from_array(front_indices,
                                                 close_loop=True)
    bonnet_connectivity = connectivity_from_array(bonnet_indices,
                                                  close_loop=True)
    windshield_connectivity = connectivity_from_array(windshield_indices,
                                                      close_loop=True)
    right_side_connectivity = connectivity_from_array(right_side_indices,
                                                      close_loop=True)

    total_conn = np.vstack([
        front_connectivity, bonnet_connectivity, windshield_connectivity,
        right_side_connectivity
    ])

    ind = np.hstack([np.arange(8), np.array([9, 11, 13, 15, 17, 19])])
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      total_conn)

    mapping = OrderedDict()
    mapping['front'] = front_indices
    mapping['bonnet'] = bonnet_indices
    mapping['windshield'] = windshield_indices
    mapping['right_side'] = right_side_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_5_10')
def car_streetscene_20_to_car_streetscene_view_5_10(pcloud):
    r"""
    Apply the 10-point semantic labels of "view 5" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - right_side

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    right_side_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])

    right_side_connectivity = connectivity_from_array(right_side_indices,
                                                      close_loop=True)

    all_connectivity = right_side_connectivity

    ind = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['right_side'] = right_side_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_6_14')
def car_streetscene_20_to_car_streetscene_view_6_14(pcloud):
    r"""
    Apply the 14-point semantic labels of "view 6" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - right_side
      - rear_windshield
      - trunk
      - rear

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    right_side_indices = np.array([0, 1, 2, 3, 5, 7, 9, 11, 13, 12])
    rear_windshield_indices = np.array([4, 5, 7, 6])
    trunk_indices = np.array([6, 7, 9, 8])
    rear_indices = np.array([8, 9, 11, 10])

    right_side_connectivity = connectivity_from_array(right_side_indices,
                                                      close_loop=True)
    rear_windshield_connectivity = connectivity_from_array(
        rear_windshield_indices, close_loop=True)
    trunk_connectivity = connectivity_from_array(trunk_indices, close_loop=True)
    rear_connectivity = connectivity_from_array(rear_indices, close_loop=True)

    all_connectivity = np.vstack([
        right_side_connectivity, rear_windshield_connectivity,
        trunk_connectivity, rear_connectivity
    ])

    ind = np.array([1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19])
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['right_side'] = right_side_indices
    mapping['rear_windshield'] = rear_windshield_indices
    mapping['trunk'] = trunk_indices
    mapping['rear'] = rear_indices

    return new_pcloud, mapping


@labeller_func(group_label='car_streetscene_view_7_8')
def car_streetscene_20_to_car_streetscene_view_7_8(pcloud):
    r"""
    Apply the 8-point semantic labels of "view 7" from the MIT Street Scene
    Car dataset (originally a 20-point markup).

    The semantic labels applied are as follows:

      - rear_windshield
      - trunk
      - rear

    References
    ----------
    .. [1] http://www.cs.cmu.edu/~vboddeti/alignment.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 20
    validate_input(pcloud, n_expected_points)

    rear_windshield_indices = np.array([0, 1, 3, 2])
    trunk_indices = np.array([2, 3, 5, 4])
    rear_indices = np.array([4, 5, 7, 6])

    rear_windshield_connectivity = connectivity_from_array(
        rear_windshield_indices, close_loop=True)
    trunk_connectivity = connectivity_from_array(trunk_indices, close_loop=True)
    rear_connectivity = connectivity_from_array(rear_indices, close_loop=True)

    all_connectivity = np.vstack([rear_windshield_connectivity,
                                  trunk_connectivity, rear_connectivity])

    ind = np.arange(8, 16)
    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points[ind],
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['rear_windshield'] = rear_windshield_indices
    mapping['trunk'] = trunk_indices
    mapping['rear'] = rear_indices

    return new_pcloud, mapping
