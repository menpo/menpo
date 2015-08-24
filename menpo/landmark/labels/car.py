import numpy as np
from menpo.landmark.base import LandmarkGroup

from menpo.landmark.labels.base import _validate_input, _connectivity_from_array


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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    _validate_input(landmark_group, 20, group)

    left_side_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])

    left_side_connectivity = _connectivity_from_array(left_side_indices,
                                                      close_loop=True)

    total_conn = left_side_connectivity

    ind = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    _validate_input(landmark_group, 20, group)

    right_side_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])

    right_side_connectivity = _connectivity_from_array(right_side_indices,
                                                       close_loop=True)

    total_conn = right_side_connectivity

    ind = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

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
    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points[ind],
                                             total_conn))

    new_landmark_group['rear_windshield'] = rear_windshield_indices
    new_landmark_group['trunk'] = trunk_indices
    new_landmark_group['rear'] = rear_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group
