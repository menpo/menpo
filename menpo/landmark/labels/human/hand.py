import numpy as np
from menpo.landmark.base import LandmarkGroup

from menpo.landmark.labels.base import _validate_input, _connectivity_from_array


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

    new_landmark_group = LandmarkGroup.init_with_all_label(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points,
                                             total_conn))

    new_landmark_group['thumb'] = thumb_indices
    new_landmark_group['index'] = index_indices
    new_landmark_group['middle'] = middle_indices
    new_landmark_group['ring'] = ring_indices
    new_landmark_group['pinky'] = pinky_indices
    new_landmark_group['palm'] = palm_indices
    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group
