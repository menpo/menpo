from collections import OrderedDict
import numpy as np

from ..base import validate_input, connectivity_from_array, labeller_func


@labeller_func(group_label='hand_ibug_39')
def hand_ibug_39_to_hand_ibug_39(pcloud):
    r"""
    Apply the IBUG 39-point semantic labels.

    The semantic labels applied are as follows:

      - thumb
      - index
      - middle
      - ring
      - pinky
      - palm
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 39
    validate_input(pcloud, n_expected_points)

    thumb_indices = np.arange(0, 5)
    index_indices = np.arange(5, 12)
    middle_indices = np.arange(12, 19)
    ring_indices = np.arange(19, 26)
    pinky_indices = np.arange(26, 33)
    palm_indices = np.hstack((np.array([32, 25, 18, 11, 33, 34, 4]),
                              np.arange(35, 39)))

    thumb_connectivity = connectivity_from_array(thumb_indices,
                                                 close_loop=False)
    index_connectivity = connectivity_from_array(index_indices,
                                                 close_loop=False)
    middle_connectivity = connectivity_from_array(middle_indices,
                                                  close_loop=False)
    ring_connectivity = connectivity_from_array(ring_indices,
                                                close_loop=False)
    pinky_connectivity = connectivity_from_array(pinky_indices,
                                                 close_loop=False)
    palm_connectivity = connectivity_from_array(palm_indices,
                                                close_loop=True)

    all_connectivity = np.vstack([thumb_connectivity, index_connectivity,
                                  middle_connectivity, ring_connectivity,
                                  pinky_connectivity, palm_connectivity])

    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points,
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['thumb'] = thumb_indices
    mapping['index'] = index_indices
    mapping['middle'] = middle_indices
    mapping['ring'] = ring_indices
    mapping['pinky'] = pinky_indices
    mapping['palm'] = palm_indices

    return new_pcloud, mapping
