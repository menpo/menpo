from collections import OrderedDict
import numpy as np
from menpo.landmark.base import LandmarkGroup

from menpo.landmark.labels.base import _validate_input, _connectivity_from_array


def bu3dfe_83(landmark_group):
    """
    Apply the BU-3DFE (Binghamton University 3D Facial Expression)
    Database 83 point facial annotation markup to this landmark group.


    The group label will be ``bu3dfe_83``.

    The semantic labels applied are as follows:

      - right_eye
      - left_eye
      - right_eyebrow
      - left_eyebrow
      - right_nose
      - left_nose
      - nostrils
      - outer_mouth
      - inner_mouth
      - jaw

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to apply semantic labels to.

    Returns
    -------
    group : `str`
        The group label: ``bu3dfe_83``
    landmark_group : :map:`LandmarkGroup`
        New landmark group.

    Raises
    ------
    :class:`menpo.landmark.exceptions.LabellingError`
        If the given landmark group contains less than 83 points

    References
    ----------
    .. [1] http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html
    """
    from menpo.shape import PointUndirectedGraph

    group = 'bu3dfe_83'
    n_points = 83
    _validate_input(landmark_group, n_points, group)

    reye_indices = np.arange(0, 8)
    leye_indices = np.arange(8, 16)
    rbrow_indices = np.arange(16, 26)
    lbrow_indices = np.arange(26, 36)
    rnose_indices = np.arange(36, 39)
    lnose_indices = np.arange(39, 42)
    nostril_indices = np.arange(42, 48)
    outermouth_indices = np.arange(48, 60)
    innermouth_indices = np.arange(60, 68)
    jaw_indices = np.arange(68, 83)

    reye_connectivity = _connectivity_from_array(reye_indices, close_loop=True)
    leye_connectivity = _connectivity_from_array(leye_indices, close_loop=True)
    rbrow_connectivity = _connectivity_from_array(rbrow_indices,
                                                  close_loop=True)
    lbrow_connectivity = _connectivity_from_array(lbrow_indices,
                                                  close_loop=True)
    rnose_connectivity = _connectivity_from_array(rnose_indices)
    nostril_connectivity = _connectivity_from_array(nostril_indices)
    lnose_connectivity = _connectivity_from_array(lnose_indices)
    outermouth_connectivity = _connectivity_from_array(outermouth_indices,
                                                       close_loop=True)
    innermouth_connectivity = _connectivity_from_array(innermouth_indices,
                                                       close_loop=True)
    jaw_connectivity = _connectivity_from_array(jaw_indices)

    total_conn = np.vstack([
        reye_connectivity, leye_connectivity,
        rbrow_connectivity, lbrow_connectivity,
        rnose_connectivity, nostril_connectivity, lnose_connectivity,
        outermouth_connectivity, innermouth_connectivity,
        jaw_connectivity
    ])

    new_landmark_group = LandmarkGroup(
        PointUndirectedGraph.init_from_edges(landmark_group.lms.points,
                                             total_conn),
        OrderedDict([('all', np.ones(n_points, dtype=np.bool))]))

    new_landmark_group['right_eye'] = reye_indices
    new_landmark_group['left_eye'] = leye_indices
    new_landmark_group['right_eyebrow'] = rbrow_indices
    new_landmark_group['left_eyebrow'] = lbrow_indices
    new_landmark_group['right_nose'] = rnose_indices
    new_landmark_group['left_nose'] = lnose_indices
    new_landmark_group['nostrils'] = nostril_indices
    new_landmark_group['outer_mouth'] = outermouth_indices
    new_landmark_group['inner_mouth'] = innermouth_indices
    new_landmark_group['jaw'] = jaw_indices

    del new_landmark_group['all']  # Remove pointless all group

    return group, new_landmark_group
