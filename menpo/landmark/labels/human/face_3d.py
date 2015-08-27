from collections import OrderedDict
import numpy as np

from ..base import labeller_func, validate_input, connectivity_from_array


@labeller_func(group_label='face_bu3dfe_83')
def face_bu3dfe_83_to_face_bu3dfe_83(pcloud):
    r"""
    Apply the BU-3DFE (Binghamton University 3D Facial Expression)
    Database 83-point facial semantic labels.

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

    References
    ----------
    .. [1] http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html
    """
    from menpo.shape import PointUndirectedGraph

    n_expected_points = 83
    validate_input(pcloud, n_expected_points)

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

    reye_connectivity = connectivity_from_array(reye_indices, close_loop=True)
    leye_connectivity = connectivity_from_array(leye_indices, close_loop=True)
    rbrow_connectivity = connectivity_from_array(rbrow_indices,
                                                 close_loop=True)
    lbrow_connectivity = connectivity_from_array(lbrow_indices,
                                                 close_loop=True)
    rnose_connectivity = connectivity_from_array(rnose_indices)
    nostril_connectivity = connectivity_from_array(nostril_indices)
    lnose_connectivity = connectivity_from_array(lnose_indices)
    outermouth_connectivity = connectivity_from_array(outermouth_indices,
                                                      close_loop=True)
    innermouth_connectivity = connectivity_from_array(innermouth_indices,
                                                      close_loop=True)
    jaw_connectivity = connectivity_from_array(jaw_indices)

    all_connectivity = np.vstack([
        reye_connectivity, leye_connectivity,
        rbrow_connectivity, lbrow_connectivity,
        rnose_connectivity, nostril_connectivity, lnose_connectivity,
        outermouth_connectivity, innermouth_connectivity,
        jaw_connectivity
    ])

    new_pcloud = PointUndirectedGraph.init_from_edges(pcloud.points,
                                                      all_connectivity)

    mapping = OrderedDict()
    mapping['right_eye'] = reye_indices
    mapping['left_eye'] = leye_indices
    mapping['right_eyebrow'] = rbrow_indices
    mapping['left_eyebrow'] = lbrow_indices
    mapping['right_nose'] = rnose_indices
    mapping['left_nose'] = lnose_indices
    mapping['nostrils'] = nostril_indices
    mapping['outer_mouth'] = outermouth_indices
    mapping['inner_mouth'] = innermouth_indices
    mapping['jaw'] = jaw_indices

    return new_pcloud, mapping
