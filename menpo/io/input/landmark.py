from collections import OrderedDict, namedtuple
import json
import warnings
import itertools

import numpy as np
from scipy.sparse import csr_matrix

from menpo.shape import PointCloud, LabelledPointUndirectedGraph
from menpo.transform import Scale


ASFPath = namedtuple('ASFPath', ['path_num', 'path_type', 'xpos', 'ypos',
                                 'point_num', 'connects_from', 'connects_to'])


def asf_importer(filepath, asset=None, **kwargs):
    r"""
    Importer for the ASF file format.

    For images, the `x` and `y` axes are flipped such that the first axis is
    `y` (height in the image domain).

    Currently only open and closed path types are supported.

    Landmark labels:

    +---------+
    | label   |
    +=========+
    | all     |
    +---------+

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : `dict` {`str`: :map:`PointCloud`}
        Dictionary mapping landmark groups to menpo shapes

    References
    ----------
    .. [1] http://www2.imm.dtu.dk/~aam/datasets/datasets.html
    """
    with filepath.open('r') as f:
        landmarks = f.read()

    # Remove comments and blank lines
    landmarks = [l for l in landmarks.splitlines()
                 if (l.rstrip() and not '#' in l)]

    # Pop the front of the list for the number of landmarks
    count = int(landmarks.pop(0))
    # Pop the last element of the list for the image_name
    image_name = landmarks.pop()

    xs = np.empty([count, 1])
    ys = np.empty([count, 1])
    connectivity = []

    # Only unpack the first 7 (the last 3 are always 0)
    split_landmarks = [ASFPath(*landmarks[i].split()[:7])
                       for i in range(count)]
    paths = [list(g)
             for k, g in itertools.groupby(split_landmarks, lambda x: x[0])]
    vert_index = 0
    for path in paths:
        if path:
            path_type = path[0].path_type
        for vertex in path:
            # Relative coordinates, will be scaled by the image size
            xs[vert_index, ...] = float(vertex.xpos)
            ys[vert_index, ...] = float(vertex.ypos)
            vert_index += 1
            # If True, isolated point
            if not (vertex.connects_from == vertex.connects_to and
                    vertex.connects_to == vertex.point_num):
                # Connectivity is defined by connects_from and connects_to
                # as well as the path_type:
                #   Bit 1: Outer edge point/Inside point
                #   Bit 2: Original annotated point/Artificial point
                #   Bit 3: Closed path point/Open path point
                #   Bit 4: Non-hole/Hole point
                # For now we only parse cases 0 and 4 (closed or open)
                connectivity.append((int(vertex.point_num),
                                     int(vertex.connects_to)))
        if path and path_type == '0':
            connectivity.append((int(path[-1].point_num),
                                 int(path[0].point_num)))

    connectivity = np.vstack(connectivity)
    points = np.hstack([ys, xs])
    if asset is not None:
        # we've been given an asset. As ASF files are normalized,
        # fix that here
        points = Scale(np.array(asset.shape)).apply(points)

    labels_to_masks = OrderedDict(
        [('all', np.ones(points.shape[0], dtype=np.bool))])
    return {'ASF': LabelledPointUndirectedGraph.init_from_edges(
                       points, connectivity, labels_to_masks)}


def pts_importer(filepath, image_origin=True, **kwargs):
    r"""
    Importer for the PTS file format. Assumes version 1 of the format.

    Implementations of this class should override the :meth:`_build_points`
    which determines the ordering of axes. For example, for images, the
    `x` and `y` axes are flipped such that the first axis is `y` (height
    in the image domain).

    Note that PTS has a very loose format definition. Here we make the
    assumption (as is common) that PTS landmarks are 1-based. That is,
    landmarks on a 480x480 image are in the range [1-480]. As Menpo is
    consistently 0-based, we *subtract 1* off each landmark value
    automatically.

    If you want to use PTS landmarks that are 0-based, you will have to
    manually add one back on to landmarks post importing.

    Landmark set label: PTS

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    image_origin : `bool`, optional
        If ``True``, assume that the landmarks exist within an image and thus
        the origin is the image origin.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : `dict` {`str`: :map:`PointCloud`}
        Dictionary mapping landmark groups to menpo shapes
    """
    with filepath.open('r') as f:
        lines = [l.strip() for l in f.readlines()]

    line = lines[0]
    while not line.startswith('{'):
        line = lines.pop(0)

    xs = []
    ys = []
    for line in lines:
        if not line.strip().startswith('}'):
            xpos, ypos = line.split()[:2]
            xs.append(xpos)
            ys.append(ypos)

    xs = np.array(xs, dtype=np.float).reshape((-1, 1))
    ys = np.array(ys, dtype=np.float).reshape((-1, 1))

    # PTS landmarks are 1-based, need to convert to 0-based (subtract 1)
    if image_origin:
        points = np.hstack([ys - 1, xs - 1])
    else:
        points = np.hstack([xs - 1, ys - 1])

    return {'PTS': PointCloud(points, copy=False)}


def lm2_importer(filepath, **kwargs):
    r"""
    Importer for the LM2 file format from the bosphorus dataset. This is a 2D
    landmark type and so it is assumed it only applies to images.

    Landmark set label: LM2

    Landmark labels:

    +------------------------+
    | label                  |
    +========================+
    | outer_left_eyebrow     |
    | middle_left_eyebrow    |
    | inner_left_eyebrow     |
    | inner_right_eyebrow    |
    | middle_right_eyebrow   |
    | outer_right_eyebrow    |
    | outer_left_eye_corner  |
    | inner_left_eye_corner  |
    | inner_right_eye_corner |
    | outer_right_eye_corner |
    | nose_saddle_left       |
    | nose_saddle_right      |
    | left_nose_peak         |
    | nose_tip               |
    | right_nose_peak        |
    | left_mouth_corner      |
    | upper_lip_outer_middle |
    | right_mouth_corner     |
    | upper_lip_inner_middle |
    | lower_lip_inner_middle |
    | lower_lip_outer_middle |
    | chin_middle            |
    +------------------------+

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : `dict` {`str`: :map:`PointCloud`}
        Dictionary mapping landmark groups to menpo shapes
    """
    with filepath.open('r') as f:
        landmarks = f.read()

    # Remove comments and blank lines
    landmark_text = [l for l in landmarks.splitlines()
                     if (l.rstrip() and '#' not in l)]

    # First line says how many landmarks there are: 22 Landmarks
    # So pop it off the front
    num_points = int(landmark_text.pop(0).split()[0])
    labels = []

    # The next set of lines defines the labels
    labels_str = landmark_text.pop(0)
    if not labels_str == 'Labels:':
        raise ValueError("LM2 landmarks are incorrectly formatted. "
                         "Expected a list of labels beginning with "
                         "'Labels:' but found '{0}'".format(labels_str))
    for i in range(num_points):
        # Lowercase, remove spaces and replace with underscores
        l = landmark_text.pop(0)
        l = '_'.join(l.lower().split())
        labels.append(l)

    # The next set of lines defines the coordinates
    coords_str = landmark_text.pop(0)
    if not coords_str == '2D Image coordinates:':
        raise ValueError("LM2 landmarks are incorrectly formatted. "
                         "Expected a list of coordinates beginning with "
                         "'2D Image coordinates:' "
                         "but found '{0}'".format(coords_str))
    xs = []
    ys = []
    for i in range(num_points):
        p = landmark_text.pop(0).split()
        xs.append(float(p[0]))
        ys.append(float(p[1]))

    xs = np.array(xs, dtype=np.float).reshape((-1, 1))
    ys = np.array(ys, dtype=np.float).reshape((-1, 1))

    # Flip the x and y
    points = np.hstack([ys, xs])
    # Create the mask whereby there is one landmark per label
    # (identity matrix)
    masks = np.eye(num_points).astype(np.bool)
    masks = np.vsplit(masks, num_points)
    masks = [np.squeeze(m) for m in masks]
    labels_to_masks = OrderedDict(zip(labels, masks))

    empty_adj_matrix = csr_matrix((num_points, num_points))
    return {'LM2': LabelledPointUndirectedGraph(points, empty_adj_matrix,
                                                labels_to_masks)}


def _ljson_parse_null_values(points_list):
    filtered_points = [np.nan if x is None else x
                       for x in itertools.chain(*points_list)]
    return np.array(filtered_points,
                    dtype=np.float).reshape([-1, len(points_list[0])])


def _parse_ljson_v1(lms_dict):
    all_points = []
    labels = []  # label per group
    labels_slices = []  # slices into the full pointcloud per label
    offset = 0
    connectivity = []
    for group in lms_dict['groups']:
        lms = group['landmarks']
        labels.append(group['label'])
        labels_slices.append(slice(offset, len(lms) + offset))
        # Create the connectivity if it exists
        conn = group.get('connectivity', [])
        if conn:
            # Offset relative connectivity according to the current index
            conn = offset + np.asarray(conn)
            connectivity += conn.tolist()
        for p in lms:
            all_points.append(p['point'])
        offset += len(lms)

    # Don't create a PointUndirectedGraph with no connectivity
    points = _ljson_parse_null_values(all_points)
    n_points = points.shape[0]

    labels_to_masks = OrderedDict()
    # go through each label and build the appropriate boolean array
    for label, l_slice in zip(labels, labels_slices):
        mask = np.zeros(n_points, dtype=np.bool)
        mask[l_slice] = True
        labels_to_masks[label] = mask

    lmarks = LabelledPointUndirectedGraph.init_from_edges(points, connectivity,
                                                          labels_to_masks)
    return {'LJSON': lmarks}


def _parse_ljson_v2(lms_dict):
    points = _ljson_parse_null_values(lms_dict['landmarks']['points'])
    connectivity = lms_dict['landmarks'].get('connectivity')

    if connectivity is None and len(lms_dict['labels']) == 0:
        lmarks = PointCloud(points)
    else:
        labels_to_mask = OrderedDict()  # masks into the pointcloud per label
        n_points = points.shape[0]
        for label in lms_dict['labels']:
            mask = np.zeros(n_points, dtype=np.bool)
            mask[label['mask']] = True
            labels_to_mask[label['label']] = mask
        # Note that we can pass connectivity as None here and the edges will be
        # empty.
        lmarks = LabelledPointUndirectedGraph.init_from_edges(
            points, connectivity, labels_to_mask)

    return {'LJSON': lmarks}


def _parse_ljson_v3(lms_dict):
    all_lms = {}
    for key, lms_dict_group in lms_dict['groups'].items():
        points = _ljson_parse_null_values(lms_dict_group['landmarks']['points'])
        connectivity = lms_dict_group['landmarks'].get('connectivity')
        # TODO: create the metadata label!

        if connectivity is None and len(lms_dict_group['labels']) == 0:
            all_lms[key] = PointCloud(points)
        else:
            # masks into the pointcloud per label
            labels_to_mask = OrderedDict()
            n_points = points.shape[0]
            for label in lms_dict_group['labels']:
                mask = np.zeros(n_points, dtype=np.bool)
                mask[label['mask']] = True
                labels_to_mask[label['label']] = mask

            # Note that we can pass connectivity as None here and the edges
            # will be empty.
            all_lms[key] = LabelledPointUndirectedGraph.init_from_edges(
                    points, connectivity, labels_to_mask)
    return all_lms


_ljson_parser_for_version = {
    1: _parse_ljson_v1,
    2: _parse_ljson_v2,
    3: _parse_ljson_v3
}


def ljson_importer(filepath, **kwargs):
    r"""
    Importer for the Menpo JSON format. This is an n-dimensional
    landmark type for both images and meshes that encodes semantic labels in
    the format.

    Landmark set label (v1, v2): JSON
    Landmark set label (v3): As defined in the file

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : `dict` {`str`: :map:`PointCloud`}
        Dictionary mapping landmark groups to menpo shapes
    """
    with filepath.open('r') as f:
        lms_dict = json.load(f, object_pairs_hook=OrderedDict)
    version = lms_dict.get('version')
    parser = _ljson_parser_for_version.get(version)

    if parser is None:
        raise ValueError("{} has unknown version {} - must be "
                         "1, or 2 or 3.".format(filepath, version))
    if version != 3:
        from menpo.base import MenpoDeprecationWarning
        warnings.warn('LJSON v{} is deprecated. export_landmark_file() will '
                      'only save out LJSON v3 files. Please convert all LJSON '
                      'files to v3 by importing into Menpo and re-exporting to '
                      'overwrite the files.'.format(version),
                      MenpoDeprecationWarning)

    return parser(lms_dict)
