from collections import OrderedDict
import json
import warnings
import itertools

import numpy as np

from menpo.landmark.base import LandmarkGroup
from menpo.shape import PointCloud, PointUndirectedGraph
from menpo.transform import Scale


def asf_importer(filepath, asset=None, image_origin=True, **kwargs):
    r"""
    Abstract base class for an importer for the ASF file format.
    Currently **does not support the connectivity specified in the format**.

    Implementations of this class should override the :meth:`_build_points`
    which determines the ordering of axes. For example, for images, the
    `x` and `y` axes are flipped such that the first axis is `y` (height
    in the image domain).

    Landmark set label: ASF

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
    image_origin : `bool`, optional
        If ``True``, assume that the landmarks exist within an image and thus
        the origin is the image origin.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : :map:`LandmarkGroup`
        The landmarks including appropriate labels if available.

    References
    ----------
    .. [1] http://www2.imm.dtu.dk/~aam/datasets/datasets.html
    """
    with open(str(filepath), 'r') as f:
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
    connectivity = np.empty([count, 2], dtype=np.int)
    for i in range(count):
        # Though unpacked, they are still all strings
        # Only unpack the first 7
        (path_num, path_type, xpos, ypos,
         point_num, connects_from, connects_to) = landmarks[i].split()[:7]
        xs[i, ...] = float(xpos)
        ys[i, ...] = float(ypos)
        connectivity[i, ...] = [int(connects_from), int(connects_to)]

    if image_origin:
        points = np.hstack([ys, xs])
    else:
        points = np.hstack([xs, ys])
    if asset is not None:
        # we've been given an asset. As ASF files are normalized,
        # fix that here
        points = Scale(np.array(asset.shape)).apply(points)

    # TODO: Use connectivity and create a graph type instead of PointCloud
    # edges = scaled_points[connectivity]

    return LandmarkGroup(PointCloud(points, copy=False),
                         OrderedDict([('all', np.ones(points.shape[0],
                                                      dtype=np.bool))]))


def pts_importer(filepath, asset=None, image_origin=True, **kwargs):
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
    image_origin : `bool`, optional
        If ``True``, assume that the landmarks exist within an image and thus
        the origin is the image origin.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : :map:`LandmarkGroup`
        The landmarks including appropriate labels if available.
    """
    f = open(str(filepath), 'r')
    for line in f:
        if line.split()[0] == '{':
            break
    xs = []
    ys = []
    for line in f:
        if line.split()[0] != '}':
            xpos, ypos = line.split()[0:2]
            xs.append(xpos)
            ys.append(ypos)
    xs = np.array(xs, dtype=np.float).reshape((-1, 1))
    ys = np.array(ys, dtype=np.float).reshape((-1, 1))
    # PTS landmarks are 1-based, need to convert to 0-based (subtract 1)
    if image_origin:
        points = np.hstack([ys - 1, xs - 1])
    else:
        points = np.hstack([xs - 1, ys - 1])

    return LandmarkGroup(PointCloud(points, copy=False),
                         OrderedDict([('all', np.ones(points.shape[0],
                                                      dtype=np.bool))]))


def lm2_importer(filepath, asset=None, **kwargs):
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
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    landmarks : :map:`LandmarkGroup`
        The landmarks including appropriate labels if available.
    """
    with open(str(filepath), 'r') as f:
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
    pointcloud = PointCloud(np.hstack([ys, xs]), copy=False)
    # Create the mask whereby there is one landmark per label
    # (identity matrix)
    masks = np.eye(num_points).astype(np.bool)
    masks = np.vsplit(masks, num_points)
    masks = [np.squeeze(m) for m in masks]
    labels_to_masks = OrderedDict(zip(labels, masks))

    return LandmarkGroup(pointcloud, labels_to_masks)


def _ljson_parse_null_values(points_list):
    filtered_points = [np.nan if x is None else x
                       for x in itertools.chain(*points_list)]
    return np.array(filtered_points,
                    dtype=np.float).reshape([-1, len(points_list[0])])


def _parse_ljson_v1(lms_dict):
    from menpo.base import MenpoDeprecationWarning
    warnings.warn('LJSON v1 is deprecated. export_landmark_file{s}() will '
                  'only save out LJSON v2 files. Please convert all LJSON '
                  'files to v2 by importing into Menpo and re-exporting to '
                  'overwrite the files.', MenpoDeprecationWarning)
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
    if len(connectivity) == 0:
        pcloud = PointCloud(points)
    else:
        pcloud = PointUndirectedGraph.init_from_edges(points, connectivity)
    labels_to_masks = OrderedDict()
    # go through each label and build the appropriate boolean array
    for label, l_slice in zip(labels, labels_slices):
        mask = np.zeros(pcloud.n_points, dtype=np.bool)
        mask[l_slice] = True
        labels_to_masks[label] = mask
    return pcloud, labels_to_masks


def _parse_ljson_v2(lms_dict):
    labels_to_mask = OrderedDict()  # masks into the full pointcloud per label

    points = _ljson_parse_null_values(lms_dict['landmarks']['points'])
    connectivity = lms_dict['landmarks'].get('connectivity')

    # Don't create a PointUndirectedGraph with no connectivity
    if connectivity is None or len(connectivity) == 0:
        pcloud = PointCloud(points)
    else:
        pcloud = PointUndirectedGraph.init_from_edges(points, connectivity)

    for label in lms_dict['labels']:
        mask = np.zeros(pcloud.n_points, dtype=np.bool)
        mask[label['mask']] = True
        labels_to_mask[label['label']] = mask

    return pcloud, labels_to_mask


_ljson_parser_for_version = {
    1: _parse_ljson_v1,
    2: _parse_ljson_v2
}


def ljson_importer(filepath, asset=None, **kwargs):
    r"""
    Importer for the Menpo JSON format. This is an n-dimensional
    landmark type for both images and meshes that encodes semantic labels in
    the format.

    Landmark set label: JSON

    Landmark labels: decided by file

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
    landmarks : :map:`LandmarkGroup`
        The landmarks including appropriate labels if available.
    """
    with open(str(filepath), 'r') as f:
        # lms_dict is now a dict rep of the JSON
        lms_dict = json.load(f, object_pairs_hook=OrderedDict)
    v = lms_dict.get('version')
    parser = _ljson_parser_for_version.get(v)

    if parser is None:
        raise ValueError("{} has unknown version {} must be "
                         "1, or 2".format(self.filepath, v))
    return LandmarkGroup(*parser(lms_dict))
