import abc
import json
from collections import OrderedDict

import numpy as np

from menpo.io.base import Importer
from menpo.landmark.base import LandmarkGroup
from menpo.shape import PointCloud
from menpo.transform import Scale


class LandmarkImporter(Importer):
    """
    Abstract base class for importing landmarks.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(LandmarkImporter, self).__init__(filepath)
        self.group_label = 'default'
        self.pointcloud = None
        self.labels_to_masks = None

    def build(self, asset=None):
        """
        Overrides the :meth:`build <menpo.io.base.Importer.build>` method.

        Parse the landmark format and return the label and landmark dictionary.

        Parameters
        ----------
        asset : object, optional
            The asset that the landmarks are being built for. Can be used to
            adjust landmarks as necessary (e.g. rescaling image landmarks
            from 0-1 to image.shape)

        Returns
        -------
        landmark_group : string
            The landmark group parsed from the file.
            Every point will be labelled.
        """
        self._parse_format(asset=asset)
        return LandmarkGroup(None, self.group_label, self.pointcloud,
                             self.labels_to_masks)

    @abc.abstractmethod
    def _parse_format(self, asset=None):
        r"""
        Read the landmarks file from disk, parse it in to semantic labels and
        :class:`menpo.shape.base.PointCloud`.

        Set the `self.label` and `self.pointcloud` attributes.
        """
        pass


def _indices_to_mask(n_points, indices):
    """
    Helper function to turn an array of indices in to a boolean mask.

    Parameters
    ----------
    n_points : int
        The total number of points for the mask
    indices : ndarray of ints
        An array of integers representing the `True` indices.

    Returns
    -------
    boolean_mask : ndarray of bools
        The mask for the set of landmarks where each index from indices is set
        to `True` and the rest are `False`
    """
    mask = np.zeros(n_points, dtype=np.bool)
    mask[indices] = True
    return mask


class ASFImporter(LandmarkImporter):
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
    filepath : string
        Absolute filepath to landmark file.

    References
    ----------
    .. [1] http://www2.imm.dtu.dk/~aam/datasets/datasets.html
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(ASFImporter, self).__init__(filepath)

    @abc.abstractmethod
    def _build_points(self, xs, ys):
        r"""
        Determines the ordering of points within the landmarks. For meshes
        `x` is the first axis, where as for images `y` is the first axis.
        """
        pass

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
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
        for i in xrange(count):
            # Though unpacked, they are still all strings
            # Only unpack the first 7
            (path_num, path_type, xpos, ypos,
             point_num, connects_from, connects_to) = landmarks[i].split()[:7]
            xs[i, ...] = float(xpos)
            ys[i, ...] = float(ypos)
            connectivity[i, ...] = [int(connects_from), int(connects_to)]

        points = self._build_points(xs, ys)
        if asset is not None:
            # we've been given an asset. As ASF files are normalized,
            # fix that here
            points = Scale(np.array(asset.shape)).apply(points)

        # TODO: Use connectivity and create a graph type instead of PointCloud
        # edges = scaled_points[connectivity]

        self.group_label = 'ASF'
        self.pointcloud = PointCloud(points)
        self.labels_to_masks = {'all': np.ones(points.shape[0], dtype=np.bool)}


class PTSImporter(LandmarkImporter):
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
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(PTSImporter, self).__init__(filepath)

    @abc.abstractmethod
    def _build_points(self, xs, ys):
        r"""
        Determines the ordering of points within the landmarks. For meshes
        `x` is the first axis, where as for images `y` is the first axis.
        """
        pass

    def _parse_format(self, asset=None):
        f = open(self.filepath, 'r')
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
        points = self._build_points(xs - 1, ys - 1)

        self.group_label = 'PTS'
        self.pointcloud = PointCloud(points)
        self.labels_to_masks = {'all': np.ones(points.shape[0], dtype=np.bool)}


class LM3Importer(LandmarkImporter):
    r"""
    Importer for the LM3 file format from the bosphorus dataset. This is a 3D
    landmark type and so it is assumed it only applies to meshes.

    Landmark set label: LM3

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
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(LM3Importer, self).__init__(filepath)

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        # Remove comments and blank lines
        landmark_text = [l for l in landmarks.splitlines()
                         if (l.rstrip() and not '#' in l)]

        # First line says how many landmarks there are: 22 Landmarks
        # So pop it off the front
        num_points = int(landmark_text.pop(0).split()[0])
        xs = []
        ys = []
        zs = []
        labels = []

        # The lines then alternate between the labels and the coordinates
        for i in xrange(num_points * 2):
            if i % 2 == 0:  # label
                # Lowercase, remove spaces and replace with underscores
                l = landmark_text[i]
                l = '_'.join(l.lower().split())
                labels.append(l)
            else:  # coordinate
                p = landmark_text[i].split()
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                zs.append(float(p[2]))

        xs = np.array(xs, dtype=np.float).reshape((-1, 1))
        ys = np.array(ys, dtype=np.float).reshape((-1, 1))
        zs = np.array(zs, dtype=np.float).reshape((-1, 1))

        self.group_label = 'LM3'
        self.pointcloud = PointCloud(np.hstack([xs, ys, zs]))
        # Create the mask whereby there is one landmark per label
        # (identity matrix)
        masks = np.eye(num_points).astype(np.bool)
        masks = np.vsplit(masks, num_points)
        masks = [np.squeeze(m) for m in masks]
        self.labels_to_masks = dict(zip(labels, masks))


class LM2Importer(LandmarkImporter):
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
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(LM2Importer, self).__init__(filepath)

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        # Remove comments and blank lines
        landmark_text = [l for l in landmarks.splitlines()
                         if (l.rstrip() and not '#' in l)]

        # First line says how many landmarks there are: 22 Landmarks
        # So pop it off the front
        num_points = int(landmark_text.pop(0).split()[0])
        labels = []

        # The next set of lines defines the labels
        labels_str = landmark_text.pop(0)
        if not labels_str == 'Labels:':
            raise ImportError("LM2 landmarks are incorrectly formatted. "
                              "Expected a list of labels beginning with "
                              "'Labels:' but found '{0}'".format(labels_str))
        for i in xrange(num_points):
            # Lowercase, remove spaces and replace with underscores
            l = landmark_text.pop(0)
            l = '_'.join(l.lower().split())
            labels.append(l)

        # The next set of lines defines the coordinates
        coords_str = landmark_text.pop(0)
        if not coords_str == '2D Image coordinates:':
            raise ImportError("LM2 landmarks are incorrectly formatted. "
                              "Expected a list of coordinates beginning with "
                              "'2D Image coordinates:' "
                              "but found '{0}'".format(coords_str))
        xs = []
        ys = []
        for i in xrange(num_points):
            p = landmark_text.pop(0).split()
            xs.append(float(p[0]))
            ys.append(float(p[1]))

        xs = np.array(xs, dtype=np.float).reshape((-1, 1))
        ys = np.array(ys, dtype=np.float).reshape((-1, 1))

        self.group_label = 'LM2'
        # Flip the x and y
        self.pointcloud = PointCloud(np.hstack([ys, xs]))
        # Create the mask whereby there is one landmark per label
        # (identity matrix)
        masks = np.eye(num_points).astype(np.bool)
        masks = np.vsplit(masks, num_points)
        masks = [np.squeeze(m) for m in masks]
        self.labels_to_masks = dict(zip(labels, masks))


class LANImporter(LandmarkImporter):
    r"""
    Importer for the LAN file format for the GOSH dataset. This is a 3D
    landmark type and so it is assumed it only applies to meshes.

    Landmark set label: LAN

    Note that the exact meaning of each landmark in this set varies,
    so all we can do is import all landmarks found under the label 'LAN'

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(LANImporter, self).__init__(filepath)

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = np.fromfile(
                f, dtype=np.float32)[3:].reshape([-1, 3]).astype(np.double)

        self.group_label = 'LM3'
        self.pointcloud = PointCloud(landmarks)
        self.labels_to_masks = {'all': np.ones(landmarks.shape[0],
                                               dtype=np.bool)}


class BNDImporter(LandmarkImporter):
    r"""
    Importer for the BND file format for the BU-3DFE dataset. This is a 3D
    landmark type and so it is assumed it only applies to meshes.

    Landmark set label: BND

    Landmark labels:

    +---------------+
    | label         |
    +===============+
    | left_eye      |
    | right_eye     |
    | left_eyebrow  |
    | right_eyebrow |
    | nose          |
    | mouth         |
    | chin          |
    +---------------+
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(BNDImporter, self).__init__(filepath)

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        # Remove blank lines
        landmark_text = [l for l in landmarks.splitlines() if l.rstrip()]
        landmark_text = [l.split() for l in landmark_text]

        n_points = len(landmark_text)
        landmarks = np.zeros([n_points, 3])
        for i, l in enumerate(landmark_text):
            # Skip the first number as it's an index into the mesh
            landmarks[i, :] = np.array([float(l[1]), float(l[2]), float(l[3])],
                                       dtype=np.float)

        self.group_label = 'BND'
        self.pointcloud = PointCloud(landmarks)
        self.labels_to_masks = {
            'left_eye': _indices_to_mask(n_points, np.arange(8)),
            'right_eye': _indices_to_mask(n_points, np.arange(8, 16)),
            'left_eyebrow': _indices_to_mask(n_points, np.arange(16, 26)),
            'right_eyebrow': _indices_to_mask(n_points, np.arange(26, 36)),
            'nose': _indices_to_mask(n_points, np.arange(36, 48)),
            'mouth': _indices_to_mask(n_points, np.arange(48, 68)),
            'chin': _indices_to_mask(n_points, np.arange(68, 83))
        }


class JSONImporter(LandmarkImporter):
    r"""
    Importer for the Menpo JSON format. This is an nD
    landmark type for both images and meshes that encodes semantic labels in
    the format

    Landmark set label: JSON

    Landmark labels: decided by file

    """
    def _parse_format(self, asset=None):
        with open(self.filepath, 'rb') as f:
            lms_dict = json.load(f)  # lms_dict is now a dict rep the JSON
        self.group_label = 'JSON'
        all_points = []
        labels = []  # label per group
        labels_slices = []  # slices into the full pointcloud per label
        start = 0
        for group in lms_dict['groups']:
            lms = group['landmarks']
            labels.append(group['label'])
            labels_slices.append(slice(start, len(lms) + start))
            start = len(lms) + start
            for p in lms:
                all_points.append(p['point'])
        self.pointcloud = PointCloud(np.array(all_points))
        self.labels_to_masks = {}
        # go through each label and build the appropriate boolean array
        for label, l_slice in zip(labels, labels_slices):
            mask = np.zeros(self.pointcloud.n_points, dtype=np.bool)
            mask[l_slice] = True
            self.labels_to_masks[label] = mask
