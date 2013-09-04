import abc
from pybug.io.base import Importer
from pybug.shape import PointCloud
import numpy as np
from pybug.transform.affine import Scale


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
        self.label = 'default'
        self.landmark_dict = {}

    def build(self, **kwargs):
        """
        Overrides the :meth:`build <pybug.io.base.Importer.build>` method.

        Parse the landmark format and return the label and landmark dictionary.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments to be passed through when parsing the landmarks

        Returns
        -------
        label : string
            The label that specifies what kind of landmarks were found
        landmark_dict : dict (string, :class:`pybug.shape.base.PointCloud`)
            A map from semantic labels to points that make up a set of
            landmarks.
        """
        self._parse_format(**kwargs)
        return self.label, self.landmark_dict

    @abc.abstractmethod
    def _parse_format(self, **kwargs):
        r"""
        Read the landmarks file from disk, parse it in to semantic labels and
        :class:`pybug.shape.base.PointCloud`.

        Set the ``self.label`` and ``self.landmark_dict`` attributes.
        """
        pass


class ASFImporter(LandmarkImporter):
    r"""
    Abstract base class for an importer for the ASF file format.
    Currently **does not support the connectivity specified in the format**.

    Implementations of this class should override the :meth:`_build_points`
    which determines the ordering of axes. For example, for images, the
    ``x`` and ``y`` axes are flipped such that the first axis is ``y`` (height
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
        ``x`` is the first axis, where as for images ``y`` is the first axis.
        """
        pass

    def _parse_format(self, scale_factors=np.array([1.0, 1.0]), **kwargs):
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
        scaled_points = Scale(np.array(scale_factors)).apply(points)

        # TODO: Use connectivity and create a graph type instead of PointCloud
        # edges = scaled_points[connectivity]

        self.label = 'ASF'
        self.landmark_dict = {'all': PointCloud(scaled_points)}


class PTSImporter(LandmarkImporter):
    r"""
    Importer for the PTS file format. Assumes version 1 of the format.

    Implementations of this class should override the :meth:`_build_points`
    which determines the ordering of axes. For example, for images, the
    ``x`` and ``y`` axes are flipped such that the first axis is ``y`` (height
    in the image domain).

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
        ``x`` is the first axis, where as for images ``y`` is the first axis.
        """
        pass

    def _parse_format(self, **kwargs):
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

        points = self._build_points(xs, ys)

        self.label = 'PTS'
        self.landmark_dict = {'all': PointCloud(points)}


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

    def _parse_format(self, **kwargs):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        # Remove comments and blank lines
        landmark_text = [l for l in landmarks.splitlines()
                         if (l.rstrip() and not '#' in l)]

        # First line says how many landmarks there are: 22 Landmarks
        # So pop it off the front
        num_points = int(landmark_text.pop(0).split()[0])
        points = []
        labels = []

        # The lines then alternate between the labels and the coordinates
        for i in xrange(num_points * 2):
            if i % 2 == 0:  # label
                labels.append(landmark_text[i])
            else:  # coordinate
                p = landmark_text[i].split()
                points.append(PointCloud(np.array([float(p[0]),
                                                  float(p[1]),
                                                  float(p[2])], ndmin=2)))

        self.label = 'LM3'
        self.landmark_dict = dict(zip(labels, points))


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

    def _parse_format(self, **kwargs):
        with open(self.filepath, 'r') as f:
            landmarks = np.fromfile(
                f, dtype=np.float32)[3:].reshape([-1, 3]).astype(np.double)
        self.label = 'LAN'
        self.landmark_dict = {'LAN' : PointCloud(landmarks)}
