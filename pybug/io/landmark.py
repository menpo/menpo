import abc
from pybug.io.base import Importer
from pybug.shape import PointCloud
import numpy as np
from pybug.transform.affine import Scale


class LandmarkImporter(Importer):
    """
    Base class for importing landmarks
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(LandmarkImporter, self).__init__(filepath)
        self.label = 'default'
        self.landmark_dict = {}

    def build(self, **kwargs):
        self._parse_format(**kwargs)
        return self.label, self.landmark_dict

    @abc.abstractmethod
    def _parse_format(self, **kwargs):
        pass


class ASFImporter(LandmarkImporter):

    def __init__(self, filepath):
        super(ASFImporter, self).__init__(filepath)

    def _parse_format(self, scale_factors=np.array([1, 1]), **kwargs):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        landmarks = [l for l in landmarks.splitlines()
                     if (l.rstrip() and not '#' in l)]

        # Pop the front of the list for the number of landmarks
        count = int(landmarks.pop(0))
        # Pop the last element of the list for the image_name
        image_name = landmarks.pop()

        points = np.empty([count, 2])
        connectivity = np.empty([count, 2], dtype=np.int)
        for i in xrange(count):
            # Though unpacked, they are still all strings
            # Only unpack the first 7
            (path_num, path_type, xpos, ypos,
             point_num, connects_from, connects_to) = landmarks[i].split()[:7]
            points[i, ...] = [float(ypos), float(xpos)]
            connectivity[i, ...] = [int(connects_from), int(connects_to)]

        scaled_points = Scale(np.array(scale_factors)).apply(points)

        # TODO: Use connectivity and create a graph type instead of PointCloud
        # edges = scaled_points[connectivity]

        self.label = 'ASF'
        self.landmark_dict = {'all': PointCloud(scaled_points)}
