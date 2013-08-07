import abc
from pybug.io.base import Importer
from pybug.shape import PointCloud
from pybug.shape.landmarks import LandmarkManager
import numpy as np


class LandmarkImporter(Importer):
    """
    Base class for importing landmarks
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(LandmarkImporter, self).__init__(filepath)

    def build(self):
        return self.label, self.landmark_dict


class ASFImporter(LandmarkImporter):

    def __init__(self, filepath):
        super(ASFImporter, self).__init__(filepath)
        self.label, self.landmark_dict = self.parse_asf(filepath)

    def parse_asf(self, filepath):
        with open(filepath, 'r') as f:
            landmarks = f.read()

        landmarks = [l for l in landmarks.splitlines()
                     if (l.rstrip() and not '#' in l)]

        # Pop the front of the list for the number of landmarks
        count = int(landmarks.pop(0))
        # Pop the last element of the list for the image_name
        image_name = landmarks.pop()

        points = np.empty([count, 2])
        tl = np.empty([count, 2], dtype=np.int)
        for i in xrange(count):
            # Though unpacked, they are still all strings
            path_num, path_type, xpos, ypos, point_num, connects_from, connects_to = landmarks[i].split()
            points[i, ...] = [float(xpos), float(ypos)]
            tl[i, ...] = [int(connects_from), int(connects_to)]

        scaled_points = np.empty_like(points)
        # TODO: Scale properly!
        scaled_points[..., 0] = points[..., 0] * 200 #* images[0].height
        scaled_points[..., 1] = points[..., 1] * 200 #* images[0].width

        return 'ASF', {'all': PointCloud(scaled_points)}

        # These are the edges
        # edges = scaled_points[tl]