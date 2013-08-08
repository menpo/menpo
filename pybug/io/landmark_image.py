from pybug.io.landmark import ASFImporter
import numpy as np


class ImageASFImporter(ASFImporter):

    def __init__(self, filepath):
        super(ImageASFImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For images, axis 0 = ys and axis 1 = xs. Therefore, return the
        appropriate points array ordering
        :param xs: Row vector of x coordinates
        :param ys: Row vector of y coordinates
        :return: 2D array of [ys; xs]
        """
        return np.hstack([ys, xs])