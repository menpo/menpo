from .base import Image
from pybug.shape import PointCloud


class LandmarkedImage(Image):

    def __init__(self, image_data, landmarks, labels):
        super(LandmarkedImage, self).__init__(image_data)
        self.points = PointCloud
        #TODO implementation of landmarks
