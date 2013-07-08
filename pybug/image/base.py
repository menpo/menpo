from pybug.visualize import ImageViewer
from pybug.base import Flattenable


class Image(Flattenable):

    def __init__(self, image_data):
        self.width, self.height, self.n_channels = image_data.shape
        self.pixels = image_data

    def view(self):
        return ImageViewer(self.pixels)

    def as_flattened(self):
        return self.pixels.flatten()

    @classmethod
    def from_flattened_with_instance(cls, flattened, instance, **kwargs):
        #TODO fill out the  from_flattened method for images
        pass
