import numpy as np
from pybug.visualize import ImageViewer
from pybug.base import Flattenable


class Image(Flattenable):

    def __init__(self, image_data, mask=None):
        self.width, self.height, self.n_channels = image_data.shape
        if mask is not None:
            assert((self.width, self.height) == mask.shape)
            self.mask = mask.astype(np.bool).copy()
        else:
            self.mask = np.ones((self.width, self.height), dtype=np.bool)
        self.n_active_pixels = np.sum(self.mask)
        self.pixels = image_data.copy()

    def view(self):
        return ImageViewer(self.pixels)

    def as_flattened(self):
        return self.pixels[self.mask].flatten()

    @classmethod
    def from_flattened_with_instance(cls, flattened, instance, **kwargs):
        mask = instance.mask
        image_data = np.zeros_like(instance.pixels)
        pixels_per_channel = flattened.reshape((-1, instance.n_channels))
        image_data[mask] = pixels_per_channel
        return Image(image_data, mask=mask)
