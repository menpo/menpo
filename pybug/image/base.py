import numpy as np
from pybug.visualize import ImageViewer
from pybug.base import Vectorizable


class Image(Vectorizable):

    def __init__(self, image_data, mask=None):
        self.width, self.height, self.n_channels = image_data.shape
        if mask is not None:
            assert((self.width, self.height) == mask.shape)
            self.mask = mask.astype(np.bool).copy()
        else:
            self.mask = np.ones((self.width, self.height), dtype=np.bool)
        self.n_masked_pixels = np.sum(self.mask)
        self.pixels = image_data.copy()

    def view(self):
        return ImageViewer(self.pixels)

    def as_vector(self):
        return self.masked_pixels.flatten()

    @property
    def masked_pixels(self):
        """
        :return: (n_active_pixels, n_channels) ndarray of pixels that have a
         True mask value
        """
        return self.pixels[self.mask]

    @property
    def masked_pixel_indices(self):
        y, x = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xy = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), 2)
        return xy[self.mask]

    def from_vector(self, flattened):
        mask = self.mask
        image_data = np.zeros_like(self.pixels)
        pixels_per_channel = flattened.reshape((-1, self.n_channels))
        image_data[mask] = pixels_per_channel
        return Image(image_data, mask=mask)
