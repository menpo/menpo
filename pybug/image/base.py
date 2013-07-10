import numpy as np
import PIL.Image as PILImage
from pybug.visualize import ImageViewer
from pybug.base import Vectorizable


class Image(Vectorizable):

    def __init__(self, image_data, mask=None):
        # we support construction from a PIL Image class
        if isinstance(image_data, PILImage.Image):
            image_data = np.array(image_data)
        # get the attributes of the image
        self.width, self.height = image_data.shape[:2]
        if len(image_data.shape) == 3:
            self.n_channels = image_data.shape[2]
        else:
            self.n_channels = 1
            # ensure all our self.pixels are 3 dimensional, even if the last
            # dim is unitary in length
            image_data = image_data[..., np.newaxis]
        if mask is not None:
            assert((self.width, self.height) == mask.shape)
            self.mask = mask.astype(np.bool).copy()
        else:
            self.mask = np.ones((self.width, self.height), dtype=np.bool)
        self.n_masked_pixels = np.sum(self.mask)

        # ensure that the data is in the right format
        if image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float64) / 255
        elif image_data.dtype != np.float64:
            # convert to double
            image_data = image_data.astype(np.float64)
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

    def as_greyscale(self):
        if self.n_channels == 1:
            print "Warning - trying to convert to greyscale an image with " \
                  "only one channel - returning a copy"
            return Image(self.pixels, self.mask)
        pil_image = self.as_PILImage()
        pil_bw_image = pil_image.convert('L')
        return Image(pil_bw_image, mask=self.mask)

    def as_PILImage(self):
        return PILImage.fromarray((self.pixels * 255).astype(np.uint8))


