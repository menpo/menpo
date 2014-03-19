import numpy as np
import PIL.Image as PILImage
from menpo.io.base import Importer
from menpo.image import MaskedImage


class PILImporter(Importer):
    r"""
    Imports an image using PIL

    Parameters
    ----------
    filepath : string
        Absolute filepath of image
    """

    def __init__(self, filepath):
        super(PILImporter, self).__init__(filepath)

    def build(self):
        r"""
        Read the image using PIL and then use the
        :class:`menpo.image.base.MaskedImage` constructor to create a class.
        Normalise between 0 and 1.0
        """
        self._pil_image = PILImage.open(self.filepath)
        image_pixels = np.array(self._pil_image, dtype=np.float) / 255.0
        return MaskedImage(image_pixels)
