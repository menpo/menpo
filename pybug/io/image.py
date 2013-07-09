import PIL.Image as PILImage
import numpy as np
from .base import Importer
from pybug.image import Image


class ImageImporter(Importer):

    def __init__(self, filepath):
        super(ImageImporter, self).__init__(filepath)
        self._pil_image = PILImage.open(self.filepath)
        # Image pixels are always saved as double precision between 0-1
        self.image = Image(np.array(self._pil_image, dtype=np.float64) / 255)

    def build(self):
        return self.image
