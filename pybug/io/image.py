import PIL.Image as PILImage
import numpy as np
from .base import Importer
from pybug.image import Image


class ImageImporter(Importer):

    def __init__(self, filepath):
        super(ImageImporter, self).__init__(filepath)
        self._pil_image = PILImage.open(self.filepath)
        self.image = Image(self._pil_image)

    def build(self):
        return self.image
