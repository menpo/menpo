import PIL
import numpy as np
from .base import Importer
from pybug.image import Image


class ImageImporter(Importer):

    def __init__(self, filepath):
        super(ImageImporter, self).__init__(filepath)
        self._pil_image = PIL.Image.open(self.filepath)
        self.image = Image(np.array(self._pil_image))

    def shape(self):
        pass