import abc
from sys import path
import PIL.Image as PILImage
from pybug.io.base import Importer
from pybug.image import Image
from pybug.io.landmark import LandmarkImporter


class ImageImporter(Importer):
    """
    Base class for importing images
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(ImageImporter, self).__init__(filepath)

        if self.landmark_path is None or not path.exists(self.landmark_path):
            self.landmark_importer = None
        else:
            self.landmark_importer = LandmarkImporter(self.landmark_path)

    @property
    def landmark_path(self):
        try:
            return path.join(self.folder, self.relative_landmark_path)
        except:
            return None

    def build(self):
        if self.landmark_importer is not None:
            landmark_dict = self.landmark_importer.build()
            self.image.add_landmark_set(landmark_dict.label, landmark_dict)
        return self.image


class PILImporter(ImageImporter):

    def __init__(self, filepath):
        super(PILImporter, self).__init__(filepath)
        self._pil_image = PILImage.open(self.filepath)
        self.image = Image(self._pil_image)

