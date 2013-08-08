import abc
import os.path as path
import PIL.Image as PILImage
from pybug.io.base import Importer, get_importer, find_alternative_files
from pybug.image import Image


class ImageImporter(Importer):
    """
    Base class for importing images
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(ImageImporter, self).__init__(filepath)
        self.attempted_landmark_search = False

        if self.landmark_path is None or not path.exists(self.landmark_path):
            self.landmark_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import image_landmark_types
            self.landmark_importer = get_importer(self.landmark_path,
                                                  image_landmark_types)

    def _search_for_landmarks(self):
        """
        Tries to find a set of landmarks with the same name as the image
        :return: The relative landmarks path, or None
        """
        # Stop searching every single time we access the property
        self.attempted_landmark_search = True
        # This import is here to avoid circular dependencies
        from pybug.io.extensions import image_landmark_types
        try:
            return find_alternative_files('landmarks', self.filepath,
                                          image_landmark_types)
        except ImportError:
            return None

    @property
    def landmark_path(self):
        """
        Get the absolute path to the landmarks. Returns None if none can be
        found. Makes it's best effort to find an appropriate landmark set by
        searching for landmarks with the same name as the image.
        """
        # Avoid attribute not being set
        if not hasattr(self, 'relative_landmark_path'):
            self.relative_landmark_path = None

        # Try find a texture path if we can
        if self.relative_landmark_path is None and \
                not self.attempted_landmark_search:
            self.relative_landmark_path = self._search_for_landmarks()

        try:
            return path.join(self.folder, self.relative_landmark_path)
        except AttributeError:
            return None

    def build(self):
        if self.landmark_importer is not None:
            label, lmark_dict = self.landmark_importer.build(
                scale_factors=self.image.shape)
            self.image.add_landmark_set(label, lmark_dict)
        return self.image


class PILImporter(ImageImporter):

    def __init__(self, filepath):
        super(PILImporter, self).__init__(filepath)
        self._pil_image = PILImage.open(self.filepath)
        self.image = Image(self._pil_image)

