import abc
import os.path as path
import numpy as np
import PIL.Image as PILImage
from pybug.io.base import Importer, get_importer, find_alternative_files
from pybug.image import MaskedImage


class ImageImporter(Importer):
    r"""
    Base class for importing images.  Image importers are capable of
    automatically importing landmarks that share the same basename as the
    image. A search in the current directory is performed whenever an image
    is imported.

    Parameters
    ----------
    filepath : string
        An absolute filepath
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(ImageImporter, self).__init__(filepath)
        self.attempted_landmark_search = False

    def _build_landmark_importer(self):
        r"""
        Check if a ``landmark_path`` exists, and if it does, then create the
        landmark importer.
        """
        if self.landmark_path is None or not path.exists(self.landmark_path):
            self.landmark_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import image_landmark_types
            self.landmark_importer = get_importer(self.landmark_path,
                                                  image_landmark_types)

    def _search_for_landmarks(self):
        r"""
        Tries to find a set of landmarks with the same name as the image. This
        is only attempted once.

        Returns
        -------
        basename : string
            The basename of the landmarks file found, eg. ``image.pts``.
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
        r"""
        Get the absolute path to the landmarks.

        Returns ``None`` if none can be found.
        Makes it's best effort to find an appropriate landmark set by
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

    @abc.abstractmethod
    def _build_image(self):
        r"""
        Abstract method that handles actually building an image. This involves
        reading the image from disk and doing any necessary processing.

        Should set the ``self.image`` attribute.
        """
        pass

    def build(self):
        r"""
        Overrides the :meth:`build <pybug.io.base.Importer.build>` method.

        Builds the image and landmarks. Assigns the landmark set to the image.

        Returns
        -------
        image : :class:`pybug.image.base.Image`
            The image object with landmarks attached if they exist
        """
        # Build the image as defined by the overridden method and then search
        # for valid landmarks that may have been defined by the importer
        self._build_image()
        self._build_landmark_importer()

        if self.landmark_importer is not None:
            lmark_group = self.landmark_importer.build(
                scale_factors=self.image.shape)
            self.image.landmarks[lmark_group.group_label] = lmark_group
        return self.image


class PILImporter(ImageImporter):
    r"""
    Imports an image using PIL

    Parameters
    ----------
    filepath : string
        Absolute filepath of image
    """

    def __init__(self, filepath):
        super(PILImporter, self).__init__(filepath)

    def _build_image(self):
        r"""
        Read the image using PIL and then use the
        :class:`pybug.image.base.MaskedImage` constructor to create a class.
        Normalise between 0 and 1.0

        Sets the newly built Image to ``self.image``.
        """
        self._pil_image = PILImage.open(self.filepath)
        image_pixels = np.array(self._pil_image, dtype=np.float) / 255.0
        self.image = MaskedImage(image_pixels)
