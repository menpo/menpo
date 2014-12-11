import numpy as np
import PIL.Image as PILImage
from .base import Importer
from menpo.image import Image, MaskedImage, BooleanImage


class PILImporter(Importer):
    r"""
    Imports an image using PIL.

    Different image modes cause different importing strategies.

    RGB, L, I:
        Imported as either `float` or `uint8` depending on normalisation flag.
    RGBA:
        Imported as :map:`MaskedImage` if normalise is ``True`` else imported
        as a 4 channel `uint8` image.
    1:
        Imported as a :map:`BooleanImage`. Normalisation is ignored.
    F:
        Imported as a floating point image. Normalisation is ignored.

    Parameters
    ----------
    filepath : string
        Absolute filepath of image
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just pass whatever PIL imports back (according
        to types rules outlined in constructor).
    """
    def __init__(self, filepath, normalise=True):
        super(PILImporter, self).__init__(filepath)
        self._pil_image = None
        self.normalise = normalise

    def build(self):
        r"""
        Read the image using PIL and then use the :map:`Image` constructor to
        create a class.
        """
        self._pil_image = PILImage.open(self.filepath)
        mode = self._pil_image.mode
        if mode == 'RGBA':
            # RGB with Alpha Channel
            # If we normalise it then we convert to floating point
            # and set the alpha channel to the mask
            if self.normalise:
                alpha = np.array(self._pil_image)[..., 3].astype(np.bool)
                image_pixels = self._pil_to_numpy(True,
                                                  convert='RGB')
                image = MaskedImage(image_pixels, mask=alpha)
            else:
                # With no normalisation we just return the pixels
                image = Image(self._pil_to_numpy(False))
        elif mode in ['L', 'I', 'RGB']:
            # Greyscale, Integer and RGB images
            image = Image(self._pil_to_numpy(self.normalise))
        elif mode == '1':
            # Can't normalise a binary image
            image = BooleanImage(self._pil_to_numpy(False))
        elif mode == 'P':
            # Convert pallete images to RGB
            image = Image(self._pil_to_numpy(self.normalise, convert='RGB'))
        elif mode == 'F':  # Floating point images
            # Don't normalise as we don't know the scale
            image = Image(self._pil_to_numpy(False))
        else:
            raise ValueError('Unexpected mode for PIL: {}'.format(mode))
        return image

    def _pil_to_numpy(self, normalise, convert=None):
        dtype = np.float if normalise else None
        p = self._pil_image.convert(convert) if convert else self._pil_image
        np_pixels = np.array(p, dtype=dtype, copy=True)
        return np_pixels / 255.0 if normalise else np_pixels


class PILGIFImporter(PILImporter):
    r"""
    Imports a GIF using PIL. Correctly encodes the pallete
    (`P` for `RGB` (Pallete mode).

    For multi-frame GIF animations, will return a list of images containing
    each frame.

    Parameters
    ----------
    filepath : string
        Absolute filepath of image
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just pass whatever PIL imports back (according
        to types rules outlined in constructor).
    """

    def __init__(self, filepath, normalise=True):
        super(PILGIFImporter, self).__init__(filepath, normalise=normalise)

    def build(self):
        r"""
        Read the image using PIL and then use the :map:`Image` constructor to
        create a class.
        """
        self._pil_image = PILImage.open(self.filepath)
        # By default GIFs use a
        if self._pil_image.mode == 'P':
            # Do we need this duration information for playback?
            # duration = self._pil_image.info['duration']
            images = []
            try:
                while 1:  # Keep looping until we hit the end of the GIF
                    np_pixels = self._pil_to_numpy(self.normalise,
                                                   convert='RGB')
                    images.append(Image(np_pixels))
                    # Seek to the next frame
                    self._pil_image.seek(self._pil_image.tell() + 1)
            except EOFError:
                pass  # Exhausted GIF
        else:
            raise ValueError('Unknown mode for GIF: {}'.format(
                self._pil_image.mode))

        return images
