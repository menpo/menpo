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
        if len(np_pixels.shape) is 3:
            np_pixels = np.rollaxis(np_pixels, -1)
        # Somewhat surprisingly, this multiplication is quite a bit faster than
        # just dividing by 255, presumably due to divide by zero checks.
        return np_pixels * (1.0 / 255.0) if normalise else np_pixels


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


class ABSImporter(Importer):
    r"""
    Allows importing the ABS file format from the FRGC dataset.

    The z-min value is stripped from the mesh to make it renderable.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath, **kwargs):
        # Setup class before super class call
        super(ABSImporter, self).__init__(filepath)

    def build(self):
        import re

        with open(self.filepath, 'r') as f:
            # Currently these are unused, but they are in the format
            # Could possibly store as metadata?
            # Assume first result for regexes
            re_rows = re.compile(u'([0-9]+) rows')
            n_rows = int(re_rows.findall(f.readline())[0])
            re_cols = re.compile(u'([0-9]+) columns')
            n_cols = int(re_cols.findall(f.readline())[0])

        # This also loads the mask
        #   >>> image_data[:, 0]
        image_data = np.loadtxt(self.filepath, skiprows=3, unpack=True)

        # Replace the lowest value with nan so that we can render properly
        data_view = image_data[:, 1:]
        corrupt_value = np.min(data_view)
        data_view[np.any(np.isclose(data_view, corrupt_value), axis=1)] = np.nan

        return MaskedImage(
            np.rollaxis(np.reshape(data_view, [n_rows, n_cols, 3]), -1),
            np.reshape(image_data[:, 0], [n_rows, n_cols]).astype(np.bool),
            copy=False)


class FLOImporter(Importer):
    r"""
    Allows importing the Middlebury FLO file format.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath, **kwargs):
        # Setup class before super class call
        super(FLOImporter, self).__init__(filepath)

    def build(self):
        with open(self.filepath, 'rb') as f:
            fingerprint = f.read(4)
            if fingerprint != 'PIEH':
                raise ValueError('Invalid FLO file.')

            width, height = np.fromfile(f, dtype=np.uint32, count=2)
            # read the raw flow data (u0, v0, u1, v1, u2, v2,...)
            rawData = np.fromfile(f, dtype=np.float32,
                                  count=width * height * 2)

        shape = (height, width)
        u_raw = rawData[::2].reshape(shape)
        v_raw = rawData[1::2].reshape(shape)
        uv = np.vstack([u_raw[None, ...], v_raw[None, ...]])

        return Image(uv, copy=False)

