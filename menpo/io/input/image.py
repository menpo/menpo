from functools import partial

import numpy as np
from pathlib import Path

from menpo.base import LazyList
from .base import Importer
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.image.base import normalize_pixels_range, channels_to_front


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
        import PIL.Image as PILImage

        self._pil_image = PILImage.open(self.filepath)
        mode = self._pil_image.mode
        if mode == 'RGBA':
            # If normalise is False, then we return the alpha as an extra
            # channel, which can be useful if the alpha channel has semantic
            # meanings!
            if self.normalise:
                alpha = np.array(self._pil_image)[..., 3].astype(np.bool)
                image_pixels = self._pil_to_numpy(True,
                                                  convert='RGB')
                image = MaskedImage(image_pixels, mask=alpha, copy=False)
            else:
                # With no normalisation we just return the pixels
                image = Image(self._pil_to_numpy(False), copy=False)
        elif mode in ['L', 'I', 'RGB']:
            # Greyscale, Integer and RGB images
            image = Image(self._pil_to_numpy(self.normalise), copy=False)
        elif mode == '1':
            # Can't normalise a binary image
            image = BooleanImage(self._pil_to_numpy(False), copy=False)
        elif mode == 'P':
            # Convert pallete images to RGB
            image = Image(self._pil_to_numpy(self.normalise, convert='RGB'))
        elif mode == 'F':  # Floating point images
            # Don't normalise as we don't know the scale
            image = Image(self._pil_to_numpy(False), copy=False)
        else:
            raise ValueError('Unexpected mode for PIL: {}'.format(mode))
        return image

    def _pil_to_numpy(self, normalise, convert=None):
        p = self._pil_image.convert(convert) if convert else self._pil_image
        p = channels_to_front(p)
        if normalise:
            return normalize_pixels_range(p)
        else:
            return p


class ABSImporter(Importer):
    r"""
    Allows importing the ABS file format from the FRGC dataset.

    The z-min value is stripped from the image to make it renderable.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the ABS file.
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
        Absolute filepath of the FLO file.
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


class ImageioImporter(Importer):
    r"""
    Imports images using the imageio library - which is actually fairly similar
    to our importing logic - but contains the necessary plugins to import lots
    of interesting image types like RAW images.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the image.
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    """

    def __init__(self, filepath, normalise=True):
        super(ImageioImporter, self).__init__(filepath)
        self._pil_image = None
        self.normalise = normalise

    def build(self):
        import imageio

        pixels = imageio.imread(self.filepath)
        pixels = channels_to_front(pixels)

        transparent_types = {'.png'}
        filepath = Path(self.filepath)
        if pixels.shape[0] == 4 and filepath.suffix in transparent_types:
            # If normalise is False, then we return the alpha as an extra
            # channel, which can be useful if the alpha channel has semantic
            # meanings!
            if self.normalise:
                p = normalize_pixels_range(pixels[:3])
                return MaskedImage(p, mask=pixels[-1].astype(np.bool),
                                   copy=False)
            else:
                return Image(pixels, copy=False)

        # Assumed not to have an Alpha channel
        if self.normalise:
            return Image(normalize_pixels_range(pixels), copy=False)
        else:
            return Image(pixels, copy=False)


class ImageioGIFImporter(Importer):
    r"""
    Imports GIF images using freeimagemulti plugin from the imageio library.
    Returns a :map:`LazyList` that gives lazy access to the GIF on a per-frame
    basis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the video.
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    """

    def __init__(self, filepath, normalise=True):
        super(ImageioGIFImporter, self).__init__(filepath)
        self.normalise = normalise

    def build(self):
        import imageio

        reader = imageio.get_reader(self.filepath, format='gif', mode='I')

        def imageio_to_menpo(imio_reader, index):
            pixels = imio_reader.get_data(index)
            pixels = channels_to_front(pixels)

            if pixels.shape[0] == 4:
                # If normalise is False, then we return the alpha as an extra
                # channel, which can be useful if the alpha channel has semantic
                # meanings!
                if self.normalise:
                    p = normalize_pixels_range(pixels[:3])
                    return MaskedImage(p, mask=pixels[-1].astype(np.bool),
                                       copy=False)
                else:
                    return Image(pixels, copy=False)

            # Assumed not to have an Alpha channel
            if self.normalise:
                return Image(normalize_pixels_range(pixels), copy=False)
            else:
                return Image(pixels, copy=False)

        index_callable = partial(imageio_to_menpo, reader)
        ll = LazyList.init_from_index_callable(index_callable,
                                               reader.get_length())
        return ll
