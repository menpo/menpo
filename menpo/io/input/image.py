from functools import partial

import numpy as np
from pathlib import Path

from menpo.base import LazyList
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.image.base import normalize_pixels_range, channels_to_front


def _pil_to_numpy(pil_image, normalize, convert=None):
    p = pil_image.convert(convert) if convert else pil_image
    p = np.asarray(p)
    if normalize:
        return normalize_pixels_range(p)
    else:
        return p


def pillow_importer(filepath, asset=None, normalize=True, **kwargs):
    r"""
    Imports an image using PIL/pillow.

    Different image modes cause different importing strategies.

    RGB, L, I:
        Imported as either `float` or `uint8` depending on normalisation flag.
    RGBA:
        Imported as :map:`MaskedImage` if normalize is ``True`` else imported
        as a 4 channel `uint8` image.
    1:
        Imported as a :map:`BooleanImage`. Normalisation is ignored.
    F:
        Imported as a floating point image. Normalisation is ignored.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of image
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    normalize : `bool`, optional
        If ``True``, normalize between 0.0 and 1.0 and convert to float. If
        ``False`` just pass whatever PIL imports back (according
        to types rules outlined in constructor).
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`Image` or subclass
        The imported image.
    """
    import PIL.Image as PILImage
    if isinstance(filepath, Path):
        filepath = str(filepath)
    pil_image = PILImage.open(filepath)
    mode = pil_image.mode
    if mode == 'RGBA':
        # If normalize is False, then we return the alpha as an extra
        # channel, which can be useful if the alpha channel has semantic
        # meanings!
        if normalize:
            alpha = np.array(pil_image)[..., 3].astype(np.bool)
            image_pixels = _pil_to_numpy(pil_image, True, convert='RGB')
            image = MaskedImage.init_from_channels_at_back(image_pixels,
                                                           mask=alpha)
        else:
            # With no normalisation we just return the pixels
            image = Image.init_from_channels_at_back(
                _pil_to_numpy(pil_image, False))
    elif mode in ['L', 'I', 'RGB']:
        # Greyscale, Integer and RGB images
        image = Image.init_from_channels_at_back(
            _pil_to_numpy(pil_image, normalize))
    elif mode == '1':
        # Convert to 'L' type (http://stackoverflow.com/a/4114122/1716869).
        # Can't normalize a binary image
        image = BooleanImage(_pil_to_numpy(pil_image, False, convert='L'),
                             copy=True)
    elif mode == 'P':
        # Convert pallete images to RGB
        image = Image.init_from_channels_at_back(
            _pil_to_numpy(pil_image, normalize, convert='RGB'))
    elif mode == 'F':  # Floating point images
        # Don't normalize as we don't know the scale
        image = Image.init_from_channels_at_back(
            _pil_to_numpy(pil_image, False))
    else:
        raise ValueError('Unexpected mode for PIL: {}'.format(mode))
    return image


def abs_importer(filepath, asset=None, **kwargs):
    r"""
    Allows importing the ABS file format from the FRGC dataset.

    The z-min value is stripped from the image to make it renderable.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`Image` or subclass
        The imported image.
    """
    import re

    with open(str(filepath), 'r') as f:
        # Currently these are unused, but they are in the format
        # Could possibly store as metadata?
        # Assume first result for regexes
        re_rows = re.compile(u'([0-9]+) rows')
        n_rows = int(re_rows.findall(f.readline())[0])
        re_cols = re.compile(u'([0-9]+) columns')
        n_cols = int(re_cols.findall(f.readline())[0])

    # This also loads the mask
    #   >>> image_data[:, 0]
    image_data = np.loadtxt(str(filepath), skiprows=3, unpack=True)

    # Replace the lowest value with nan so that we can render properly
    data_view = image_data[:, 1:]
    corrupt_value = np.min(data_view)
    data_view[np.any(np.isclose(data_view, corrupt_value), axis=1)] = np.nan

    return MaskedImage(
        np.rollaxis(np.reshape(data_view, [n_rows, n_cols, 3]), -1),
        np.reshape(image_data[:, 0], [n_rows, n_cols]).astype(np.bool),
        copy=False)


def flo_importer(filepath, asset=None, **kwargs):
    r"""
    Allows importing the Middlebury FLO file format.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`Image` or subclass
        The imported image.
    """
    with open(str(filepath), 'rb') as f:
        fingerprint = f.read(4)
        if fingerprint != b'PIEH':
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


def imageio_importer(filepath, asset=None, normalize=True, **kwargs):
    r"""
    Imports images using the imageio library - which is actually fairly similar
    to our importing logic - but contains the necessary plugins to import lots
    of interesting image types like RAW images.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the image.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    normalize : `bool`, optional
        If ``True``, normalize between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`Image` or subclass
        The imported image.
    """
    import imageio

    pixels = imageio.imread(str(filepath))
    pixels = channels_to_front(pixels)

    transparent_types = {'.png'}
    if pixels.shape[0] == 4 and filepath.suffix in transparent_types:
        # If normalize is False, then we return the alpha as an extra
        # channel, which can be useful if the alpha channel has semantic
        # meanings!
        if normalize:
            p = normalize_pixels_range(pixels[:3])
            return MaskedImage(p, mask=pixels[-1].astype(np.bool),
                               copy=False)
        else:
            return Image(pixels, copy=False)

    # Assumed not to have an Alpha channel
    if normalize:
        return Image(normalize_pixels_range(pixels), copy=False)
    else:
        return Image(pixels, copy=False)


def imageio_gif_importer(filepath, asset=None, normalize=True, **kwargs):
    r"""
    Imports GIF images using freeimagemulti plugin from the imageio library.
    Returns a :map:`LazyList` that gives lazy access to the GIF on a per-frame
    basis.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the video.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    normalize : `bool`, optional
        If ``True``, normalize between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`LazyList`
        A :map:`LazyList` containing :map:`Image` or subclasses per frame
        of the GIF.
    """
    import imageio

    reader = imageio.get_reader(str(filepath), format='gif', mode='I')

    def imageio_to_menpo(imio_reader, index):
        pixels = imio_reader.get_data(index)
        pixels = channels_to_front(pixels)

        if pixels.shape[0] == 4:
            # If normalize is False, then we return the alpha as an extra
            # channel, which can be useful if the alpha channel has semantic
            # meanings!
            if normalize:
                p = normalize_pixels_range(pixels[:3])
                return MaskedImage(p, mask=pixels[-1].astype(np.bool),
                                   copy=False)
            else:
                return Image(pixels, copy=False)

        # Assumed not to have an Alpha channel
        if normalize:
            return Image(normalize_pixels_range(pixels), copy=False)
        else:
            return Image(pixels, copy=False)

    index_callable = partial(imageio_to_menpo, reader)
    ll = LazyList.init_from_index_callable(index_callable,
                                           reader.get_length())
    return ll
