from functools import partial

from menpo.image.base import normalize_pixels_range, channels_to_front
from menpo.image import Image
from menpo.base import LazyList


def ffmpeg_types():
    r"""The supported FFMPEG types.

    Returns
    -------
    supported_types : `dict`
        A dictionary of extensions to the :map:`imageio_ffmpeg_importer`.
    """
    try:
        import imageio

        # Lazy way to get all the extensions supported by imageio/FFMPEG
        ffmpeg_exts = imageio.formats['ffmpeg'].extensions
        return dict(zip(ffmpeg_exts,
                        [imageio_ffmpeg_importer] * len(ffmpeg_exts)))
    except ImportError:
        return {}


def imageio_ffmpeg_importer(filepath, asset=None, normalise=True, **kwargs):
    r"""
    Imports videos using the FFMPEG plugin from the imageio library. Returns a
    :map:`LazyList` that gives lazy access to the video on a per-frame basis.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the video.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`LazyList`
        A :map:`LazyList` containing :map:`Image` or subclasses per frame
        of the video.
    """
    import imageio

    reader = imageio.get_reader(str(filepath), format='ffmpeg', mode='I')

    def imageio_to_menpo(imio_reader, index):
        pixels = imio_reader.get_data(index)
        pixels = channels_to_front(pixels)

        if normalise:
            return Image(normalize_pixels_range(pixels), copy=False)
        else:
            return Image(pixels, copy=False)

    index_callable = partial(imageio_to_menpo, reader)
    ll = LazyList.init_from_index_callable(index_callable,
                                           reader.get_length())
    ll.fps = reader.get_meta_data()['fps']
    return ll
