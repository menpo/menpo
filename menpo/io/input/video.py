from functools import partial

from menpo.image.base import normalize_pixels_range, channels_to_front
from menpo.image import Image
from menpo.base import LazyList

from .base import Importer


class ImageioFFMPEGImporter(Importer):
    r"""
    Imports videos using the FFMPEG plugin from the imageio library. Returns a
    :map:`LazyList` that gives lazy access to the video on a per-frame basis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the video.
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    """

    def __init__(self, filepath, normalise=True):
        super(ImageioFFMPEGImporter, self).__init__(filepath)
        self.normalise = normalise

    @classmethod
    def ffmpeg_types(cls):
        try:
            import imageio

            # Lazy way to get all the extensions supported by imageio/FFMPEG
            ffmpeg_exts = imageio.formats['ffmpeg'].extensions
            return dict(zip(ffmpeg_exts,
                        [ImageioFFMPEGImporter] * len(ffmpeg_exts)))
        except ImportError:
            return {}

    def build(self):
        import imageio

        reader = imageio.get_reader(self.filepath, format='ffmpeg', mode='I')

        def imageio_to_menpo(imio_reader, index):
            pixels = imio_reader.get_data(index)
            pixels = channels_to_front(pixels)

            if self.normalise:
                return Image(normalize_pixels_range(pixels), copy=False)
            else:
                return Image(pixels, copy=False)

        index_callable = partial(imageio_to_menpo, reader)
        ll = LazyList.init_from_index_callable(index_callable,
                                               reader.get_length())
        ll.fps = reader.get_meta_data()['fps']
        return ll
