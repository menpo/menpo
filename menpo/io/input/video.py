import warnings
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

        if len(ll) != 0:
            # TODO: Remove when imageio fixes the ffmpeg importer duration/start
            # This is a bit grim but the frame->timestamp logic in imageio at
            # the moment is not very accurate and so we really need to ensure
            # that the user is returned a list they can actually index into. So
            # we just remove all the frames that we can't actually index into.
            # Remove from the front
            for start in range(len(ll)):
                if start > 10:  # Arbitrary but probably safe
                    warnings.warn('Highly inaccurate frame->timestamp mapping '
                                  'returned by imageio - many frames are being '
                                  'dropped and thus importing may be very slow.'
                                  ' Please see the documentation.')
                try:
                    ll[start]
                    break
                except:
                    pass
            else:
                # If we never broke out then ALL frames raised exceptions
                ll = LazyList([])
            # Only take the frames after the initial broken ones
            ll = ll[start:]

        if len(ll) != 0:
            n_frames = len(ll) - 1
            for end in range(n_frames, -1, -1):
                if end < n_frames - 10:  # Arbitrary but probably safe
                    warnings.warn('Highly inaccurate frame->timestamp mapping '
                                  'returned by imageio - many frames are being '
                                  'dropped and thus importing may be very slow.'
                                  ' Please see the documentation.')
                try:
                    ll[end]
                    break
                except:
                    pass
            # Only take the frames before the broken ones
            ll = ll[:end+1]

        return ll
