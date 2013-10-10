from copy import deepcopy
import numpy as np
import PIL.Image as PILImage
import scipy.linalg
from pybug.image.masked import MaskedNDImage


class Abstract2DImage(MaskedNDImage):
    r"""
    Represents a 2-dimensional image with k number of channels, of size
    ``(M, N, C)``. ``np.uint8`` pixel data is converted to ``np.float64``
    and scaled between ``0`` and ``1`` by dividing each pixel by ``255``.
    All Image2D instances have values for channels between 0-1,
    and have a dtype of np.float.

    Parameters
    ----------
    image_data :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        image = np.asarray(image_data)
        if image.ndim != 3:
            raise ValueError("Trying to build a 2DImage with "
                             "{} dims".format(image.ndim - 1))
        # ensure pixels are np.float [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float) / 255
        # ensure that anything else is cast to float
        image = np.require(image, dtype=np.float)
        # Ensure values are between 0 and 1 inclusive
        if np.any(image > 1.0) or np.any(image < 0.0):
            raise ValueError("2D Images can only have values in the "
                             "range [0-1]")
        super(Abstract2DImage, self).__init__(image, mask=mask)

    # noinspection PyUnresolvedReferences
    def as_PILImage(self):
        r"""
        Return a PIL copy of the image. Scales the image by ``255`` and
        converts to ``np.uint8``.

        Returns
        -------
        pil_image : ``PILImage``
            PIL copy of image as ``np.uint8``
        """
        return PILImage.fromarray((self.pixels * 255).astype(np.uint8))


class RGBImage(Abstract2DImage):
    r"""
    Represents a 2-dimensional image with 3 channels for RBG respectively,
    of size ``(M, N, 3)``. ``np.uint8`` pixel data is converted to ``np
    .float64``, and scaled between ``0`` and ``1`` by dividing each pixel by
    ``255``.

    Parameters
    ----------
    image_data :  ndarray or ``PILImage``
        The pixel data for the image, where the last axis represents the
        three channels.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        super(RGBImage, self).__init__(image_data, mask=mask)
        if self.n_channels != 3:
            raise ValueError("Trying to build an RGBImage with {} channels -"
                             " you must provide a numpy array of size (M, N,"
                             " 3)".format(self.n_channels))

    @classmethod
    def blank(cls, shape, fill=0, mask=None, **kwargs):
        r"""
        Returns a blank RGBImage

        Parameters
        ----------
        shape : tuple or list
            The shape of the image

        fill : int, optional
            The value to fill all pixels with

            Default: 0
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanNDImage`

        Returns
        -------
        blank_image : :class:`RGBImage`
            A new RGB image of the requested size.
        """
        n_channels = kwargs.get('n_channels', 3)
        if n_channels != 3:
            raise ValueError('The number of channels of a RGBImage must be '
                             'set to 3')
        return super(RGBImage, cls).blank(
            shape, n_channels=n_channels, fill=fill, dtype=np.float,
            mask=mask)

    def __str__(self):
        return ('{} RGBImage. '
                'Attached mask {:.1%} true'.format(
                self._str_shape, self.mask.proportion_true))

    def as_greyscale(self, mode='luminosity', channel=None):
        r"""
        Returns a greyscale version of the RGB image.

        Parameters
        ----------
        mode : {'average', 'luminosity', 'channel'}
            'luminosity' - Calculates the luminance using the CCIR 601 formula
                ``Y' = 0.2989 R' + 0.5870 G' + 0.1140 B'``
            'average' - intensity is an equal average of all three channels
            'channel' - a specific channel is used

            Default 'luminosity'

        channel: int, optional
            The channel to be taken. Only used if mode is 'channel'.

            Default: None

        Returns
        -------
        greyscale_image: :class:`IntensityImage`
            A copy of this image in greyscale.
        """
        pixels = None
        if mode == 'luminosity':
            # Invert the transformation matrix to get more precise values
            T = scipy.linalg.inv(np.array([[1.0, 0.956, 0.621],
                                           [1.0, -0.272, -0.647],
                                           [1.0, -1.106, 1.703]]))
            coef = T[0, :]
            pixels = np.dot(self.pixels, coef.T)
        elif mode == 'average':
            pixels = np.mean(self.pixels, axis=-1)
        elif mode == 'channel':
            if channel is None:
                raise ValueError("for the 'channel' mode you have to provide"
                                 " a channel index")
            elif channel < 0 or channel > 2:
                raise ValueError("channel can only be 0, 1, or 2 "
                                 "in RGB images.")
            pixels = self.pixels[..., channel]
        mask = deepcopy(self.mask)
        greyscale = IntensityImage(pixels, mask=mask)
        greyscale.landmarks = self.landmarks
        return greyscale


class IntensityImage(Abstract2DImage):
    r"""
    Represents a 2-dimensional image with 1 channel for intensity,
    of size ``(M, N, 1)``. ``np.uint8`` pixel data is converted to ``np
    .float64``, and scaled between ``0`` and ``1`` by dividing each pixel by
    ``255``.

    Parameters
    ----------
    image_data : (M, N)  ndarray or ``PILImage``
        The pixel data for the image. Note that a channel axis should not be
         provided.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        # add the channel axis on
        super(IntensityImage, self).__init__(image_data[..., None], mask=mask)
        if self.n_channels != 1:
            raise ValueError("Trying to build a IntensityImage with {} "
                             "channels (you shouldn't provide a channel "
                             "axis to the IntensityImage constructor)"
                             .format(self.n_channels))

    @classmethod
    def _init_with_channel(cls, image_data_with_channel, mask):
        if image_data_with_channel.ndim != 3 or \
           image_data_with_channel.shape[-1] != 1:
            raise ValueError("IntensityImage must be constructed with 3 "
                             "dimensions and 1 channel.")
        return cls(image_data_with_channel[..., 0], mask)

    @classmethod
    def blank(cls, shape, fill=0, mask=None, **kwargs):
        r"""
        Returns a blank IntensityImage

        Parameters
        ----------
        shape : tuple or list
            The shape of the image

        fill : int, optional
            The value to fill all pixels with

            Default: 0
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanNDImage`

        Returns
        -------
        blank_image : :class:`IntensityImage`
            A new intensity image of the requested size.
        """
        n_channels = kwargs.get('n_channels', 1)
        if n_channels != 1:
            raise ValueError('The number of channels of a IntensityImage '
                             'must be set to 1')
        return super(IntensityImage, cls).blank(
            shape, n_channels=n_channels, fill=fill, dtype=np.float,
            mask=mask)

    def __str__(self):
        return ('{} IntensityImage. '
                'Attached mask {:.1%} true'.format(
                self._str_shape, self.mask.proportion_true))
