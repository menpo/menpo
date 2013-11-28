from pybug.image.masked import MaskedNDImage
import pybug.features as fc
import numpy as np


class FeatureNDImage(MaskedNDImage):
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

    pass


class HOG2DImage(FeatureNDImage):
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
    def __init__(self, image_data, mask=None, method='dense',
                 algorithm='dalaltriggs', num_bins=9, cell_size=8,
                 block_size=2, signed_gradient=True, l2_norm_clip=0.2,
                 window_height=1, window_width=1, window_unit='blocks',
                 window_step_vertical=1, window_step_horizontal=1,
                 window_step_unit='pixels', padding=True, verbose=False
                 ):
        self.params = {'method': method,
                       'algorithm': algorithm,
                       'num_bins': num_bins,
                       'cell_size': cell_size,
                       'block_size': block_size,
                       'signed_gradient': signed_gradient,
                       'l2_norm_clip': l2_norm_clip,
                       'window_height': window_height,
                       'window_width': window_width,
                       'window_unit': window_unit,
                       'window_step_vertical': window_step_vertical,
                       'window_step_horizontal': window_step_horizontal,
                       'window_step_unit': window_step_unit,
                       'padding': padding,
                       'verbose': verbose}
        descriptors, window_centres = fc.hog(image_data, **self.params)
        super(HOG2DImage, self).__init__(descriptors, mask=mask)
        self.window_centres = window_centres

    @classmethod
    def blank(cls):
        pass
