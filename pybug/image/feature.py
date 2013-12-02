from pybug.image.masked import MaskedNDImage
import pybug.features as fc
from pybug.visualize.base import ImageViewer
import numpy as np
import math


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
    def __init__(self, image_data, mask=None, mode='dense',
                 algorithm='dalaltriggs', num_bins=9, cell_size=8,
                 block_size=2, signed_gradient=True, l2_norm_clip=0.2,
                 window_height=1, window_width=1, window_unit='blocks',
                 window_step_vertical=1, window_step_horizontal=1,
                 window_step_unit='pixels', padding=True, verbose=False
                 ):
        self.params = {'mode': mode,
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
                       'verbose': verbose,
                       'original_image_height': image_data.shape[0],
                       'original_image_width': image_data.shape[1],
                       'original_image_channels': image_data.shape[2]}
        #hog, window_centres = fc.hog(image_data, **self.params)
        hog, window_centres = fc.hog(image_data,
                                     self.params['mode'],
                                     self.params['algorithm'],
                                     self.params['num_bins'],
                                     self.params['cell_size'],
                                     self.params['block_size'],
                                     self.params['signed_gradient'],
                                     self.params['l2_norm_clip'],
                                     self.params['window_height'],
                                     self.params['window_width'],
                                     self.params['window_unit'],
                                     self.params['window_step_vertical'],
                                     self.params['window_step_horizontal'],
                                     self.params['window_step_unit'],
                                     self.params['padding'],
                                     self.params['verbose'])
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                # assume that the mask is a boolean image then!
                mask = mask.pixels
            mask = mask[..., 0][window_centres[:, :, 0],
                                window_centres[:, :, 1]]
        super(HOG2DImage, self).__init__(hog, mask=mask)
        self.window_centres = window_centres

    @classmethod
    def blank(cls):
        pass

    def _view(self, figure_id=None, new_figure=False, channel=None,
              masked=True, glyph=True, block_size=None, num_bins=None,
              **kwargs):
        r"""
        View the image using the default image viewer. Currently only
        supports the rendering of 2D images.

        Returns
        -------
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within

        Raises
        ------
        DimensionalityError
            If Image is not 2D
        """
        if glyph:
            channel = 0  # Vectorized images always have 1 channel
            if block_size is None:
                block_size = self.params['block_size']
            if num_bins is None:
                num_bins = self.params['num_bins']
            hog_vector_image, mask_to_view = fc.hog_vector_image(
                self.pixels, self.mask.mask,
                block_size=block_size, num_bins=num_bins)
            pixels_to_view = hog_vector_image[..., None]
            if masked:
                mask = mask_to_view
            else:
                mask = None
        else:
            pixels_to_view = self.pixels
            if masked:
                mask = self.mask.mask
            else:
                mask = None
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channel=channel,
                           mask=mask).render(**kwargs)

    def __str__(self):
        header = (
            '{} 2D HOGImage with {} channels. '
            'Attached mask {:.1%} true.'.format(self._str_shape,
                                                self.n_channels,
                                                self.mask.proportion_true))
        info_str = 'Mode is %s.\nWindow Iterator:\n  - Input image is ' \
                   '%dW x %dH with %d channels.\n' % \
                   (self.params['mode'], self.params['original_image_width'],
                    self.params['original_image_height'],
                    self.params['original_image_channels'])
        cell_pixels = self.params['cell_size']
        block_pixels = self.params['block_size'] * cell_pixels
        if self.params['mode'] == 'dense':
            if self.params['window_unit'] == 'blocks':
                window_height = self.params['window_height'] * block_pixels
                window_width = self.params['window_width'] * block_pixels
            else:
                window_height = self.params['window_height']
                window_width = self.params['window_width']
            if self.params['window_step_unit'] == 'cells':
                window_step_vertical = \
                    self.params['window_step_vertical'] * cell_pixels
                window_step_horizontal = \
                    self.params['window_step_horizontal'] * cell_pixels
            else:
                window_step_vertical = self.params['window_step_vertical']
                window_step_horizontal = self.params['window_step_horizontal']
            pad_flag = self.params['padding']
        else:
            if self.params['algorithm'] == 'dalaltriggs':
                window_height = block_pixels
                window_width = block_pixels
                window_step_vertical = cell_pixels
                window_step_horizontal = cell_pixels
            else:
                window_height = 3*cell_pixels
                window_width = 3*cell_pixels
                window_step_vertical = cell_pixels
                window_step_horizontal = cell_pixels
            pad_flag = False
        info_str = '%s  - Window of size %dW x %dH and step (%dW,%dH).\n' % \
                   (info_str, window_width, window_height,
                    window_step_horizontal, window_step_vertical)
        if pad_flag:
            info_str = '%s  - Padding is enabled.\n' % info_str
        else:
            info_str = '%s  - Padding is disabled.\n' % info_str
        info_str = '%s  - Number of windows is %dW x %dH.\n' % \
                   (info_str, self.pixels.shape[1], self.pixels.shape[0])
        if self.params['algorithm'] == 'dalaltriggs':
            info_str = '%sHOG features:\n  - Algorithm of Dalal & Triggs.\n' \
                       % info_str
            info_str = '%s  - Cell is %dx%d pixels.\n' % \
                           (info_str, self.params['cell_size'],
                            self.params['cell_size'])
            info_str = '%s  - Block is %dx%d cells.\n' % \
                       (info_str, self.params['block_size'],
                        self.params['block_size'])
            if self.params['signed_gradient']:
                info_str = '%s  - %d orientation bins and signed angles.\n' \
                           % (info_str, self.params['num_bins'])
            else:
                info_str = '%s  - %d orientation bins and unsigned angles.\n' \
                           % (info_str, self.params['num_bins'])
            info_str = '%s  - L2-norm clipped at %.1f\n' \
                       % (info_str, self.params['l2_norm_clip'])
            descriptor_length_per_block = \
                self.params['block_size'] * self.params['block_size'] * \
                self.params['num_bins']
            hist1 = 2 + math.ceil(-0.5 + window_height/cell_pixels)
            hist2 = 2 + math.ceil(-0.5 + window_width/cell_pixels)
            descriptor_length_per_window = \
                (hist1-2-(self.params['block_size']-1)) * \
                (hist2-2-(self.params['block_size']-1)) * \
                descriptor_length_per_block
            num_blocks_per_window_vertically = \
                hist1-2-(self.params['block_size']-1)
            num_blocks_per_window_horizontally = \
                hist2-2-(self.params['block_size']-1)
            info_str = '%s  - Number of blocks per window = %dW x %dH.\n' \
                       % (info_str, num_blocks_per_window_horizontally,
                          num_blocks_per_window_vertically)
            info_str = '%s  - Descriptor length per window = ' \
                       '%dW x %dH x %d = %d x 1.\n' \
                       % (info_str, num_blocks_per_window_horizontally,
                          num_blocks_per_window_vertically,
                          descriptor_length_per_block,
                          descriptor_length_per_window)
        else:
            info_str = '%sHOG features:\n  - Algorithm of Zhu & Ramanan.\n' \
                       % info_str
            info_str = '%s  - Cell is %dx%d pixels.\n' % \
                           (info_str, self.params['cell_size'],
                            self.params['cell_size'])
            info_str = '%s  - Block is %dx%d cells.\n' % \
                       (info_str, self.params['block_size'],
                        self.params['block_size'])
            hist1 = round(window_height/cell_pixels)
            hist2 = round(window_width/cell_pixels)
            num_blocks_per_window_vertically = max(hist1-2, 0)
            num_blocks_per_window_horizontally = max(hist2-2, 0)
            descriptor_length_per_block = 27+4
            descriptor_length_per_window = \
                num_blocks_per_window_horizontally * \
                num_blocks_per_window_vertically * \
                descriptor_length_per_block
            info_str = '%s  - Number of blocks per window = %dW x %dH.\n' \
                       % (info_str, num_blocks_per_window_horizontally,
                          num_blocks_per_window_vertically)
            info_str = '%s  - Descriptor length per window = ' \
                       '%dW x %dH x %d = %d x 1.\n' \
                       % (info_str, num_blocks_per_window_horizontally,
                          num_blocks_per_window_vertically,
                          descriptor_length_per_block,
                          descriptor_length_per_window)
        return '\n'.join([header, info_str])