import numpy as np

#from pybug.image.masked import MaskedImage
import pybug.features as fc


# class Feature2DImage(MaskedImage):
#     r"""
#     Represents a 2-dimensional features image with k number of channels, of
#     size ``(M, N, C)`` and data type ``np.float``.
#
#     Parameters
#     ----------
#     image_data :  ndarray
#         The pixel data for the image, where the last axis represents the
#         number of channels.
#     mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
#         A binary array representing the mask. Must be the same
#         shape as the image. Only one mask is supported for an image (so the
#         mask is applied to every channel equally).
#
#         Default: :class:`BooleanNDImage` covering the whole image
#
#     Raises
#     -------
#     ValueError
#         Mask is not the same shape as the image
#     """
#     def blank(cls):
#         r"""
#         Replaces the blank function of the MaskedImage.
#         """
#         pass
#
#     def _view(self, figure_id=None, new_figure=False, channels=None,
#               masked=True, glyph=True, vectors_block_size=10,
#               use_negative=False, **kwargs):
#         r"""
#         View feature image using the default feature image viewer. It can be
#         visualized in glyph mode or subplots mode.
#
#         Parameters
#         ----------
#         figure_id : object
#             A figure id. Could be any valid object that identifies
#             a figure in a given framework (string, int, etc)
#
#             Default: None
#         new_figure : bool
#             Whether the rendering engine should create a new figure.
#
#             Default: False
#         channels: int or list or 'all' or None
#             A specific selection of channels to render. The user can choose
#             either a single or multiple channels. If all, render all channels.
#             If None, in the case of glyph=True, render the first
#             min(pixels.shape[2], 9) and in the case of glyph=False subplot the
#             first min(pixels.shape[2], 36).
#
#             Default: None
#         masked : bool
#             Whether to render the masked feature image.
#
#             Default: True
#         glyph: bool
#             Defines whether to plot the glyph image or the different channels
#             in subplots.
#
#             Default: True
#         vectors_block_size: int
#             Defines the size of each block with vectors of the glyph image.
#
#             Default: 10
#         use_negative: bool
#             If this flag is enabled and if the feature pixels have negative
#             values and if the glyph mode is selected, then there will be
#             created an image containing the glyph of positive and negative
#             values concatenated one on top of the other.
#
#             Default: False
#
#         Raises
#         ------
#         DimensionalityError
#             If Image is not 2D
#         """
#         pixels_to_view = self.pixels
#         if masked:
#             mask = self.mask.mask
#         else:
#             mask = None
#         return FeatureImageViewer(figure_id, new_figure, self.n_dims,
#                                   pixels_to_view, channels=channels, mask=mask,
#                                   glyph=glyph,
#                                   vectors_block_size=vectors_block_size,
#                                   use_negative=use_negative).render(**kwargs)
#
#
# class HOG2DImage(Feature2DImage):
#     r"""
#     Represents a 2-dimensional HOG features image with k number of channels, of
#     size ``(M, N, C)`` and data type ``np.float``.
#
#     Parameters
#     ----------
#     image :  class:`Abstract2DImage`
#         An image object that contains pixels and mask fields.
#     mode : 'dense' or 'sparse'
#         The 'sparse' case refers to the traditional usage of HOGs, so default
#         parameters values are passed to the ImageWindowIterator. The sparse
#         case of 'dalaltriggs' algorithm sets the window height and width equal
#         to block size and the window step horizontal and vertical equal to cell
#         size. Thse sparse case of 'zhuramanan' algorithm sets the window height
#         and width equal to 3 times the cell size and the window step horizontal
#         and vertical equal to cell size. In the 'dense' case, the user can
#         change the ImageWindowIterator related parameters (window_height,
#         window_width, window_unit, window_step_vertical,
#         window_step_horizontal, window_step_unit, padding).
#
#         Default: 'dense'
#     window_height : float
#         Defines the height of the window for the ImageWindowIterator object.
#         The metric unit is defined by window_unit.
#
#         Default: 1
#     window_width : float
#         Defines the width of the window for the ImageWindowIterator object.
#         The metric unit is defined by window_unit.
#
#         Default: 1
#     window_unit : 'blocks' or 'pixels'
#         Defines the metric unit of the window_height and window_width
#         parameters for the ImageWindowIterator object.
#
#         Default: 'blocks'
#     window_step_vertical : float
#         Defines the vertical step by which the window in the
#         ImageWindowIterator is moved, thus it controls the features density.
#         The metric unit is defined by window_step_unit.
#
#         Default: 1
#     window_step_horizontal : float
#         Defines the horizontal step by which the window in the
#         ImageWindowIterator is moved, thus it controls the features density.
#         The metric unit is defined by window_step_unit.
#
#         Default: 1
#     window_step_unit : 'pixels' or 'cells'
#         Defines the metric unit of the window_step_vertical and
#         window_step_horizontal parameters for the ImageWindowIterator object.
#
#         Default: 'pixels'
#     padding : bool
#         Enables/disables padding for the close-to-boundary windows in the
#         ImageWindowIterator object. When padding is enabled, the
#         out-of-boundary pixels are set to zero.
#
#         Default: True
#     algorithm : 'dalaltriggs' or 'zhuramanan'
#         Specifies the algorithm used to compute HOGs.
#
#         Default: 'dalaltriggs'
#     cell_size : float
#         Defines the cell size in pixels. This value is set to both the width
#         and height of the cell. This option is valid for both algorithms.
#
#         Default: 8
#     block_size : float
#         Defines the block size in cells. This value is set to both the width
#         and height of the block. This option is valid only for the
#         'dalaltriggs' algorithm.
#
#         Default: 2
#     num_bins : float
#         Defines the number of orientation histogram bins. This option is valid
#         only for the 'dalaltriggs' algorithm.
#
#         Default: 9
#     signed_gradient : bool
#         Flag that defines whether we use signed or unsigned gradient angles.
#         This option is valid only for the 'dalaltriggs' algorithm.
#
#         Default: True
#     l2_norm_clip : float
#         Defines the clipping value of the gradients' L2-norm. This option is
#         valid only for the 'dalaltriggs' algorithm.
#
#         Default: 0.2
#     verbose : bool
#         Flag to print HOG related information.
#
#         Default: False
#
#     Raises
#     -------
#     ValueError
#         Mask is not the same shape as the image
#     ValueError
#         HOG features mode must be either dense or sparse
#     ValueError
#         Algorithm must be either dalaltriggs or zhuramanan
#     ValueError
#         Number of orientation bins must be > 0
#     ValueError
#         Cell size (in pixels) must be > 0
#     ValueError
#         Block size (in cells) must be > 0
#     ValueError
#         Value for L2-norm clipping must be > 0.0
#     ValueError
#         Window height must be >= block size and <= image height
#     ValueError
#         Window width must be >= block size and <= image width
#     ValueError
#         Window unit must be either pixels or blocks
#     ValueError
#         Horizontal window step must be > 0
#     ValueError
#         Vertical window step must be > 0
#     ValueError
#         Window step unit must be either pixels or cells
#     """
#     def __init__(self, image, mode='dense', algorithm='dalaltriggs',
#                  num_bins=9, cell_size=8, block_size=2, signed_gradient=True,
#                  l2_norm_clip=0.2, window_height=1, window_width=1,
#                  window_unit='blocks', window_step_vertical=1,
#                  window_step_horizontal=1, window_step_unit='pixels',
#                  padding=True, verbose=False):
#         self.params = {'mode': mode,
#                        'algorithm': algorithm,
#                        'num_bins': num_bins,
#                        'cell_size': cell_size,
#                        'block_size': block_size,
#                        'signed_gradient': signed_gradient,
#                        'l2_norm_clip': l2_norm_clip,
#                        'window_height': window_height,
#                        'window_width': window_width,
#                        'window_unit': window_unit,
#                        'window_step_vertical': window_step_vertical,
#                        'window_step_horizontal': window_step_horizontal,
#                        'window_step_unit': window_step_unit,
#                        'padding': padding,
#                        'verbose': verbose,
#                        'original_image_height': image.pixels.shape[0],
#                        'original_image_width': image.pixels.shape[1],
#                        'original_image_channels': image.pixels.shape[2]}
#         #hog, window_centres = fc.hog(image_data, **self.params)
#         hog, window_centres = fc.hog(image.pixels,
#                                      self.params['mode'],
#                                      self.params['algorithm'],
#                                      self.params['num_bins'],
#                                      self.params['cell_size'],
#                                      self.params['block_size'],
#                                      self.params['signed_gradient'],
#                                      self.params['l2_norm_clip'],
#                                      self.params['window_height'],
#                                      self.params['window_width'],
#                                      self.params['window_unit'],
#                                      self.params['window_step_vertical'],
#                                      self.params['window_step_horizontal'],
#                                      self.params['window_step_unit'],
#                                      self.params['padding'],
#                                      self.params['verbose'])
#         mask = image.mask
#         if mask is not None:
#             if not isinstance(mask, np.ndarray):
#                 # assume that the mask is a boolean image then!
#                 mask = mask.pixels
#             mask = mask[..., 0][window_centres[:, :, 0],
#                                 window_centres[:, :, 1]]
#         super(HOG2DImage, self).__init__(hog, mask=mask)
#         self.window_centres = window_centres
#
#     def __str__(self):
#         cell_pixels = self.params['cell_size']
#         block_pixels = self.params['block_size'] * cell_pixels
#         if self.params['mode'] == 'dense':
#             if self.params['window_unit'] == 'blocks':
#                 window_height = self.params['window_height'] * block_pixels
#                 window_width = self.params['window_width'] * block_pixels
#             else:
#                 window_height = self.params['window_height']
#                 window_width = self.params['window_width']
#             if self.params['window_step_unit'] == 'cells':
#                 window_step_vertical = \
#                     self.params['window_step_vertical'] * cell_pixels
#                 window_step_horizontal = \
#                     self.params['window_step_horizontal'] * cell_pixels
#             else:
#                 window_step_vertical = self.params['window_step_vertical']
#                 window_step_horizontal = self.params['window_step_horizontal']
#             pad_flag = self.params['padding']
#         else:
#             if self.params['algorithm'] == 'dalaltriggs':
#                 window_height = block_pixels
#                 window_width = block_pixels
#                 window_step_vertical = cell_pixels
#                 window_step_horizontal = cell_pixels
#             else:
#                 window_height = 3*cell_pixels
#                 window_width = 3*cell_pixels
#                 window_step_vertical = cell_pixels
#                 window_step_horizontal = cell_pixels
#             pad_flag = False
#
#         info_str = "{} 2D HOGImage with {} channels. " \
#                    "Attached mask {:.1%} true.\n" \
#                    "Mode is {}.\n" \
#                    "Window Iterator:\n" \
#                    "  - Input image is {}W x {}H with {} channels.\n" \
#                    "  - Window of size {}W x {}H and step ({}W,{}H).\n"\
#             .format(self._str_shape, self.n_channels,
#                     self.mask.proportion_true, self.params['mode'],
#                     self.params['original_image_width'],
#                     self.params['original_image_height'],
#                     self.params['original_image_channels'], window_width,
#                     window_height, window_step_horizontal,
#                     window_step_vertical)
#         if pad_flag:
#             info_str = "{}  - Padding is enabled.\n".format(info_str)
#         else:
#             info_str = "{}  - Padding is disabled.\n".format(info_str)
#         info_str = "{}  - Number of windows is {}W x {}H.\n"\
#             .format(info_str, self.pixels.shape[1], self.pixels.shape[0])
#         if self.params['algorithm'] == 'dalaltriggs':
#             descriptor_length_per_block = \
#                 self.params['block_size'] * self.params['block_size'] * \
#                 self.params['num_bins']
#             hist1 = 2 + np.ceil(-0.5 + window_height/cell_pixels)
#             hist2 = 2 + np.ceil(-0.5 + window_width/cell_pixels)
#             descriptor_length_per_window = \
#                 (hist1-2-(self.params['block_size']-1)) * \
#                 (hist2-2-(self.params['block_size']-1)) * \
#                 descriptor_length_per_block
#             num_blocks_per_window_vertically = \
#                 hist1-2-(self.params['block_size']-1)
#             num_blocks_per_window_horizontally = \
#                 hist2-2-(self.params['block_size']-1)
#             # complete info_str
#             info_str = "{0}HOG features:\n" \
#                        "  - Algorithm of Dalal & Triggs.\n" \
#                        "  - Cell is {1}x{1} pixels.\n" \
#                        "  - Block is {2}x{2} cells.\n"\
#                 .format(info_str, self.params['cell_size'],
#                         self.params['block_size'])
#             if self.params['signed_gradient']:
#                 info_str = "{}  - {} orientation bins and signed angles.\n"\
#                     .format(info_str, self.params['num_bins'])
#             else:
#                 info_str = "{}  - {} orientation bins and unsigned angles.\n"\
#                     .format(info_str, self.params['num_bins'])
#             info_str = "{0}  - L2-norm clipped at {1:.1}.\n" \
#                        "  - Number of blocks per window = {2}W x {3}H.\n" \
#                        "  - Descriptor length per window " \
#                        "= {2}W x {3}H x {4} = {5} x 1.\n"\
#                 .format(info_str, self.params['l2_norm_clip'],
#                         int(num_blocks_per_window_horizontally),
#                         int(num_blocks_per_window_vertically),
#                         int(descriptor_length_per_block),
#                         int(descriptor_length_per_window))
#         else:
#             hist1 = round(window_height/cell_pixels)
#             hist2 = round(window_width/cell_pixels)
#             num_blocks_per_window_vertically = max(hist1-2, 0)
#             num_blocks_per_window_horizontally = max(hist2-2, 0)
#             descriptor_length_per_block = 27+4
#             descriptor_length_per_window = \
#                 num_blocks_per_window_horizontally * \
#                 num_blocks_per_window_vertically * \
#                 descriptor_length_per_block
#             # complete info_str
#             info_str = "{0}HOG features:\n" \
#                        "  - Algorithm of Zhu & Ramanan.\n" \
#                        "  - Cell is {1}x{1} pixels.\n" \
#                        "  - Block is {2}x{2} cells.\n" \
#                        "  - Number of blocks per window = {3}W x {4}H.\n" \
#                        "  - Descriptor length per window " \
#                        "{3}W x {4}H x {5} = {6} x 1.\n"\
#                 .format(info_str, self.params['cell_size'],
#                         self.params['block_size'],
#                         int(num_blocks_per_window_horizontally),
#                         int(num_blocks_per_window_vertically),
#                         int(descriptor_length_per_block),
#                         int(descriptor_length_per_window))
#         return info_str
#
#
# class IGO2DImage(Feature2DImage):
#     r"""
#     Represents a 2-dimensional IGO features image with k=[2,4] number of
#     channels, of size ``(M, N, C)`` and data type ``np.float``.
#
#     Parameters
#     ----------
#     image :  class:`Abstract2DImage`
#         An image object that contains pixels and mask fields.
#     double_angles : bool
#         Assume that phi represents the gradient orientations. If this flag is
#         disabled, the features image is the concatenation of cos(phi) and
#         sin(phi), thus 2 channels. If it is enabled, the features image is
#         the concatenation of cos(phi), sin(phi), cos(2*phi), sin(2*phi).
#
#         Default: False
#     verbose : bool
#         Flag to print IGO related information.
#
#         Default: False
#
#     Raises
#     -------
#     ValueError
#         Mask is not the same shape as the image
#     """
#     def __init__(self, image, double_angles=False, verbose=False):
#         self.params = {'double_angles': double_angles,
#                        'verbose': verbose,
#                        'original_image_height': image.pixels.shape[0],
#                        'original_image_width': image.pixels.shape[1],
#                        'original_image_channels': image.pixels.shape[2]}
#         # compute igo
#         if image.n_channels == 3:
#             grad = image.as_greyscale().gradient()
#         else:
#             grad = image.gradient()
#         grad_orient = np.angle(grad.pixels[..., 0] + 1j*grad.pixels[..., 1])
#         if double_angles:
#             igo = np.concatenate((np.cos(grad_orient)[..., np.newaxis],
#                                   np.sin(grad_orient)[..., np.newaxis],
#                                   np.cos(2*grad_orient)[..., np.newaxis],
#                                   np.sin(2*grad_orient)[..., np.newaxis]), 2)
#         else:
#             igo = np.concatenate((np.cos(grad_orient)[..., np.newaxis],
#                                   np.sin(grad_orient)[..., np.newaxis]), 2)
#         super(IGO2DImage, self).__init__(igo, mask=image.mask)
#         if verbose:
#             print self.__str__()
#
#     def __str__(self):
#         info_str = "{} 2D IGOImage with {} channels. " \
#                    "Attached mask {:.1%} true.\n" \
#                    "  - Input image is {}W x {}H with {} channels.\n" \
#             .format(self._str_shape, self.n_channels,
#                     self.mask.proportion_true,
#                     self.params['original_image_width'],
#                     self.params['original_image_height'],
#                     self.params['original_image_channels'])
#         if self.params['double_angles']:
#             info_str = "{}  - Double angles are enabled.\n".format(info_str)
#         else:
#             info_str = "{}  - Double angles are disabled.\n".format(info_str)
#         return info_str


class FeatureExtraction(object):
    r"""
    Class that given an image object, it adds the feature extraction
    functionality of Feature2DImage class. The feature extraction also includes
    potential correction of landmarks. Currently it is used by Abstract2DImage
    class.

    Parameters
    ----------
    image :  Abstract2DImage object
        The object created by Abstract2DImage class.
    """

    def __init__(self, image):
        self._image = image

    def hog(self, mode='dense', algorithm='dalaltriggs', num_bins=9,
            cell_size=8, block_size=2, signed_gradient=True, l2_norm_clip=0.2,
            window_height=1, window_width=1, window_unit='blocks',
            window_step_vertical=1, window_step_horizontal=1,
            window_step_unit='pixels', padding=True, verbose=False):
        r"""
        Represents a 2-dimensional HOG features image with k number of
        channels. It returns an image object with potential landmarks of the
        same type as the input image object.

        Parameters
        ----------
        For the parameters explanation, please refer to HOG2DImage class of
        this file.
        """
        # compute hog features
        hog, window_centres = fc.hog(self._image.pixels, mode=mode,
                                     algorithm=algorithm, cell_size=cell_size,
                                     block_size=block_size, num_bins=num_bins,
                                     signed_gradient=signed_gradient,
                                     l2_norm_clip=l2_norm_clip,
                                     window_height=window_height,
                                     window_width=window_width,
                                     window_unit=window_unit,
                                     window_step_vertical=window_step_vertical,
                                     window_step_horizontal=
                                     window_step_horizontal,
                                     window_step_unit=window_step_unit,
                                     padding=padding, verbose=verbose)
        mask = self._image.mask
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                # assume that the mask is a boolean image then!
                mask = mask.pixels
            mask = mask[..., 0][window_centres[:, :, 0],
                                window_centres[:, :, 1]]
        super(HOG2DImage, self).__init__(hog, mask=mask)
        self.window_centres = window_centres

        # hog = HOG2DImage(self._image,
        #                  mode=mode, algorithm=algorithm, cell_size=cell_size,
        #                  block_size=block_size, num_bins=num_bins,
        #                  signed_gradient=signed_gradient,
        #                  l2_norm_clip=l2_norm_clip,
        #                  window_height=window_height,
        #                  window_width=window_width, window_unit=window_unit,
        #                  window_step_vertical=window_step_vertical,
        #                  window_step_horizontal=window_step_horizontal,
        #                  window_step_unit=window_step_unit, padding=padding,
        #                  verbose=verbose)
        # correct landmarks
        # hog.landmarks = self._image.landmarks
        # if hog.landmarks.has_landmarks:
        #     for l_group in hog.landmarks:
        #         l = hog.landmarks[l_group[0]]
        #         # make sure window steps are in pixels mode
        #         window_step_vertical = (hog.window_centres[1, 0, 0] -
        #                                 hog.window_centres[0, 0, 0])
        #         window_step_horizontal = (hog.window_centres[0, 1, 1] -
        #                                   hog.window_centres[0, 0, 1])
        #         # convert points by subtracting offset (controlled by padding)
        #         # and dividing with step at each direction
        #         l.lms.points[:, 0] = (l.lms.points[:, 0] -
        #                               hog.window_centres[:, :, 0].min()) / \
        #             window_step_vertical
        #         l.lms.points[:, 1] = (l.lms.points[:, 1] -
        #                               hog.window_centres[:, :, 1].min()) / \
        #             window_step_horizontal
        return hog





#     def __init__(self, image, mode='dense', algorithm='dalaltriggs',
#                  num_bins=9, cell_size=8, block_size=2, signed_gradient=True,
#                  l2_norm_clip=0.2, window_height=1, window_width=1,
#                  window_unit='blocks', window_step_vertical=1,
#                  window_step_horizontal=1, window_step_unit='pixels',
#                  padding=True, verbose=False):
#         self.params = {'mode': mode,
#                        'algorithm': algorithm,
#                        'num_bins': num_bins,
#                        'cell_size': cell_size,
#                        'block_size': block_size,
#                        'signed_gradient': signed_gradient,
#                        'l2_norm_clip': l2_norm_clip,
#                        'window_height': window_height,
#                        'window_width': window_width,
#                        'window_unit': window_unit,
#                        'window_step_vertical': window_step_vertical,
#                        'window_step_horizontal': window_step_horizontal,
#                        'window_step_unit': window_step_unit,
#                        'padding': padding,
#                        'verbose': verbose,
#                        'original_image_height': image.pixels.shape[0],
#                        'original_image_width': image.pixels.shape[1],
#                        'original_image_channels': image.pixels.shape[2]}
#         #hog, window_centres = fc.hog(image_data, **self.params)
#         hog, window_centres = fc.hog(image.pixels,
#                                      self.params['mode'],
#                                      self.params['algorithm'],
#                                      self.params['num_bins'],
#                                      self.params['cell_size'],
#                                      self.params['block_size'],
#                                      self.params['signed_gradient'],
#                                      self.params['l2_norm_clip'],
#                                      self.params['window_height'],
#                                      self.params['window_width'],
#                                      self.params['window_unit'],
#                                      self.params['window_step_vertical'],
#                                      self.params['window_step_horizontal'],
#                                      self.params['window_step_unit'],
#                                      self.params['padding'],
#                                      self.params['verbose'])
#         mask = image.mask
#         if mask is not None:
#             if not isinstance(mask, np.ndarray):
#                 # assume that the mask is a boolean image then!
#                 mask = mask.pixels
#             mask = mask[..., 0][window_centres[:, :, 0],
#                                 window_centres[:, :, 1]]
#         super(HOG2DImage, self).__init__(hog, mask=mask)
#         self.window_centres = window_centres








    # def igo(self, double_angles=False, verbose=False):
    #     r"""
    #     Represents a 2-dimensional IGO features image with k=[2,4] number of
    #     channels. It returns an image object with potential landmarks of the
    #     same type as the input image object.
    #
    #     Parameters
    #     ----------
    #     For the parameters explanation, please refer to HOG2DImage class of
    #     this file.
    #     """
    #     # compute igo features
    #     igo = IGO2DImage(self._image, double_angles=double_angles,
    #                      verbose=verbose)
    #     # correct landmarks
    #     igo.landmarks = self._image.landmarks
    #     return igo
