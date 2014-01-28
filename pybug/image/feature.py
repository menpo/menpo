import numpy as np
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

class FeatureExtraction(object):
    r"""
    Class that given an image object, it adds the feature extraction
    functionality to the object and computes the features image. The output
    image's class is either MaskedImage or Image depending on the input image.
    The mask and landmarks are corrected based on the features calculation.

    Parameters
    ----------
    image :  Any image object
        The object created by any image class (Image, MaskedImage,
        SpatialImage, DepthImage).
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
        channels. The output object's class is either MaskedImage or Image
        depending on the input image.

        Parameters
        ----------
        mode : 'dense' or 'sparse'
            The 'sparse' case refers to the traditional usage of HOGs, so
            default parameters values are passed to the ImageWindowIterator.
            The sparse case of 'dalaltriggs' algorithm sets the window height
            and width equal to block size and the window step horizontal and
            vertical equal to cell size. Thse sparse case of 'zhuramanan'
            algorithm sets the window height and width equal to 3 times the
            cell size and the window step horizontal and vertical equal to cell
            size. In the 'dense' case, the user can change the
            ImageWindowIterator related parameters (window_height,
            window_width, window_unit, window_step_vertical,
            window_step_horizontal, window_step_unit, padding).

        Default: 'dense'
        window_height : float
            Defines the height of the window for the ImageWindowIterator
            object. The metric unit is defined by window_unit.

            Default: 1
        window_width : float
            Defines the width of the window for the ImageWindowIterator object.
            The metric unit is defined by window_unit.

            Default: 1
        window_unit : 'blocks' or 'pixels'
            Defines the metric unit of the window_height and window_width
            parameters for the ImageWindowIterator object.

            Default: 'blocks'
        window_step_vertical : float
            Defines the vertical step by which the window in the
            ImageWindowIterator is moved, thus it controls the features
            density. The metric unit is defined by window_step_unit.

            Default: 1
        window_step_horizontal : float
            Defines the horizontal step by which the window in the
            ImageWindowIterator is moved, thus it controls the features
            density. The metric unit is defined by window_step_unit.

            Default: 1
        window_step_unit : 'pixels' or 'cells'
            Defines the metric unit of the window_step_vertical and
            window_step_horizontal parameters for the ImageWindowIterator
            object.

            Default: 'pixels'
        padding : bool
            Enables/disables padding for the close-to-boundary windows in the
            ImageWindowIterator object. When padding is enabled, the
            out-of-boundary pixels are set to zero.

            Default: True
        algorithm : 'dalaltriggs' or 'zhuramanan'
            Specifies the algorithm used to compute HOGs.

            Default: 'dalaltriggs'
        cell_size : float
            Defines the cell size in pixels. This value is set to both the
            width and height of the cell. This option is valid for both
            algorithms.

            Default: 8
        block_size : float
            Defines the block size in cells. This value is set to both the
            width and height of the block. This option is valid only for the
            'dalaltriggs' algorithm.

            Default: 2
        num_bins : float
            Defines the number of orientation histogram bins. This option is
            valid only for the 'dalaltriggs' algorithm.

            Default: 9
        signed_gradient : bool
            Flag that defines whether we use signed or unsigned gradient
            angles. This option is valid only for the 'dalaltriggs' algorithm.

            Default: True
        l2_norm_clip : float
            Defines the clipping value of the gradients' L2-norm. This option
            is valid only for the 'dalaltriggs' algorithm.

            Default: 0.2
        verbose : bool
            Flag to print HOG related information.

            Default: False

        Raises
        -------
        ValueError
            HOG features mode must be either dense or sparse
        ValueError
            Algorithm must be either dalaltriggs or zhuramanan
        ValueError
            Number of orientation bins must be > 0
        ValueError
            Cell size (in pixels) must be > 0
        ValueError
            Block size (in cells) must be > 0
        ValueError
            Value for L2-norm clipping must be > 0.0
        ValueError
            Window height must be >= block size and <= image height
        ValueError
            Window width must be >= block size and <= image width
        ValueError
            Window unit must be either pixels or blocks
        ValueError
            Horizontal window step must be > 0
        ValueError
            Vertical window step must be > 0
        ValueError
            Window step unit must be either pixels or cells
        """
        # compute hog features and windows_centres
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
        # create hog image object
        hog_image = self._init_feature_image(hog,
                                             window_centres=window_centres)
        # store parameters
        hog_image.parameters = {'mode': mode, 'algorithm': algorithm,
                                'num_bins': num_bins, 'cell_size': cell_size,
                                'block_size': block_size,
                                'signed_gradient': signed_gradient,
                                'l2_norm_clip': l2_norm_clip,
                                'window_height': window_height,
                                'window_width': window_width,
                                'window_unit': window_unit,
                                'window_step_vertical': window_step_vertical,
                                'window_step_horizontal':
                                window_step_horizontal,
                                'window_step_unit': window_step_unit,
                                'padding': padding,
                                'original_image_height':
                                self._image.pixels.shape[0],
                                'original_image_width':
                                self._image.pixels.shape[1],
                                'original_image_channels':
                                self._image.pixels.shape[2]}
        return hog_image

    def igo(self, double_angles=False, verbose=False):
        r"""
        Represents a 2-dimensional IGO features image with k=[2,4] number of
        channels. The output object's class is either MaskedImage or Image
        depending on the input image.

        Parameters
        ----------
        image :  class:`Abstract2DImage`
            An image object that contains pixels and mask fields.
        double_angles : bool
            Assume that phi represents the gradient orientations. If this flag
            is disabled, the features image is the concatenation of cos(phi)
            and sin(phi), thus 2 channels. If it is enabled, the features image
            is the concatenation of cos(phi), sin(phi), cos(2*phi), sin(2*phi).

            Default: False
        verbose : bool
            Flag to print IGO related information.

            Default: False
        """
        # compute igo
        if self._image.n_channels == 3:
            grad = self._image.as_greyscale().gradient()
        else:
            grad = self._image.gradient()
        grad_orient = np.angle(grad.pixels[..., 0] + 1j*grad.pixels[..., 1])
        if double_angles:
            igo = np.concatenate((np.cos(grad_orient)[..., np.newaxis],
                                  np.sin(grad_orient)[..., np.newaxis],
                                  np.cos(2*grad_orient)[..., np.newaxis],
                                  np.sin(2*grad_orient)[..., np.newaxis]), 2)
        else:
            igo = np.concatenate((np.cos(grad_orient)[..., np.newaxis],
                                  np.sin(grad_orient)[..., np.newaxis]), 2)
        # print information
        if verbose:
            info_str = "IGO Features:\n" \
                       "  - Input image is {}W x {}H with {} channels.\n"\
                .format(self._image.pixels.shape[1],
                        self._image.pixels.shape[0],
                        self._image.pixels.shape[2])
            if double_angles:
                info_str = "{}  - Double angles are enabled.\n"\
                    .format(info_str)
            else:
                info_str = "{}  - Double angles are disabled.\n"\
                    .format(info_str)
            info_str = "{}Output image size {}W x {}H x {}."\
                .format(info_str, igo.shape[0], igo.shape[1], igo.shape[2])
            print info_str
        # create igo image object
        igo_image = self._init_feature_image(igo)
        # store parameters
        igo_image.parameters = {'double_angles': double_angles,
                                'original_image_height':
                                self._image.pixels.shape[0],
                                'original_image_width':
                                self._image.pixels.shape[1],
                                'original_image_channels':
                                self._image.pixels.shape[2]}
        return igo_image

    def _init_feature_image(self, feature_pixels, window_centres=None):
        r"""
        Creates a new image object to store the feature_pixels. If the original
        object is of MaskedImage class, then the features object is of
        MaskedImage as well. If the original object is of any other image
        class, the output object is of Image class.

        Parameter
        ---------
        feature_pixels :  ndarray.
            The pixels of the features image.
        window_centres :  ndarray.
            The sampled pixels from where the features were extracted. It has
            size n_rows x n_columns x 2, where window_centres[:, :, 0] are the
            row indices and window_centres[:, :, 1] are the column indices.
        """
        from pybug.image import MaskedImage, Image
        if isinstance(self._image, MaskedImage):
            # if we have a MaskedImage object
            feature_image = MaskedImage(feature_pixels)
            # fix mask
            self.transfer_mask(feature_image, window_centres=window_centres)
        else:
            # if we have an Image object
            feature_image = Image(feature_pixels)
        # fix landmarks
        self.transfer_landmarks(feature_image, window_centres=window_centres)
        if window_centres is not None:
            feature_image.window_centres = window_centres
        return feature_image

    def transfer_landmarks(self, target_image, window_centres=None):
        r"""
        Transfers its own landmarks to the target_image object after
        appropriately correcting them. The landmarks correction is achieved
        based on the windows_centres of the features object.

        Parameters
        ----------
        target_image :  Either MaskedImage or Image class.
            The target image object that includes the windows_centres.

        window_centres : ndarray, optional
            If set, use these window centres to rescale the landmarks
            appropriately. If None, no scaling is applied.
        """
        target_image.landmarks = self._image.landmarks
        if window_centres is not None:
            if target_image.landmarks.has_landmarks:
                for l_group in target_image.landmarks:
                    l = target_image.landmarks[l_group[0]]
                    # make sure window steps are in pixels mode
                    window_step_vertical = \
                        (window_centres[1, 0, 0] -
                         window_centres[0, 0, 0])
                    window_step_horizontal = \
                        (window_centres[0, 1, 1] -
                         window_centres[0, 0, 1])
                    # convert points by subtracting offset (controlled by
                    # padding)
                    # and dividing with step at each direction
                    l.lms.points[:, 0] = \
                        (l.lms.points[:, 0] -
                         window_centres[:, :, 0].min()) / \
                        window_step_vertical
                    l.lms.points[:, 1] = \
                        (l.lms.points[:, 1] -
                         window_centres[:, :, 1].min()) / \
                        window_step_horizontal

    def transfer_mask(self, target_image, window_centres=None):
        r"""
        Transfers its own mask to the target_image object after
        appropriately correcting it. The mask correction is achieved based on
        the windows_centres of the features object.

        Parameters
        ----------
        target_image :  Either MaskedImage or Image class.
            The target image object that includes the windows_centres.

        window_centres : ndarray, optional
            If set, use these window centres to rescale the landmarks
            appropriately. If None, no scaling is applied.
        """
        from pybug.image import BooleanImage
        mask = self._image.mask.mask  # don't want a channel axis!
        if window_centres is not None:
            mask = mask[window_centres[..., 0], window_centres[..., 1]]
        target_image.mask = BooleanImage(mask.copy())
