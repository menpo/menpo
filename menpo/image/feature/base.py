import features as fc


class ImageFeatures(object):
    r"""
    Utility class that exposes feature computation on an image. The output
    image's class is either MaskedImage or Image depending on the input image.
    The mask and landmarks are corrected based on the features calculation.

    Parameters
    ----------
    image :  Any image object
        The object created by any image class (Image, MaskedImage).
    """

    def __init__(self, image):
        self._image = image

    def hog(self, mode='dense', algorithm='dalaltriggs', num_bins=9,
            cell_size=8, block_size=2, signed_gradient=True, l2_norm_clip=0.2,
            window_height=1, window_width=1, window_unit='blocks',
            window_step_vertical=1, window_step_horizontal=1,
            window_step_unit='pixels', padding=True, verbose=False,
            constrain_landmarks=True):
        r"""
        Represents a 2-dimensional HOG features image with C number of
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
        constrain_landmarks : bool
            Flag that if enabled, it constrains landmarks that ended up outside
            of the features image bounds.

            Default: True
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
                                             window_centres=window_centres,
                                             constrain_landmarks=
                                             constrain_landmarks)
        # store parameters
        hog_image.hog_parameters = {'mode': mode, 'algorithm': algorithm,
                                    'num_bins': num_bins,
                                    'cell_size': cell_size,
                                    'block_size': block_size,
                                    'signed_gradient': signed_gradient,
                                    'l2_norm_clip': l2_norm_clip,
                                    'window_height': window_height,
                                    'window_width': window_width,
                                    'window_unit': window_unit,
                                    'window_step_vertical':
                                    window_step_vertical,
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

    def igo(self, double_angles=False, constrain_landmarks=True,
            verbose=False):
        r"""
        Represents a 2-dimensional IGO features image with N*C number of
        channels, where N is the number of channels of the original image and
        C=[2,4] depending on whether double angles are used. The output
        object's class is either MaskedImage or Image depending on the original
        image.

        Parameters
        ----------
        image :  class:`Image`
            An image object that contains pixels and mask fields.
        double_angles : bool
            Assume that phi represents the gradient orientations. If this flag
            is disabled, the features image is the concatenation of cos(phi)
            and sin(phi), thus 2 channels. If it is enabled, the features image
            is the concatenation of cos(phi), sin(phi), cos(2*phi), sin(2*phi),
            thus 4 channels.

            Default: False
        constrain_landmarks : bool
            Flag that if enabled, it constrains landmarks that ended up outside
            of the features image bounds.

            Default: True
        verbose : bool
            Flag to print IGO related information.

            Default: False

        Raises
        -------
        ValueError
            Image has to be 2D in order to extract IGOs.
        """
        # compute igo features
        igo = fc.igo(self._image.pixels, double_angles=double_angles,
                     verbose=verbose)
        # create igo image object
        igo_image = self._init_feature_image(igo, constrain_landmarks=
                                             constrain_landmarks)
        # store parameters
        igo_image.igo_parameters = {'double_angles': double_angles,
                                    'original_image_height':
                                    self._image.pixels.shape[0],
                                    'original_image_width':
                                    self._image.pixels.shape[1],
                                    'original_image_channels':
                                    self._image.pixels.shape[2]}
        return igo_image

    def es(self, constrain_landmarks=True, verbose=False):
        r"""
        Represents a 2-dimensional Edge Structure (ES) features image with N*C
        number of channels, where N is the number of channels of the original
        image and C=2. The output object's class is either MaskedImage or
        Image depending on the original image.

        Parameters
        ----------
        image :  class:`Image`
            An image object that contains pixels and mask fields.
        constrain_landmarks : bool
            Flag that if enabled, it constrains landmarks that ended up outside
            of the features image bounds.

            Default: True
        verbose : bool
            Flag to print IGO related information.

            Default: False

        Raises
        -------
        ValueError
            Image has to be 2D in order to extract ES features.
        """
        # compute es features
        es = fc.es(self._image.pixels, verbose=verbose)
        # create es image object
        es_image = self._init_feature_image(es, constrain_landmarks=
                                            constrain_landmarks)
        # store parameters
        es_image.es_parameters = {'original_image_height':
                                  self._image.pixels.shape[0],
                                  'original_image_width':
                                  self._image.pixels.shape[1],
                                  'original_image_channels':
                                  self._image.pixels.shape[2]}
        return es_image

    def lbp(self, radius=range(1, 5), samples=[8]*4, mapping_type='riu2',
            window_step_vertical=1, window_step_horizontal=1,
            window_step_unit='pixels', padding=True, verbose=False,
            constrain_landmarks=True):
        r"""
        Represents a 2-dimensional LBP features image with N*C number of
        channels, where N is the number of channels of the original image and
        C is the number of radius/samples values combinations that are used in
        the LBP computation. The output object's class is either MaskedImage or
        Image depending on the input image.

        Parameters
        ----------
        radius : int or list of integers
            It defines the radius of the circle (or circles) at which the
            sampling points will be extracted. The radius (or radii) values
            must be greater than zero. There must be a radius value for each
            samples value, thus they both need to have the same length.

            Default: [1, 2, 3, 4]
        samples : int or list of integers
            It defines the number of sampling points that will be extracted at
            each circle. The samples value (or values) must be greater than
            zero. There must be a samples value for each radius value, thus
            they both need to have the same length.

            Default: [8, 8, 8, 8]
        mapping_type : 'u2' or 'ri' or 'riu2' or 'none'
            It defines the mapping type of the LBP codes. Select 'u2' for
            uniform-2 mapping, 'ri' for rotation-invariant mapping, 'riu2' for
            uniform-2 and rotation-invariant mapping and 'none' to use no
            mapping and only the decimal values instead.

            Default: 'riu2'
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
        window_step_unit : 'pixels' or 'window'
            Defines the metric unit of the window_step_vertical and
            window_step_horizontal parameters for the ImageWindowIterator
            object.

            Default: 'pixels'
        padding : bool
            Enables/disables padding for the close-to-boundary windows in the
            ImageWindowIterator object. When padding is enabled, the
            out-of-boundary pixels are set to zero.

            Default: True
        verbose : bool
            Flag to print LBP related information.

            Default: False
        constrain_landmarks : bool
            Flag that if enabled, it constrains landmarks that ended up outside
            of the features image bounds.

            Default: True

        Raises
        -------
        ValueError
            Radius and samples must both be either integers or lists
        ValueError
            Radius and samples must have the same length
        ValueError
            Radius must be > 0
        ValueError
            Radii must be > 0
        ValueError
            Samples must be > 0
        ValueError
            Mapping type must be u2, ri, riu2 or none
        ValueError
            Horizontal window step must be > 0
        ValueError
            Vertical window step must be > 0
        ValueError
            Window step unit must be either pixels or window
        """
        # compute lbp features and windows_centres
        lbp, window_centres = fc.lbp(self._image.pixels, radius=radius,
                                     samples=samples,
                                     mapping_type=mapping_type,
                                     window_step_vertical=window_step_vertical,
                                     window_step_horizontal=
                                     window_step_horizontal,
                                     window_step_unit=window_step_unit,
                                     padding=padding, verbose=verbose)
        # create lbp image object
        lbp_image = self._init_feature_image(lbp,
                                             window_centres=window_centres,
                                             constrain_landmarks=
                                             constrain_landmarks)
        # store parameters
        lbp_image.lbp_parameters = {'radius': radius, 'samples': samples,
                                    'mapping_type': mapping_type,
                                    'window_step_vertical':
                                    window_step_vertical,
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
        return lbp_image

    def _init_feature_image(self, feature_pixels, window_centres=None,
                            constrain_landmarks=True):
        r"""
        Creates a new image object to store the feature_pixels. If the original
        object is of MaskedImage class, then the features object is of
        MaskedImage as well. If the original object is of any other image
        class, the output object is of Image class.

        Parameters
        ----------
        feature_pixels :  ndarray.
            The pixels of the features image.

        window_centres :  ndarray.
            The sampled pixels from where the features were extracted. It has
            size n_rows x n_columns x 2, where window_centres[:, :, 0] are the
            row indices and window_centres[:, :, 1] are the column indices.

        constrain_landmarks : bool
            Flag that if enabled, it constrains landmarks to image bounds.

            Default: True
        """
        from menpo.image import MaskedImage, Image
        if isinstance(self._image, MaskedImage):
            # if we have a MaskedImage object
            feature_image = MaskedImage(feature_pixels, copy=False)
            # fix mask
            self.transfer_mask(feature_image, window_centres=window_centres)
        else:
            # if we have an Image object
            feature_image = Image(feature_pixels, copy=False)
        # fix landmarks
        self.transfer_landmarks(feature_image, window_centres=window_centres,
                                constrain_landmarks=constrain_landmarks)
        if window_centres is not None:
            feature_image.window_centres = window_centres
        return feature_image

    def transfer_landmarks(self, target_image, window_centres=None,
                           constrain_landmarks=True):
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

        constrain_landmarks : bool
            Flag that if enabled, it constrains landmarks to image bounds.

            Default: True
        """
        target_image.landmarks = self._image.landmarks
        if window_centres is not None:
            if target_image.landmarks.has_landmarks:
                for l_group in target_image.landmarks:
                    l = target_image.landmarks[l_group]
                    # find the vertical and horizontal sampling steps
                    step_vertical = window_centres[0, 0, 0]
                    if window_centres.shape[0] > 1:
                        step_vertical = \
                            (window_centres[1, 0, 0] -
                             window_centres[0, 0, 0])
                    step_horizontal = window_centres[0, 0, 1]
                    if window_centres.shape[1] > 1:
                        step_horizontal = \
                            (window_centres[0, 1, 1] -
                             window_centres[0, 0, 1])
                    # convert points by subtracting offset and dividing with
                    # step at each direction
                    l.lms.points[:, 0] = \
                        (l.lms.points[:, 0] -
                         window_centres[:, :, 0].min()) / \
                        step_vertical
                    l.lms.points[:, 1] = \
                        (l.lms.points[:, 1] -
                         window_centres[:, :, 1].min()) / \
                        step_horizontal
        # constrain landmarks to image bounds if asked
        if constrain_landmarks:
            target_image.constrain_landmarks_to_bounds()

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
        from menpo.image import BooleanImage
        mask = self._image.mask.mask  # don't want a channel axis!
        if window_centres is not None:
            mask = mask[window_centres[..., 0], window_centres[..., 1]]
        target_image.mask = BooleanImage(mask.copy())
