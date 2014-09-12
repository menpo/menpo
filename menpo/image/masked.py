from __future__ import division
from warnings import warn
import numpy as np
binary_erosion = None  # expensive, from scipy.ndimage

from menpo.visualize.base import ImageViewer
gradient = None  # avoid circular reference, from menpo.feature

from .base import Image
from .boolean import BooleanImage


class MaskedImage(Image):
    r"""
    Represents an n-dimensional k-channel image, which has a mask.
    Images can be masked in order to identify a region of interest. All
    images implicitly have a mask that is defined as the the entire image.
    The mask is an instance of :map:`BooleanImage`.

    Parameters
    ----------
    image_data :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.
    mask : (M, N) `np.bool` ndarray or :map:`BooleanImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :map:`BooleanImage` covering the whole image

    copy: bool, optional
        If False, the image_data will not be copied on assignment. If a mask is
        provided, this also won't be copied.
        In general this should only be used if you know what you are doing.

        Default False
    Raises
    ------
    ValueError
        Mask is not the same shape as the image

    """
    def __init__(self, image_data, mask=None, copy=True):
        super(MaskedImage, self).__init__(image_data, copy=copy)
        if mask is not None:
            # Check if we need to create a BooleanImage or not
            if not isinstance(mask, BooleanImage):
                # So it's a numpy array.
                mask_image = BooleanImage(mask, copy=copy)
            else:
                # It's a BooleanImage object.
                if copy:
                    mask = mask.copy()
                mask_image = mask
            if mask_image.shape == self.shape:
                self.mask = mask_image
            else:
                raise ValueError("Trying to construct a Masked Image of "
                                 "shape {} with a Mask of differing "
                                 "shape {}".format(self.shape,
                                                   mask.shape))
        else:
            # no mask provided - make the default.
            self.mask = BooleanImage.blank(self.shape, fill=True)

    def as_unmasked(self, copy=True):
        r"""
        Return a copy of this image without the masking behavior.

        By default the mask is simply discarded. In the future more options
        may be possible.

        Parameters
        ----------
        copy : `bool`, optional
            If False, the produced :map:`Image` will share pixels with
            ``self``. Only suggested to be used for performance.

        Returns
        -------
        image : :map:`Image`
            An image with the same pixels and landmarks as this one, but with
            no mask.
        """
        img = Image(self.pixels, copy=copy)
        img.landmarks = self.landmarks
        return img

    @classmethod
    def blank(cls, shape, n_channels=1, fill=0, dtype=np.float, mask=None):
        r"""
        Returns a blank image

        Parameters
        ----------
        shape : tuple or list
            The shape of the image. Any floating point values are rounded up
            to the nearest integer.

        n_channels: int, optional
            The number of channels to create the image with

            Default: 1
        fill : int, optional
            The value to fill all pixels with

            Default: 0
        dtype: numpy datatype, optional
            The datatype of the image.

            Default: np.float
        mask: (M, N) boolean ndarray or :class:`BooleanImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanImage`

        Notes
        -----
        Subclasses of `MaskedImage` need to overwrite this method and
        explicitly call this superclass method:

            super(SubClass, cls).blank(shape,**kwargs)

        in order to appropriately propagate the SubClass type to cls.

        Returns
        -------
        blank_image : :class:`MaskedImage`
            A new masked image of the requested size.
        """
        # Ensure that the '+' operator means concatenate tuples
        shape = tuple(np.ceil(shape).astype(np.int))
        if fill == 0:
            pixels = np.zeros(shape + (n_channels,), dtype=dtype)
        else:
            pixels = np.ones(shape + (n_channels,), dtype=dtype) * fill
        return cls(pixels, copy=False, mask=mask)

    @property
    def n_true_pixels(self):
        return self.mask.n_true

    @property
    def n_false_pixels(self):
        return self.mask.n_false

    @property
    def n_true_elements(self):
        return self.n_true_pixels * self.n_channels

    @property
    def n_false_elements(self):
        return self.n_false_pixels * self.n_channels

    @property
    def indices(self):
        r"""
        Return the indices of all true pixels in this image.

        :type: (`n_dims`, `n_true_pixels`) ndarray

        """
        return self.mask.true_indices

    @property
    def masked_pixels(self):
        r"""
        Get the pixels covered by the `True` values in the mask.

        :type: (`mask.n_true`, `n_channels`) ndarray
        """
        if self.mask.all_true:
            return self.pixels
        return self.pixels[self.mask.mask]

    def set_masked_pixels(self, pixels, copy=True):
        r"""Update the masked pixels only to new values.

        Parameters
        ----------
        pixels: ndarray
            The new pixels to set.

        copy: `bool`, optional
            If False a copy will be avoided in assignment. This can only happen
            if the mask is all True - in all other cases it will raise a
            warning.

        Raises
        ------

        Warning : If the copy=False flag cannot be honored.

        """
        if self.mask.all_true:
            # reshape the vector into the image again
            pixels = pixels.reshape(self.shape + (self.n_channels,))
            if not copy:
                if not pixels.flags.c_contiguous:
                    warn('The copy flag was NOT honoured. A copy HAS been '
                         'made. Copy can only be avoided if MaskedImage has '
                         'an all_true mask and the pixels provided are '
                         'C-contiguous.')
                    pixels = pixels.copy()
            else:
                pixels = pixels.copy()
            self.pixels = pixels
        else:
            self.pixels[self.mask.mask] = pixels
            # oh dear, couldn't avoid a copy. Did the user try to?
            if not copy:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                    'copy can only be avoided if MaskedImage has an all_true'
                    'mask.')

    def __str__(self):
        return ('{} {}D MaskedImage with {} channels. '
                'Attached mask {:.1%} true'.format(
            self._str_shape, self.n_dims, self.n_channels,
            self.mask.proportion_true))

    def _as_vector(self, keep_channels=False):
        r"""
        Convert image to a vectorized form. Note that the only pixels
        returned here are from the masked region on the image.

        Parameters
        ----------
        keep_channels : bool, optional

            ========== ====================================
            Value      Return shape
            ========== ====================================
            `True`     (`mask.n_true`,`n_channels`)
            `False`    (`mask.n_true` x `n_channels`,)
            ========== ====================================

            Default: `False`

        Returns
        -------
        vectorized_image : (shape given by `keep_channels`) ndarray
            Vectorized image
        """
        if keep_channels:
            return self.masked_pixels.reshape([-1, self.n_channels])
        else:
            return self.masked_pixels.ravel()

    def from_vector(self, vector, n_channels=None):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        On masked images, the vector is always copied.

        The `n_channels` argument is useful for when we want to add an extra
        channel to an image but maintain the shape. For example, when
        calculating the gradient.

        Note that landmarks are transferred in the process.

        Parameters
        ----------
        vector : (`n_pixels`,)
            A flattened vector of all pixels and channels of an image.
        n_channels : int, optional
            If given, will assume that vector is the same
            shape as this image, but with a possibly different number of
            channels

            Default: Use the existing image channels

        Returns
        -------
        image : :class:`MaskedImage`
            New image of same shape as this image and the number of
            specified channels.

        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels is None else n_channels
        # Creates zeros of size (M x N x ... x n_channels)
        if self.mask.all_true:
            # we can just reshape the array!
            image_data = vector.reshape((self.shape + (n_channels,)))
        else:
            image_data = np.zeros(self.shape + (n_channels,))
            pixels_per_channel = vector.reshape((-1, n_channels))
            image_data[self.mask.mask] = pixels_per_channel
        new_image = MaskedImage(image_data, mask=self.mask)
        new_image.landmarks = self.landmarks
        return new_image

    def from_vector_inplace(self, vector, copy=True):
        r"""
        Takes a flattened vector and updates this image by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        Parameters
        ----------
        vector : (`n_parameters`,)
            A flattened vector of all pixels and channels of an image.

        copy: `bool`, optional
            If False, the vector will be set as the pixels with no copy made.
            If True a copy of the vector is taken.

            Default: True

        Raises
        ------

        Warning : If copy=False cannot be honored.

        """
        self.set_masked_pixels(vector.reshape((-1, self.n_channels)),
                               copy=copy)

    def _view(self, figure_id=None, new_figure=False, channels=None,
              masked=True, **kwargs):
        r"""
        View the image using the default image viewer. Currently only
        supports the rendering of 2D images.

        Returns
        -------
        image_viewer : :class:`menpo.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within

        Raises
        ------
        DimensionalityError
            If Image is not 2D
        """
        mask = self.mask.mask if masked else None
        pixels_to_view = self.pixels
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channels=channels,
                           mask=mask).render(**kwargs)

    def crop_inplace(self, min_indices, max_indices,
                     constrain_to_boundary=True):
        r"""
        Crops this image using the given minimum and maximum indices.
        Landmarks are correctly adjusted so they maintain their position
        relative to the newly cropped image.

        Parameters
        -----------
        min_indices: (n_dims, ) ndarray
            The minimum index over each dimension

        max_indices: (n_dims, ) ndarray
            The maximum index over each dimension

        constrain_to_boundary: boolean, optional
            If True the crop will be snapped to not go beyond this images
            boundary. If False, a ImageBoundaryError will be raised if an
            attempt is made to go beyond the edge of the image.

            Default: True

        Returns
        -------
        cropped_image : :class:`type(self)`
            This image, but cropped.

        Raises
        ------
        ValueError
            min_indices and max_indices both have to be of length n_dims.
            All max_indices must be greater than min_indices.

        ImageBoundaryError
            Raised if constrain_to_boundary is False, and an attempt is made
            to crop the image in a way that violates the image bounds.

        """
        # crop our image
        super(MaskedImage, self).crop_inplace(
            min_indices, max_indices,
            constrain_to_boundary=constrain_to_boundary)
        # crop our mask
        self.mask.crop_inplace(min_indices, max_indices,
                               constrain_to_boundary=constrain_to_boundary)
        return self

    def crop_to_true_mask(self, boundary=0, constrain_to_boundary=True):
        r"""
        Crop this image to be bounded just the `True` values of it's mask.

        Parameters
        ----------

        boundary: int, Optional
            An extra padding to be added all around the true mask region.

            Default: 0

        constrain_to_boundary: boolean, optional
            If `True` the crop will be snapped to not go beyond this images
            boundary. If `False`, a ImageBoundaryError will be raised if an
            attempt is made to go beyond the edge of the image. Note that is
            only possible if boundary != 0.

            Default: `True`

        Raises
        ------
        ImageBoundaryError
            Raised if constrain_to_boundary is `False`, and an attempt is
            made to crop the image in a way that violates the image bounds.
        """
        min_indices, max_indices = self.mask.bounds_true(
            boundary=boundary, constrain_to_bounds=False)
        # no point doing the bounds check twice - let the crop do it only.
        self.crop_inplace(min_indices, max_indices,
                          constrain_to_boundary=constrain_to_boundary)

    def warp_to_mask(self, template_mask, transform, warp_landmarks=False,
                     order=1, mode='constant', cval=0.):
        r"""
        Warps this image into a different reference space.

        Parameters
        ----------
        template_mask : :map:`BooleanImage`
            Defines the shape of the result, and what pixels should be
            sampled.

        transform : :map:`Transform`
            Transform **from the template space back to this image**.
            Defines, for each pixel location on the template, which pixel
            location should be sampled from on this image.

        warp_landmarks : `bool`, optional
            If `True`, warped_image will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        mode : `str`, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').

        cval : `float`, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        warped_image : ``type(self)``
            A copy of this image, warped.
        """
        # call the super variant and get ourselves a MaskedImage back
        # with a blank mask
        warped_image = Image.warp_to_mask(self, template_mask, transform,
                                          warp_landmarks=warp_landmarks,
                                          order=order, mode=mode, cval=cval)
        warped_mask = self.mask.warp_to_mask(template_mask, transform,
                                             warp_landmarks=warp_landmarks,
                                             mode=mode, cval=cval)
        warped_image.mask = warped_mask
        return warped_image

    def warp_to_shape(self, template_shape, transform, warp_landmarks=False,
                      order=1, mode='constant', cval=0.):
        """
        Return a copy of this :map:`MaskedImage` warped into a different
        reference space.

        Parameters
        ----------
        template_shape : (n_dims, ) tuple or ndarray
            Defines the shape of the result, and what pixel indices should be
            sampled (all of them).

        transform : :map:`Transform`
            Transform **from the template_shape space back to this image**.
            Defines, for each index on template_shape, which pixel location
            should be sampled from on this image.

        warp_landmarks : `bool`, optional
            If `True`, ``warped_image`` will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        mode : `str`, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').

        cval : `float`, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        warped_image : :map:`MaskedImage`
            A copy of this image, warped.
        """
        # call the super variant and get ourselves an Image back
        warped_image = Image.warp_to_shape(self, template_shape, transform,
                                           warp_landmarks=warp_landmarks,
                                           order=order, mode=mode, cval=cval)
        # warp the mask separately and reattach.
        mask = self.mask.warp_to_shape(template_shape, transform,
                                       warp_landmarks=warp_landmarks,
                                       mode=mode, cval=cval)
        # efficiently turn the Image into a MaskedImage, attaching the
        # landmarks
        masked_warped_image = MaskedImage(warped_image.pixels, mask=mask,
                                          copy=False)
        masked_warped_image.landmarks = warped_image.landmarks
        return masked_warped_image

    def normalize_std_inplace(self, mode='all', limit_to_mask=True):
        r"""
        Normalizes this image such that it's pixel values have zero mean and
        unit variance.

        Parameters
        ----------

        mode : {'all', 'per_channel'}
            If 'all', the normalization is over all channels. If
            'per_channel', each channel individually is mean centred and
            normalized in variance.
        limit_to_mask : `bool`
            If ``True``, the normalization is only performed wrt the masked
            pixels.
            If ``False``, the normalization is wrt all pixels, regardless of
            their masking value.
        """
        self._normalize_inplace(np.std, mode=mode,
                                limit_to_mask=limit_to_mask)

    def normalize_norm_inplace(self, mode='all', limit_to_mask=True,
                               **kwargs):
        r"""
        Normalizes this image such that it's pixel values have zero mean and
        its norm equals 1.

        Parameters
        ----------

        mode : {'all', 'per_channel'}
            If 'all', the normalization is over all channels. If
            'per_channel', each channel individually is mean centred and
            normalized in variance.

        limit_to_mask : `bool`
            If ``True``, the normalization is only performed wrt the masked
            pixels.
            If ``False``, the normalization is wrt all pixels, regardless of
            their masking value.
        """

        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        self._normalize_inplace(scale_func, mode=mode,
                                limit_to_mask=limit_to_mask)

    def _normalize_inplace(self, scale_func, mode='all', limit_to_mask=True):
        if limit_to_mask:
            pixels = self.as_vector(keep_channels=True)
        else:
            pixels = Image.as_vector(self, keep_channels=True)
        if mode == 'all':
            centered_pixels = pixels - np.mean(pixels)
            scale_factor = scale_func(centered_pixels)

        elif mode == 'per_channel':
            centered_pixels = pixels - np.mean(pixels, axis=0)
            scale_factor = scale_func(centered_pixels, axis=0)
        else:
            raise ValueError("mode has to be 'all' or 'per_channel' - '{}' "
                             "was provided instead".format(mode))

        if np.any(scale_factor == 0):
            raise ValueError("Image has 0 variance - can't be "
                             "normalized")
        else:
            normalized_pixels = centered_pixels / scale_factor

        if limit_to_mask:
            self.from_vector_inplace(normalized_pixels.flatten())
        else:
            Image.from_vector_inplace(self,
                                      normalized_pixels.flatten())

    def gradient(self, nullify_values_at_mask_boundaries=False):
        r"""
        Returns a :map:`MaskedImage` which is the gradient of this one. In the
        case of multiple channels, it returns the gradient over each axis over
        each channel as a flat list.

        Parameters
        ----------
        nullify_values_at_mask_boundaries : bool, optional
            If `True` a one pixel boundary is set to 0 around the edge of
            the `True` mask region. This is useful in situations where
            there is absent data in the image which will cause erroneous
            gradient settings.

        Default: False

        Returns
        -------
        gradient : :map:`MaskedImage`
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length `2`.
            The length of a 2D, 3-channel image, will have length `6`.
        """
        global binary_erosion, gradient
        if gradient is None:
            from menpo.feature import gradient  # avoid circular reference
        # use the feature to take the gradient as normal
        grad_image = gradient(self)
        if nullify_values_at_mask_boundaries:
            if binary_erosion is None:
                from scipy.ndimage import binary_erosion  # expensive
            # Erode the edge of the mask in by one pixel
            eroded_mask = binary_erosion(self.mask.mask, iterations=1)

            # replace the eroded mask with the diff between the two
            # masks. This is only true in the region we want to nullify.
            np.logical_and(~eroded_mask, self.mask.mask, out=eroded_mask)
            # nullify all the boundary values in the grad image
            grad_image.pixels[eroded_mask] = 0.0
        return grad_image

    def constrain_mask_to_landmarks(self, group=None, label=None,
                                    trilist=None):
        r"""
        Restricts this image's mask to be equal to the convex hull
        around the landmarks chosen.

        Parameters
        ----------
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: None

        trilist: (t, 3) ndarray, Optional
            Triangle list to be used on the landmarked points in selecting
            the mask region. If None defaults to performing Delaunay
            triangulation on the points.

            Default: None
        """
        self.mask.constrain_to_pointcloud(self.landmarks[group][label],
                                          trilist=trilist)

    def build_mask_around_landmarks(self, patch_size, group=None, label=None):
        r"""
        Restricts this image's mask to be equal to the convex hull
        around the landmarks chosen.

        Parameters
        ----------
        patch_shape : tuple
            The size of the patch. Any floating point values are rounded up
            to the nearest integer.
        group : `string`, optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: `None`
        label : `string`, optional
            The label of of the landmark manager that you wish to use. If
            `None` all landmarks are used.

            Default: `None`
        """
        pc = self.landmarks[group][label]
        patch_size = np.ceil(patch_size)
        patch_half_size = patch_size / 2
        mask = np.zeros(self.shape)
        max_x = self.shape[0] - 1
        max_y = self.shape[1] - 1

        for i, point in enumerate(pc.points):
            start = np.floor(point - patch_half_size).astype(int)
            finish = np.floor(point + patch_half_size).astype(int)
            x, y = np.mgrid[start[0]:finish[0], start[1]:finish[1]]
            # deal with boundary cases
            x[x > max_x] = max_x
            y[y > max_y] = max_y
            x[x < 0] = 0
            y[y < 0] = 0
            mask[x.flatten(), y.flatten()] = True

        self.mask = BooleanImage(mask)
