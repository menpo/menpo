from __future__ import division
from copy import deepcopy

import numpy as np
from scipy.ndimage import binary_erosion

from menpo.visualize.base import ImageViewer

from .base import Image
from .boolean import BooleanImage
from .feature import features


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
            if copy:
                pixels = pixels.copy()
            # Our mask is all True, so if they don't want a copy
            # we can respect their wishes
            self.pixels = pixels.reshape(self.shape + (self.n_channels,))
        else:
            self.pixels[self.mask.mask] = pixels
            # oh dear, couldn't avoid a copy. Did the user try to?
            if not copy:
                raise Warning('The copy flag was NOT honoured. '
                              'A copy HAS been made. copy can only be avoided'
                              ' if MaskedImage has an all_true mask.')

    def __str__(self):
        return ('{} {}D MaskedImage with {} channels. '
                'Attached mask {:.1%} true'.format(
            self._str_shape, self.n_dims, self.n_channels,
            self.mask.proportion_true))

    def copy(self):
        r"""
        Return a new image with copies of the pixels, landmarks, and masks of
        this image.

        This is an efficient copy method. If you need to copy all the state on
        the object, consider deepcopy instead.

        Returns
        -------

        image: :map:`MaskedImage`
            A new image with the same pixels, mask and landmarks as this one,
            just copied.

        """
        new_image = MaskedImage(self.pixels, mask=self.mask)
        new_image.landmarks = self.landmarks
        return new_image

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

    def warp_to(self, template_mask, transform, warp_landmarks=False,
                warp_mask=False, interpolator='scipy', **kwargs):
        r"""
        Warps this image into a different reference space.

        Parameters
        ----------
        template_mask : :class:`menpo.image.boolean.BooleanImage`
            Defines the shape of the result, and what pixels should be
            sampled.
        transform : :class:`menpo.transform.base.Transform`
            Transform **from the template space back to this image**.
            Defines, for each pixel location on the template, which pixel
            location should be sampled from on this image.
        warp_landmarks : bool, optional
            If `True`, warped_image will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

            Default: `False`
        warp_mask : bool, optional
            If `True`, sample the `image.mask` at all `template_image`
            points, setting the returned image mask to the sampled value
            **within the masked region of `template_image`**.

            Default: `False`

            .. note::

                This is most commonly set `True` in combination with an all
                True `template_mask`, as this is then a warp of the image
                and it's full mask. If `template_mask`
                has False mask values, only the True region of the mask
                will be updated, which is rarely the desired behavior,
                but is possible for completion.
        interpolator : 'scipy', optional
            The interpolator that should be used to perform the warp.

            Default: 'scipy'
        kwargs : dict
            Passed through to the interpolator. See `menpo.interpolation`
            for details.

        Returns
        -------
        warped_image : type(self)
            A copy of this image, warped.
        """
        warped_image = Image.warp_to(self, template_mask, transform,
                                     warp_landmarks=warp_landmarks,
                                     interpolator=interpolator,
                                     **kwargs)
        # note that _build_warped_image for MaskedImage classes attaches
        # the template mask by default. If the user doesn't want to warp the
        # mask, we are done. If they do want to warp the mask, we warp the
        # mask separately and reattach.
        # TODO an optimisation could be added here for the case where mask
        # is all true/all false.
        if warp_mask:
            warped_mask = self.mask.warp_to(template_mask, transform,
                                            warp_landmarks=warp_landmarks,
                                            interpolator=interpolator,
                                            **kwargs)
            warped_image.mask = warped_mask
        return warped_image

    def normalize_std_inplace(self, mode='all', limit_to_mask=True):

        r"""
        Normalizes this image such that it's pixel values have zero mean and
        unit variance.

        Parameters
        ----------

        mode: {'all', 'per_channel'}
            If 'all', the normalization is over all channels. If
            'per_channel', each channel individually is mean centred and
            normalized in variance.

        limit_to_mask: Boolean
            If True, the normalization is only performed wrt the masked
            pixels.
            If False, the normalization is wrt all pixels, regardless of
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

        mode: {'all', 'per_channel'}
            If 'all', the normalization is over all channels. If
            'per_channel', each channel individually is mean centred and
            normalized in variance.

        limit_to_mask: Boolean
            If True, the normalization is only performed wrt the masked
            pixels.
            If False, the normalization is wrt all pixels, regardless of
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

    def _build_warped_image(self, template_mask, sampled_pixel_values,
                            **kwargs):
        r"""
        Builds the warped image from the template mask and
        sampled pixel values. Overridden for BooleanImage as we can't use
        the usual from_vector_inplace method.
        """
        return super(MaskedImage, self)._build_warped_image(
            template_mask, sampled_pixel_values, mask=template_mask)

    def gradient(self, nullify_values_at_mask_boundaries=False):
        r"""
        Returns a MaskedImage which is the gradient of this one. In the case
        of multiple channels, it returns the gradient over each axis over
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
        gradient : :class:`MaskedImage`
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length `2`.
            The length of a 2D, 3-channel image, will have length `6`.
        """
        grad_image_pixels = features.gradient(self.pixels)
        grad_image = MaskedImage(grad_image_pixels,
                                 mask=deepcopy(self.mask))

        if nullify_values_at_mask_boundaries:
            # Erode the edge of the mask in by one pixel
            eroded_mask = binary_erosion(self.mask.mask, iterations=1)

            # replace the eroded mask with the diff between the two
            # masks. This is only true in the region we want to nullify.
            np.logical_and(~eroded_mask, self.mask.mask, out=eroded_mask)
            # nullify all the boundary values in the grad image
            grad_image.pixels[eroded_mask] = 0.0
        grad_image.landmarks = self.landmarks
        return grad_image

    # TODO maybe we should be stricter about the trilist here, feels flakey
    def constrain_mask_to_landmarks(self, group=None, label='all',
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
        from menpo.transform.piecewiseaffine import PiecewiseAffine
        from menpo.transform.piecewiseaffine import TriangleContainmentError

        if self.n_dims != 2:
            raise ValueError("can only constrain mask on 2D images.")

        pc = self.landmarks[group][label].lms
        if trilist is not None:
            from menpo.shape import TriMesh

            pc = TriMesh(pc.points, trilist)

        pwa = PiecewiseAffine(pc, pc)
        try:
            pwa.apply(self.indices)
        except TriangleContainmentError, e:
            self.mask.from_vector_inplace(~e.points_outside_source_domain)

    def rescale(self, scale, interpolator='scipy', round='ceil', **kwargs):
        r"""A copy of this MaskedImage, rescaled by a given factor.

        All image information (landmarks and mask) are rescaled appropriately.

        Parameters
        ----------
        scale : float or tuple
            The scale factor. If a tuple, the scale to apply to each dimension.
            If a single float, the scale will be applied uniformly across
            each dimension.
        round: {'ceil', 'floor', 'round'}
            Rounding function to be applied to floating point shapes.

            Default: 'ceil'
        kwargs : dict
            Passed through to the interpolator. See `menpo.interpolation`
            for details.

        Returns
        -------
        rescaled_image : type(self)
            A copy of this image, rescaled.

        Raises
        ------
        ValueError:
            If less scales than dimensions are provided.
            If any scale is less than or equal to 0.
        """
        # just call normal Image version, passing the warp_mask=True flag
        return super(MaskedImage, self).rescale(scale,
                                                interpolator=interpolator,
                                                round=round,
                                                warp_mask=True,
                                                **kwargs)

    def build_mask_around_landmarks(self, patch_size, group=None,
                                    label='all'):
        r"""
        Restricts this image's mask to be equal to the convex hull
        around the landmarks chosen.

        Parameters
        ----------
        patch_shape: tuple
            The size of the patch. Any floating point values are rounded up
            to the nearest integer.
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None
        label: string, Optional
            The label of of the landmark manager that you wish to use. If
            'all' all landmarks are used.

            Default: 'all'
        """
        pc = self.landmarks[group][label].lms
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

    def gaussian_pyramid(self, n_levels=3, downscale=2, sigma=None, order=1,
                         mode='reflect', cval=0):
        r"""
        Return the gaussian pyramid of this image. The first image of the
        pyramid will be the original, unmodified, image.

        Parameters
        ----------
        n_levels : int
            Number of levels in the pyramid. When set to -1 the maximum
            number of levels will be build.

            Default: 3

        downscale : float, optional
            Downscale factor.

            Default: 2

        sigma : float, optional
            Sigma for gaussian filter. Default is `2 * downscale / 6.0` which
            corresponds to a filter mask twice the size of the scale factor
            that covers more than 99% of the gaussian distribution.

            Default: None

        order : int, optional
            Order of splines used in interpolation of downsampling. See
            `scipy.ndimage.map_coordinates` for detail.

            Default: 1

        mode :  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            The mode parameter determines how the array borders are handled,
            where cval is the value when mode is equal to 'constant'.

            Default: 'reflect'

        cval : float, optional
            Value to fill past edges of input if mode is 'constant'.

            Default: 0

        Returns
        -------
        image_pyramid:
            Generator yielding pyramid layers as menpo image objects.
        """
        image_pyramid = Image.gaussian_pyramid(
            self, n_levels=n_levels, downscale=downscale, sigma=sigma,
            order=order, mode=mode, cval=cval)
        for image in image_pyramid:
            image.mask = self.mask.resize(image.shape)
            yield image

    def smoothing_pyramid(self, n_levels=3, downscale=2, sigma=None,
                          mode='reflect', cval=0):
        r"""
        Return the smoothing pyramid of this image. The first image of the
        pyramid will be the original, unmodified, image.

        Parameters
        ----------
        n_levels : int
            Number of levels in the pyramid. When set to -1 the maximum
            number of levels will be build.

            Default: 3

        downscale : float, optional
            Downscale factor.

            Default: 2

        sigma : float, optional
            Sigma for gaussian filter. Default is `2 * downscale / 6.0` which
            corresponds to a filter mask twice the size of the scale factor
            that covers more than 99% of the gaussian distribution.

            Default: None

        mode :  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            The mode parameter determines how the array borders are handled,
            where cval is the value when mode is equal to 'constant'.

            Default: 'reflect'

        cval : float, optional
            Value to fill past edges of input if mode is 'constant'.

            Default: 0

        Returns
        -------
        image_pyramid:
            Generator yielding pyramid layers as menpo image objects.
        """
        image_pyramid = Image.smoothing_pyramid(
            self, n_levels=n_levels, downscale=downscale, sigma=sigma,
            mode=mode, cval=cval)
        for image in image_pyramid:
            image.mask = self.mask
            yield image
