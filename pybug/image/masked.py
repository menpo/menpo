from copy import deepcopy
import itertools
import numpy as np
from scipy.ndimage import binary_erosion
from pybug.image.base import AbstractNDImage
from pybug.image.boolean import BooleanNDImage
from pybug.visualize.base import ImageViewer


class MaskedNDImage(AbstractNDImage):
    r"""
    Represents an n-dimensional k-channel image, which has a mask.
    Images can be masked in order to identify a region of interest. All
    images implicitly have a mask that is defined as the the entire image.
    The mask is an instance of
    :class:`BooleanNDImage`.

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
    ------
    ValueError
        Mask is not the same shape as the image
    """
    def __init__(self, image_data, mask=None):
        super(MaskedNDImage, self).__init__(image_data)
        if mask is not None:
            if not isinstance(mask, BooleanNDImage):
                mask_image = BooleanNDImage(mask)
            else:
                mask_image = mask
            # have a BooleanNDImage object that we definitely own
            if mask_image.shape == self.shape:
                self.mask = mask_image
            else:
                raise ValueError("Trying to construct a Masked Image of "
                                 "shape {} with a Mask of differing "
                                 "shape {}".format(self.shape,
                                                   mask.shape))
        else:
            # no mask provided - make the default.
            self.mask = BooleanNDImage.blank(self.shape, fill=True)

    # noinspection PyMethodOverriding
    @classmethod
    def _init_with_channel(cls, image_data_with_channel, mask):
        r"""
        Constructor that always requires the image has a
        channel on the last axis. Only used by from_vector. By default,
        just calls the constructor. Subclasses with constructors that don't
        require channel axes need to overwrite this.
        """
        return cls(image_data_with_channel, mask)

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
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanNDImage`

        Notes
        -----
        Subclasses of `MaskedNDImage` need to overwrite this method and
        explicitly call this superclass method:

            super(SubClass, cls).blank(shape,**kwargs)

        in order to appropriately propagate the SubClass type to cls.

        Returns
        -------
        blank_image : :class:`MaskedNDImage`
            A new masked image of the requested size.
        """
        # Ensure that the '+' operator means concatenate tuples
        shape = tuple(np.ceil(shape))
        if fill == 0:
            pixels = np.zeros(shape + (n_channels,), dtype=dtype)
        else:
            pixels = np.ones(shape + (n_channels,), dtype=dtype) * fill
        return cls._init_with_channel(pixels, mask=mask)

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
    def masked_pixels(self):
        r"""
        Get the pixels covered by the ``True`` values in the mask.

        :type: (``mask.n_true``, ``n_channels``) ndarray
        """
        return self.pixels[self.mask.mask]

    @masked_pixels.setter
    def masked_pixels(self, value):
        self.pixels[self.mask.mask] = value

    def __str__(self):
        return ('{} {}D MaskedImage with {} channels. '
                'Attached mask {:.1%} true'.format(
                self._str_shape, self.n_dims, self.n_channels,
                self.mask.proportion_true))

    def as_vector(self, keep_channels=False):
        r"""
        Convert image to a vectorized form. Note that the only pixels
        returned here are from the masked region on the image.

        Parameters
        ----------
        keep_channels : bool, optional

            ========== ====================================
            Value      Return shape
            ========== ====================================
            ``True``   (``mask.n_true``,``n_channels``)
            ``False``  (``mask.n_true`` x ``n_channels``,)
            ========== ====================================

            Default: ``False``

        Returns
        -------
        vectorized_image : (shape given by ``keep_channels``) ndarray
            Vectorized image
        """
        if keep_channels:
            return self.masked_pixels.reshape([-1, self.n_channels])
        else:
            return self.masked_pixels.flatten()

    def from_vector(self, flattened, n_channels=None):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        The ``n_channels`` argument is useful for when we want to add an extra
        channel to an image but maintain the shape. For example, when
        calculating the gradient.

        Note that landmarks are transferred in the process.

        Parameters
        ----------
        flattened : (``n_pixels``,)
            A flattened vector of all pixels and channels of an image.
        n_channels : int, optional
            If given, will assume that flattened is the same
            shape as this image, but with a possibly different number of
            channels

            Default: Use the existing image channels

        Returns
        -------
        image : :class:`MaskedNDImage`
            New image of same shape as this image and the number of
            specified channels.
        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels is None else n_channels
        # Creates zeros of size (M x N x ... x n_channels)
        image_data = np.zeros(self.shape + (n_channels,))
        pixels_per_channel = flattened.reshape((-1, n_channels))
        image_data[self.mask.mask] = pixels_per_channel
        # call the constructor accounting for the fact that some image
        # classes expect a channel axis and some don't.
        new_image = type(self)._init_with_channel(image_data, mask=self.mask)
        new_image.landmarks = self.landmarks
        return new_image

    def from_vector_inplace(self, vector):
        r"""
        Takes a flattened vector and updates this image by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        Parameters
        ----------
        vector : (``n_pixels``,)
            A flattened vector of all pixels and channels of an image.
        """
        self.masked_pixels = vector.reshape((-1, self.n_channels))

    def _view(self, figure_id=None, new_figure=False, channel=None,
              masked=True, **kwargs):
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
        mask = None
        if masked:
            mask = self.mask.mask
        pixels_to_view = self.pixels
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channel=channel,
                           mask=mask).render(**kwargs)

    def crop(self, min_indices, max_indices,
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
        super(MaskedNDImage, self).crop(
            min_indices, max_indices,
            constrain_to_boundary=constrain_to_boundary)
        # crop our mask
        self.mask.crop(min_indices, max_indices,
                       constrain_to_boundary=constrain_to_boundary)
        return self

    def crop_to_true_mask(self, boundary=0, constrain_to_boundary=True):
        r"""
        Crop this image to be bounded just the ``True`` values of it's mask.

        Parameters
        ----------

        boundary: int, Optional
            An extra padding to be added all around the true mask region.

            Default: 0

        constrain_to_boundary: boolean, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, a ImageBoundaryError will be raised if an
            attempt is made to go beyond the edge of the image. Note that is
            only possible if boundary != 0.

            Default: ``True``

        Raises
        ------
        ImageBoundaryError
            Raised if constrain_to_boundary is ``False``, and an attempt is
            made to crop the image in a way that violates the image bounds.
        """
        min_indices, max_indices = self.mask.bounds_true(
            boundary=boundary, constrain_to_bounds=False)
        # no point doing the bounds check twice - let the crop do it only.
        self.crop(min_indices, max_indices,
                  constrain_to_boundary=constrain_to_boundary)

    def warp_to(self, template_mask, transform, warp_landmarks=False,
                warp_mask=False, interpolator='scipy', **kwargs):
        r"""
        Warps this image into a different reference space.

        Parameters
        ----------
        template_mask : :class:`pybug.image.boolean.BooleanNDImage`
            Defines the shape of the result, and what pixels should be
            sampled.
        transform : :class:`pybug.transform.base.Transform`
            Transform **from the template space back to this image**.
            Defines, for each pixel location on the template, which pixel
            location should be sampled from on this image.
        warp_landmarks : bool, optional
            If ``True``, warped_image will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

            Default: ``False``
        warp_mask : bool, optional
            If ``True``, sample the ``image.mask`` at all ``template_image``
            points, setting the returned image mask to the sampled value
            **within the masked region of ``template_image``**.

            Default: ``False``

            .. note::

                This is most commonly set ``True`` in combination with an all
                True ``template_mask``, as this is then a warp of the image
                and it's full mask. If ``template_mask``
                has False mask values, only the True region of the mask
                will be updated, which is rarely the desired behavior,
                but is possible for completion.
        interpolator : 'scipy' or 'cinterp' or func, optional
            The interpolator that should be used to perform the warp.

            Default: 'scipy'
        kwargs : dict
            Passed through to the interpolator. See `pybug.interpolation`
            for details.

        Returns
        -------
        warped_image : type(self)
            A copy of this image, warped.
        """
        warped_image = AbstractNDImage.warp_to(self, template_mask, transform,
                                               warp_landmarks=warp_landmarks,
                                               interpolator=interpolator,
                                               **kwargs)
        # note that _build_warped_image for MaskedNDImage classes attaches
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

    def normalize_inplace(self, mode='all', limit_to_mask=True):
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
        if limit_to_mask:
            pixels = self.as_vector(keep_channels=True)
        else:
            pixels = AbstractNDImage.as_vector(self, keep_channels=True)
        if mode == 'all':
            centered_pixels = pixels - np.mean(pixels)
            std_dev = np.std(centered_pixels)

        elif mode == 'per_channel':
            centered_pixels = pixels - np.mean(pixels, axis=0)
            std_dev = np.std(centered_pixels, axis=0)
        else:
            raise ValueError("mode has to be 'all' or 'per_channel' - '{}' "
                             "was provided instead".format(mode))

        if np.any(std_dev == 0):
            raise ValueError("Image has 0 variance - can't be "
                             "normalized")
        else:
            normalized_pixels = centered_pixels / std_dev

        if limit_to_mask:
            self.from_vector_inplace(normalized_pixels.flatten())
        else:
            AbstractNDImage.from_vector_inplace(self,
                                                normalized_pixels.flatten())

    def _build_warped_image(self, template_mask, sampled_pixel_values):
        r"""
        Builds the warped image from the template mask and
        sampled pixel values. Overridden for BooleanNDImage as we can't use
        the usual from_vector_inplace method. All other Image classes share
        this implementation.
        """
        warped_image = self.blank(template_mask.shape, mask=template_mask,
                                  n_channels=self.n_channels)
        warped_image.from_vector_inplace(sampled_pixel_values.flatten())
        return warped_image

    def gradient(self, nullify_values_at_mask_boundaries=False):
        r"""
        Returns a MaskedNDImage which is the gradient of this one. In the case
        of multiple channels, it returns the gradient over each axis over
        each channel as a flat list.

        Parameters
        ----------
        nullify_values_at_mask_boundaries : bool, optional
            If ``True`` a one pixel boundary is set to 0 around the edge of
            the ``True`` mask region. This is useful in situations where
            there is absent data in the image which will cause erroneous
            gradient settings.

        Default: False

        Returns
        -------
        gradient : :class:``MaskedNDImage``
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length ``2``.
            The length of a 2D, 3-channel image, will have length ``6``.
        """
        grad_per_dim_per_channel = [np.gradient(g) for g in
                                    np.rollaxis(self.pixels, -1)]
        # Flatten out the separate dims
        grad_per_channel = list(itertools.chain.from_iterable(
                                grad_per_dim_per_channel))
        # Add a channel axis for broadcasting
        grad_per_channel = [g[..., None] for g in grad_per_channel]
        # Concatenate gradient list into an array (the new_image)
        grad_image_pixels = np.concatenate(grad_per_channel, axis=-1)
        grad_image = MaskedNDImage(grad_image_pixels,
                                   mask=deepcopy(self.mask))

        if nullify_values_at_mask_boundaries:
            # Erode the edge of the mask in by one pixel
            eroded_mask = binary_erosion(self.mask.mask, iterations=1)

            # replace the eroded mask with the diff between the two
            # masks. This is only true in the region we want to nullify.
            np.logical_and(~eroded_mask, self.mask. mask, out=eroded_mask)
            # nullify all the boundary values in the grad image
            grad_image.pixels[eroded_mask] = 0.0
        grad_image.landmarks = self.landmarks
        return grad_image

    # TODO maybe we should be stricter about the trilist here, feels flakey
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
        from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
        from pybug.transform.piecewiseaffine import TriangleContainmentError

        if self.n_dims != 2:
            raise ValueError("can only constrain mask on 2D images.")

        pc = self.landmarks[group][label].lms
        if trilist is not None:
            from pybug.shape import TriMesh
            pc = TriMesh(pc.points, trilist)

        pwa = PiecewiseAffineTransform(pc, pc)
        try:
            pwa.apply_inplace(self.mask.all_indices)
        except TriangleContainmentError, e:
            self.mask.from_vector_inplace(~e.points_outside_source_domain)
