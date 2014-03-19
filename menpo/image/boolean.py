from copy import deepcopy
import numpy as np
from menpo.image.base import Image
from skimage.transform import pyramid_gaussian


class BooleanImage(Image):
    r"""
    A mask image made from binary pixels. The region of the image that is
    left exposed by the mask is referred to as the 'masked region'. The
    set of 'masked' pixels is those pixels corresponding to a True value in
    the mask.

    Parameters
    -----------
    mask_data : (M, N, ..., L) ndarray
        The binary mask data. Note that there is no channel axis - a 2D Mask
         Image is built from just a 2D numpy array of mask_data.
        Automatically coerced in to boolean values.
    """

    def __init__(self, mask_data):
        # Enforce boolean pixels, and add a channel dim
        mask_data = np.asarray(mask_data[..., None], dtype=np.bool)
        super(BooleanImage, self).__init__(mask_data)

    @classmethod
    def _init_with_channel(cls, image_data_with_channel):
        r"""
        Constructor that always requires the image has a
        channel on the last axis. Only used by from_vector. By default,
        just calls the constructor. Subclasses with constructors that don't
        require channel axes need to overwrite this.
        """
        return cls(image_data_with_channel[..., 0])

    @classmethod
    def blank(cls, shape, fill=True, round='ceil', **kwargs):
        r"""
        Returns a blank :class:`BooleanImage` of the requested shape

        Parameters
        ----------
        shape : tuple or list
            The shape of the image. Any floating point values are rounded
            according to the ``round`` kwarg.

        fill : True or False, optional
            The mask value to be set everywhere

            Default: True (masked region is the whole image - meaning the whole
                 image is exposed)
        round: {'ceil', 'floor', 'round'}
            Rounding function to be applied to floating point shapes.

            Default: 'ceil'

        Returns
        -------
        blank_image : :class:`BooleanImage`
            A blank mask of the requested size
        """
        if round not in ['ceil', 'round', 'floor']:
            raise ValueError('round must be either ceil, round or floor')
            # Ensure that the '+' operator means concatenate tuples
        shape = tuple(getattr(np, round)(shape))
        if fill:
            mask = np.ones(shape, dtype=np.bool)
        else:
            mask = np.zeros(shape, dtype=np.bool)
        return cls(mask)

    @property
    def mask(self):
        r"""
        Returns the pixels of the mask with no channel axis. This is what
        should be used to mask any k-dimensional image.

        :type: (M, N, ..., L), np.bool ndarray
        """
        return self.pixels[..., 0]

    @property
    def n_true(self):
        r"""
        The number of ``True`` values in the mask

        :type: int
        """
        return np.sum(self.pixels)

    @property
    def n_false(self):
        r"""
        The number of ``False`` values in the mask

        :type: int
        """
        return self.n_pixels - self.n_true

    @property
    def proportion_true(self):
        r"""
        The proportion of the mask which is ``True``

        :type: double
        """
        return (self.n_true * 1.0) / self.n_pixels

    @property
    def proportion_false(self):
        r"""
        The proportion of the mask which is ``False``

        :type: double
        """
        return (self.n_false * 1.0) / self.n_pixels

    @property
    def true_indices(self):
        r"""
        The indices of pixels that are true.

        :type: (``n_dims``, ``n_true``) ndarray
        """
        # Ignore the channel axis
        return np.vstack(np.nonzero(self.pixels[..., 0])).T

    @property
    def false_indices(self):
        r"""
        The indices of pixels that are false.

        :type: (``n_dims``, ``n_false``) ndarray
        """
        # Ignore the channel axis
        return np.vstack(np.nonzero(~self.pixels[..., 0])).T

    @property
    def all_indices(self):
        r"""
        Indices into all pixels of the mask, as consistent with
        true_indices and false_indices

        :type: (``n_dims``, ``n_pixels``) ndarray
        """
        return np.indices(self.shape).reshape([self.n_dims, -1]).T

    def __str__(self):
        return ('{} {}D mask, {:.1%} '
                'of which is True '.format(self._str_shape, self.n_dims,
                                           self.proportion_true))

    def from_vector(self, flattened):
        r"""
        Takes a flattened vector and returns a new
        :class:`BooleanImage` formed by
        reshaping the vector to the correct dimensions. Note that this is
        rebuilding a boolean image **itself** from boolean values. The mask
        is in no way interpreted in performing the operation, in contrast to
        MaskedImage, where only the masked region is used in from_vector()
        and as_vector(). Any image landmarks are transferred in the process.

        Parameters
        ----------
        flattened : (``n_pixels``,) np.bool ndarray
            A flattened vector of all the pixels of a BooleanImage.

        Returns
        -------
        image : :class:`BooleanImage`
            New BooleanImage of same shape as this image
        """
        mask = BooleanImage(flattened.reshape(self.shape))
        mask.landmarks = self.landmarks
        return mask

    def invert(self):
        r"""
        Inverts the current mask in place, setting all True values to False,
        and all False values to True.
        """
        self.pixels = ~self.pixels

    def inverted_copy(self):
        r"""
        Returns a copy of this Boolean image, which is inverted.

        Returns
        -------
        inverted_image: :class:`BooleanNSImage`
            An inverted copy of this boolean image.
        """
        inverse = deepcopy(self)
        inverse.invert()
        return inverse

    def bounds_true(self, boundary=0, constrain_to_bounds=True):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the True mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the True pixel values.

        Parameters
        ----------
        boundary : int, optional
            A number of pixels that should be added to the extent. A
            negative value can be used to shrink the bounds in.

            Default: 0

        constrain_to_bounds: bool, optional
            If True, the bounding extent is snapped to not go beyond
            the edge of the image. If False, the bounds are left unchanged.

            Default: True

        Returns
        --------
        min_b : (D,) ndarray
            The minimum extent of the True mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.

        max_b : (D,) ndarray
            The maximum extent of the True mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.
        """
        mpi = self.true_indices
        maxes = np.max(mpi, axis=0) + boundary
        mins = np.min(mpi, axis=0) - boundary
        if constrain_to_bounds:
            maxes = self.constrain_points_to_bounds(maxes)
            mins = self.constrain_points_to_bounds(mins)
        return mins, maxes

    def bounds_false(self, boundary=0, constrain_to_bounds=True):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the False mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the False pixel values.

        Parameters
        ----------
        boundary : int >= 0, optional
            A number of pixels that should be added to the extent. A
            negative value can be used to shrink the bounds in.

            Default: 0

        constrain_to_bounds: bool, optional
            If True, the bounding extent is snapped to not go beyond
            the edge of the image. If False, the bounds are left unchanged.

            Default: True

        Returns
        --------
        min_b : (D,) ndarray
            The minimum extent of the False mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.

        max_b : (D,) ndarray
            The maximum extent of the False mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.
        """
        return self.inverted_copy().bounds_true(
            boundary=boundary, constrain_to_bounds=constrain_to_bounds)

    def warp_to(self, template_mask, transform, warp_landmarks=False,
                interpolator='scipy', **kwargs):
        r"""
        Warps this BooleanImage into a different reference space.

        Parameters
        ----------
        template_mask : :class:`menpo.image.boolean.BooleanImage`
            Defines the shape of the result, and what pixels should be
            sampled.
        transform : :class:`menpo.transform.base.Transform`
            Transform **from the template space back to this image**.
            Defines, for each True pixel location on the template, which pixel
            location should be sampled from on this image.
        warp_landmarks : bool, optional
            If ``True``, warped_image will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

            Default: ``False``
        interpolator : 'scipy' or 'c', optional
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
        # enforce the order as 0, for this boolean data, then call super
        manually_set_order = kwargs.get('order', 0)
        if manually_set_order != 0:
            raise ValueError(
                "The order of the interpolation on a boolean image has to be "
                "0 (attempted to set {})".format(manually_set_order))
        kwargs['order'] = 0
        return Image.warp_to(self, template_mask, transform,
                             warp_landmarks=warp_landmarks,
                             interpolator=interpolator, **kwargs)

    def _build_warped_image(self, template_mask, sampled_pixel_values,
                            **kwargs):
        r"""
        Builds the warped image from the template mask and
        sampled pixel values. Overridden for BooleanImage as we can't use
        the usual from_vector_inplace method.
        """
        warped_image = BooleanImage.blank(template_mask.shape)
        # As we are a mask image, we have to implement the update a little
        # more manually than other image classes.
        warped_image.pixels[warped_image.mask] = sampled_pixel_values
        return warped_image
