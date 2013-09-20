# noinspection PyPackageRequirements
from copy import deepcopy
import numpy as np
from pybug.image.base import AbstractNDImage


class BooleanNDImage(AbstractNDImage):
    r"""
    A mask image made from binary pixels. The region of the image that is
    left exposed by the mask is referred to as the 'masked region'. The
    set of 'masked' pixels is those pixels corresponding to a True value in
    the mask.

    Parameters
    -----------
    mask_data : (M, N, ...,) ndarray
        The binary mask data. Note that there is no channel axis - a 2D Mask
         Image is built from just a 2D numpy array of mask_data.
        Automatically coerced in to boolean values.
    """

    def __init__(self, mask_data):
        # Enforce boolean pixels, and add a channel dim
        mask_data = np.asarray(mask_data[..., None], dtype=np.bool)
        super(BooleanNDImage, self).__init__(mask_data)

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
    def blank(cls, shape, fill=True):
        r"""
        Returns a blank :class:`BooleanNDImage` of the requested shape

        Parameters
        ----------
        shape : tuple or list
            The shape of the mask image image

        fill : True or False, optional
            The mask value to be set everywhere

        Default: True (masked region is whole image - whole image is exposed)

        Returns
        -------
        blank_image : :class:`BooleanNDImage`
            A blank mask of the requested size
        """
        if fill:
            mask = np.ones(shape, dtype=np.bool)
        else:
            mask = np.zeros(shape, dtype=np.bool)
        return cls(mask)

    @property
    def mask(self):
        r"""
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

        :type: (n_dim, n_true_pixels) ndarray
        """
        # Ignore the channel axis
        return np.vstack(np.nonzero(self.pixels[..., 0])).T

    @property
    def false_indices(self):
        r"""
        The indices of pixels that are false.

        :type: (n_dim, n_false_pixels) ndarray
        """
        # Ignore the channel axis
        return np.vstack(np.nonzero(~self.pixels[..., 0])).T

    @property
    def all_indices(self):
        r"""
        Indices into all pixels of the mask, as consistent with
        true_indices & false_indices

        :type: (n_dim, n_pixels) ndarray
        """
        return np.indices(self.shape).reshape([self.n_dims, -1]).T

    def __str__(self):
        return ('{} {}D mask, {:.1%} '
                'of which is True '.format(self._str_shape, self.n_dims,
                                           self.proportion_true))

    def from_vector(self, flattened):
        r"""
        Takes a flattened vector and returns a new BooleanImage formed by
        reshaping the vector to the correct dimensions. Note that this is
        rebuilding a boolean image **itself** from boolean values. The mask
        is in no way interpreted in performing the operation, in contrast to
        MaskedNDImage, where only the masked region is used in
        {from, as}_vector.

        Parameters
        ----------
        flattened : (``n_pixels``,)
            A flattened vector of all the pixels of a BooleanImage.

        Returns
        -------
        image : :class:`BooleanNDImage`
            New BooleanImage of same shape as this image
        """
        return BooleanNDImage(flattened.reshape(self.shape))

    def update_from_vector(self, flattened):
        r"""
        Takes a flattened vector and update this Boolean image by
        reshaping the vector to the correct dimensions. Note that this is
        rebuilding a boolean image **itself** from boolean values. The mask
        is in no way interpreted in performing the operation, in contrast to
        MaskedNDImage, where only the masked region is used in
        {from, as}_vector.

        Parameters
        ----------
        flattened : (``n_pixels``,)
            A flattened vector of all the pixels of a BooleanImage.

        Returns
        -------
        image : :class:`BooleanNDImage`
            This image post update
        """
        self.pixels = flattened.reshape(self.pixels.shape)
        return self

    def invert(self):
        r"""
        Inverts the current mask in place.
        """
        self.pixels = ~self.pixels

    def inverted_copy(self):
        r"""
        Returns a copy of this Boolean image, which is inverted.
        """
        inverse = deepcopy(self)
        inverse.invert()
        return inverse

    # noinspection PyTypeChecker
    def bounds_true(self, boundary=0):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the False mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the False pixel values.

        Parameters
        ----------
        boundary : int >= 0, optional
            A number of pixels that should be added to the extent.

            Default: 0

            .. note::
                The bounding extent is snapped to not go beyond
                the edge of the image.

        Returns
        -------
        bounding_extent : (``n_dims``, 2) ndarray
            The bounding extent where
            ``[k, :] = [min_bounding_dim_k, max_bounding_dim_k]``
        """
        mpi = self.true_indices
        maxes = np.max(mpi, axis=0) + boundary
        mins = np.min(mpi, axis=0) - boundary
        # check we don't stray under any edges
        mins[mins < 0] = 0
        # check we don't stray over any edges
        over_image = self.shape - maxes < 0
        maxes[over_image] = np.array(self.shape)[over_image]
        return mins, maxes

    # noinspection PyTypeChecker
    def bounds_false(self, boundary=0):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the true mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the False pixel values.

        Parameters
        ----------
        boundary : int >= 0, optional
            A number of pixels that should be added to the extent.

            Default: 0

            .. note::
                The bounding extent is snapped to not go beyond
                the edge of the image.

        Returns
        -------
        bounding_extent : (``n_dims``, 2) ndarray
            The bounding extent where
            ``[k, :] = [min_bounding_dim_k, max_bounding_dim_k]``
        """
        return self.inverted_copy().bounds_true(boundary=boundary)
