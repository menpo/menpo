import numpy as np
import PIL.Image as PILImage
from pybug.transform.affine import Translation
from pybug.visualize import ImageViewer
from pybug.base import Vectorizable
import itertools


class AbstractImage(Vectorizable):
    """
    An abstract representation of an image. All images can be
    vectorized/built from vector, viewed,
    all have an image_shape, all are n_dimensional
    """
    def __init__(self, image_data):
        self.pixels = np.array(image_data)

    @property
    def width(self):
        return self.pixels.shape[0]

    @property
    def height(self):
        return self.pixels.shape[1]

    @property
    def n_pixels(self):
        return self.pixels.size

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def n_dims(self):
        return len(self.shape)

    def as_vector(self):
        r"""
        Convert AbstractImage to a vectorized form.

        :return: Vectorized image
        :rtype: ndarray [n_pixels]
        """
        return self.pixels.flatten()

    @classmethod
    def blank(cls, shape, fill=0, dtype=None):
        r"""
        Returns a blank image

        :param shape: The shape of the image
        :type shape: tuple or list
        :param fill: The value to fill all pixels with
        :type fill: int
        :param dtype: The numpy datatype to use
        :type dtype: numpy.dtype
        :return: A new AbstactImage of the requested size.
        :rtype: :class:`Image <pybug.image.base.AbstractImage>`
        """
        if dtype is None:
            dtype = np.float
        pixels = np.ones(shape, dtype=dtype) * fill
        return cls(pixels)

    def view(self):
        r"""
        View the image using the default image viewer. Currently only
        supports the rendering of 2D images.
        """
        if self.n_dims == 2:
            return ImageViewer(self.pixels)
        else:
            raise Exception("n_dim Image rendering is not yet supported.")

    def copy(self):
        r"""
        Return a copy of this image by instantiating an image with the same
        pixels

        :return: A copy of this image
        :rtype: :class:`Image <pybug.image.base.Image>`
        """
        return type(self)(self.pixels)

    def from_vector(self, flattened):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels.

        :param flattened: A flattened vector of all pixels of an
            abstract image
        :type flattened: ndarray [N x 1]
        :rtype: :class:`Image <pybug.image.base.AbstractImage>`
        """
        return type(self)(flattened.reshape(self.shape))

    def crop(self, *args):
        r"""
        Crops the image using the given slice objects. Expects
        ``len(args) == self.n_dims``.
        Also returns the
        :class:`Translation <pybug.transform.affine.Translation>` that would
        translate the cropped portion back in to the original reference
        position. **Note:** Only 2D and 3D images are currently supported.

        :param args: The slices to take over each axis
        :type args: List of slice objects
        :raises: AssertionError, ValueError
        :return: (cropped_image, translation)

                cropped_image:
                    Cropped portion of the image, including cropped mask.

                translation:
                    Translation transform that repositions the cropped image
                    in the reference frame of the original.
        :rtype: (:class:`Image <pybug.image.base.Image>`,
            :class:`Translation <pybug.transform.affine.Translation>`)
        """
        assert(self.n_dims == len(args))
        if len(args) == 2:
            cropped_image = self.pixels[args[0], args[1], ...]
            translation = np.array([args[0].start, args[1].start])
        elif len(args) == 3:
            cropped_image = self.pixels[args[0], args[1], args[2], ...]
            translation = np.array([args[0].start, args[1].start,
                                    args[2].start])
        else:
            raise ValueError("Only 2D and 3D images are currently supported.")

        return (type(self)(cropped_image),
                Translation(translation))


class MaskImage(AbstractImage):
    """
    A mask image with only 0's and 1's.
    """

    def __init__(self, mask_data):
        super(MaskImage, self).__init__(mask_data)
        self.pixels = self.pixels.astype(np.bool)  # enforce boolean pixels

    @property
    def n_true(self):
        r"""
        The number of ``True`` values in the mask

        :return: Number of pixels with a true value
        :rtype: int
        """
        return np.sum(self.pixels)

    @property
    def n_false(self):
        r"""
        The number of ``False`` values in the mask

        :return: Number of pixels with a false value
        :rtype: int
        """
        return self.n_pixels - self.n_true

    @property
    def true_indices(self):
        r"""
        The indices of pixels that are true.

        :return: array of indices
        :rtype: ndarray
        """
        return np.vstack(np.nonzero(self.pixels)).T

    @property
    def false_indices(self):
        r"""
        The indices of pixels that are false.

        :return: array of indices
        :rtype: ndarray
        """
        return np.vstack(np.nonzero(~self.pixels)).T

    def true_bounding_extent(self, boundary=0):
        r"""
        Returns the maximum and minimum values along all dimensions that the
        mask includes.

        :keyword boundary: A number of pixels that should be added to the
            extent.

            **Note:** the bounding extent is snapped to not go beyond
            the edge of the image.
        :type boundary: int >= 0
        :return: The bounding extent
        :rtype: ndarray [n_dims, 2] where
            [k, :] = [min_bounding_dim_k, max_bounding_dim_k]
        """
        mpi = self.true_indices
        maxes = np.max(mpi, axis=0) + boundary
        mins = np.min(mpi, axis=0) - boundary
        # check we don't stray under any edges
        mins[mins < 0] = 0
        # check we don't stray over any edges
        over_image = self.shape - maxes < 0
        maxes[over_image] = np.array(self.shape)[over_image]
        return np.vstack((mins, maxes)).T

    def true_bounding_extent_slicer(self, boundary=0):
        r"""
        Returns a slice object that can be used to retrieve the bounding
        extent.

        :keyword boundary: Passed through to :meth:`mask_bounding_extent
            <pybug.image.base.Image.mask_bounding_extent>`. The number of
            pixels that should be added to the extent.
        :type boundary: int >= 0
        :returns: Bounding extend slice object
        :rtype: slice
        """
        extents = self.true_bounding_extent(boundary)
        return [slice(x[0], x[1]) for x in list(extents)]

    def mask_bounding_extent_meshgrids(self, boundary=0):
        r"""
        Returns a list of meshgrids, the ``i`` th item being the meshgrid over
        the bounding extent of the ``i`` th dimension.

        :keyword boundary: Passed through to :meth:`mask_bounding_extent
            <pybug.image.base.Image.mask_bounding_extent>`. The number of
            pixels that should be added to the extent.
        :type boundary: int >= 0
        :return: list of ndarrays (output of ``np.meshgrid``)
        :rtype: list
        """
        extents = self.true_bounding_extent(boundary)
        return np.meshgrid(*[np.arange(*list(x)) for x in list(extents)])


class Image(AbstractImage):
    r"""
    Represents an N-dimensional image of size ``[M x N x ...]``. Images can be
    masked in order to identify a region on interest. All images implicitly
    have a mask that is defined as the the entire image. Supports
    construction from a PILImage.

    ``np.uint8`` pixel data is converted to ``np.float64`` and scaled between
    ``0`` and ``1`` by dividing each pixel by ``255``.

    :param image_data: The pixel data for the image, where the last axis
        represents the number of channels.
    :type image_data: ndarray [M x N ... x n_channels] or PILImage
    :keyword mask: A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).
    :type mask: ndarray [M x N ...] of ``np.bool`
    """

    def __init__(self, image_data, mask=None):

        # Correct for intensity images having no channel
        if len(image_data.shape) == 2:
            image_data = image_data[..., np.newaxis]

        # ensure datatype is float [0,1]
        if image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float64) / 255
        elif image_data.dtype != np.float64:
            # convert to double
            image_data = image_data.astype(np.float64)

        # now we let super have our data
        super(Image, self).__init__(image_data)

        if mask is not None:
            if self.shape != mask.shape:
                raise Exception("The mask is not of the same shape as the "
                                "image")
            if isinstance(mask, MaskImage):
                # have a MaskImage object - pull out the mask itself
                mask = mask.pixels
            self.mask = MaskImage(mask)
        else:
            self.mask = MaskImage(np.ones(self.shape))

    @property
    def shape(self):
        """
        Returns the shape of the image (with n_channel values at each point)
        """
        return self.pixels.shape[:-1]

    @property
    def n_channels(self):
        return self.pixels.shape[-1]

    def as_vector(self, keep_channels=False):
        r"""
        Convert image to a vectorized form.

        :keyword keep_channels:
            ``True``: returns ``n_channels`` column vectors.

            ``False``: returns a 1D array of all channels concatenated.
        :return: Vectorized image
        :rtype: ndarray [n_pixels] or [n_pixels x n_channels] if
         ``keep_channels is True``
        """
        if keep_channels:
            return self.masked_pixels.reshape([-1, self.n_channels])
        else:
            return self.masked_pixels.flatten()

    @classmethod
    def blank(cls, shape, n_channels=1, fill=0, mask=None):
        r"""
        Returns a blank image

        :param shape: The shape of the image
        :type shape: tuple or list
        :param n_channels: The number of channels the image should have
        :type n_channels: int
        :param fill: The value to fill all pixels with
        :type fill: int
        :param mask: An optional mask that can be applied
        :type mask: ndarray [M x N ...]
        :return: A new Image of the requested size.
        :rtype: :class:`Image <pybug.image.base.Image>`
        """
        pixels = np.ones(shape + (n_channels,)) * fill
        return Image(pixels, mask=mask)

    def copy(self):
        r"""
        Return a copy of this image by instantiating an image with the same
        pixel and mask data

        :return: A copy of this image
        :rtype: :class:`Image <pybug.image.base.Image>`
        """
        return Image(self.pixels, mask=self.mask)

    @property
    def masked_pixels(self):
        r"""
        Get the pixels covered by the ``True`` values in the mask.

        :return: Pixels that have a ``True`` mask value
        :rtype: ndarray [n_active_pixels, n_channels]
        """
        return self.pixels[self.mask.pixels]

    def mask_bounding_pixels(self, boundary=0):
        r"""
        Returns the pixels inside the bounding extent of the mask.

        :keyword boundary: Passed through to :meth:`mask_bounding_extent_slicer
            <pybug.image.base.Image.mask_bounding_extent_slicer>`. The
            number of pixels that should be added to the extent.
        :type boundary: int >= 0
        :return: Pixels inside the bounding extent of the mask
        :rtype: ndarray [M x N ... x n_channels]
        """
        return self.pixels[self.mask.true_bounding_extent_slicer(boundary)]

    def from_vector(self, flattened, n_channels=-1):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels.

        :param flattened: A flattened vector of all pixels and channels of an
            image
        :type flattened: ndarray [N x 1]
        :keyword n_channels: If given, will assume that flattened is the same
            shape as this image, but with a possibly different number of
            channels
        :type n_channels: int
        :return: New image of same shape as this image and the number of
            specified channels.
        :rtype: :class:`Image <pybug.image.base.Image>`
        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels == -1 else n_channels
        # Creates zeros of size (M x N x n_channels)
        image_data = np.zeros(self.shape + (n_channels,))
        pixels_per_channel = flattened.reshape((-1, n_channels))
        mask_array = self.mask.pixels  # the 'pixels' of a MaskImage is the
        # numpy mask array itself
        image_data[mask_array] = pixels_per_channel
        return Image(image_data, mask=mask_array)

    # TODO: can we do this mathematically and consistently ourselves?
    def as_greyscale(self):
        r"""
        Returns a greyscale copy of the image. This uses PIL in order to
        achieve this and so is only guaranteed to work for 2D images. The
        output image is guaranteed to have 1 channel.

        :return: A greyscale copy of the image
        :rtype: :class:`Image <pybug.image.base.Image>`
        """
        if self.n_channels == 1:
            print "Warning - trying to convert to greyscale an image with " \
                  "only one channel - returning a copy"
            return Image(self.pixels, self.mask.pixels)
        if self.n_channels != 3 or self.n_dims != 2:
            raise Exception("Trying to perform RGB-> greyscale conversion on"
                            " a non-2D-RGB Image.")
        pil_image = self.as_PILImage()
        pil_bw_image = pil_image.convert('L')
        return Image(pil_bw_image, mask=self.mask.pixels)

    def as_PILImage(self):
        r"""
        Return a PIL copy of the image. Scales the image by ``255`` and
        converts to ``np.uint8``.

        :return: PIL copy of image as ``np.uint8``
        :rtype: PILImage
        """
        return PILImage.fromarray((self.pixels * 255).astype(np.uint8))

    def crop(self, *args):
        r"""
        Crops the image using the given slice objects. Expects
        ``len(args) == self.n_dims``. Maintains the cropped portion of the
        mask. Also returns the
        :class:`Translation <pybug.transform.affine.Translation>` that would
        translate the cropped portion back in to the original reference
        position. **Note:** Only 2D and 3D images are currently supported.

        :param args: The slices to take over each axis
        :type args: List of slice objects
        :raises: AssertionError, ValueError
        :return: (cropped_image, translation)

                cropped_image:
                    Cropped portion of the image, including cropped mask.

                translation:
                    Translation transform that repositions the cropped image
                    in the reference frame of the original.
        :rtype: (:class:`Image <pybug.image.base.Image>`,
            :class:`Translation <pybug.transform.affine.Translation>`)
        """
        assert(self.n_dims == len(args))
        if len(args) == 2:
            cropped_image = self.pixels[args[0], args[1], ...]
            cropped_mask = self.mask[args[0], args[1], ...]
            translation = np.array([args[0].start, args[1].start])
        elif len(args) == 3:
            cropped_image = self.pixels[args[0], args[1], args[2], ...]
            cropped_mask = self.mask[args[0], args[1], args[2], ...]
            translation = np.array([args[0].start, args[1].start,
                                    args[2].start])
        else:
            raise ValueError("Only 2D and 3D images are currently supported.")

        return (Image(cropped_image, mask=cropped_mask),
                Translation(translation))

    def gradient(self, inc_unmasked_pixels=False):
        r"""
        Returns an Image which is the gradient of this one. In the case of
        multiple channels, it returns the gradient over each axis over each
        channel as a flat list.

        :return: The gradient over each axis over each channel
        :rtype: list
        """
        if inc_unmasked_pixels:
            gradients = [np.gradient(g) for g in
                         np.rollaxis(self.pixels, -1)]
            # Flatten the lists
            gradients = list(itertools.chain.from_iterable(gradients))
            # Add an extra axis for broadcasting
            gradients = [g[..., None] for g in gradients]
            # Concatenate gradient list into an array (the new_image)
            new_image = np.concatenate(gradients, axis=-1)
        else:
            masked_square_image = self.mask_bounding_pixels(boundary=3)
            bounding_mask = self.mask_bounding_extent_slicer(3)
            gradients = [np.gradient(g) for g in
                         np.rollaxis(masked_square_image, -1)]
            # Flatten the lists
            gradients = list(itertools.chain.from_iterable(gradients))
            # Add an extra axis for broadcasting
            gradients = [g[..., None] for g in gradients]
            # Concatenate gradient list into a vector
            gradient_array = np.concatenate(gradients, axis=-1)
            # make a new blank image
            new_image = np.empty((self.shape + (len(gradients),)))
            # populate the new image with the gradient
            new_image[bounding_mask] = gradient_array
        return Image(new_image, mask=self.mask)
