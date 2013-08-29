import numpy as np
import PIL.Image as PILImage
from pybug.exceptions import DimensionalityError
from pybug.transform.affine import Translation
from pybug.landmark import Landmarkable
from pybug.base import Vectorizable
from skimage.morphology import diamond, binary_erosion
import itertools
from pybug.visualize.base import Viewable, ImageViewer


class AbstractImage(Vectorizable, Landmarkable, Viewable):
    r"""
    An abstract representation of an image. All images can be
    vectorized/built from vector, viewed, all have a ``shape``,
    all are ``n_dimensional``.

    Images are also :class:`pybug.landmark.Landmarkable`.

    Parameters
    -----------
    image_data: (M, N, ...) ndarray
        Array representing the image pixels.
    """
    def __init__(self, image_data):
        Landmarkable.__init__(self)
        self.pixels = np.array(image_data)

    @property
    def width(self):
        r"""
        The width of the image.

        This is the width according to image semantics, and is thus the size
        of the **second** dimension.

        :type: int
        """
        return self.pixels.shape[1]

    @property
    def height(self):
        r"""
        The height of the image.

        This is the height according to image semantics, and is thus the size
        of the **first** dimension.

        :type: int
        """
        return self.pixels.shape[0]

    @property
    def n_pixels(self):
        r"""
        Total number of pixels in the image (``prod(shape)``)

        :type: int
        """
        return self.pixels.size

    @property
    def shape(self):
        r"""
        The shape of the pixels array.

        This is a tuple of the form ``(M, N, ...)``

        :type: tuple
        """
        return self.pixels.shape

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the image.

        :type: int
        """
        return len(self.shape)

    def as_vector(self):
        r"""
        Convert :class:`AbstractImage` to a vectorized form.

        Returns
        --------
        vectorized_image : (``n_pixels``,) ndarray
            A 1D array representing a vectorized form of the image
        """
        return self.pixels.flatten()

    @classmethod
    def blank(cls, shape, fill=0, dtype=None):
        r"""
        Returns a blank image

        Parameters
        ----------
        shape : tuple or list
            The shape of the image
        fill : int, optional
            The value to fill all pixels with

            Default: 0
        dtype : numpy.dtype, optional
            The numpy datatype to use

            Default: ``np.float``

        Returns
        -------
        blank_image : :class:`AbstractImage`
            A blank image of the requested size
        """
        if dtype is None:
            dtype = np.float
        pixels = np.ones(shape, dtype=dtype) * fill
        return cls(pixels)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
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
        return ImageViewer(figure_id, new_figure,
                           self.n_dims, self.pixels).render(**kwargs)

    def copy(self):
        r"""
        Return a copy of this image by instantiating an image with the same
        pixels

        Returns
        -------
        image_copy :
            A copy of the image
        """
        return type(self)(self.pixels)

    def from_vector(self, flattened):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels.

        Parameters
        ----------
        flattened : (``n_pixels``,) ndarray
            A flattened vector of all pixels of an abstract image

        Returns
        -------
        image : :class:`AbstractImage`
            A new image built from the vector
        """
        return type(self)(flattened.reshape(self.shape))

    def crop(self, *slice_args):
        r"""
        Crops the image using the given slice objects. Expects
        ``len(args) == self.n_dims``.

        Also returns the
        :class:`Translation <pybug.transform.affine.Translation>` that would
        translate the cropped portion back in to the original reference
        position.

        .. note::
            Only 2D and 3D images are currently supported.

        Parameters
        -----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`Image`
            Cropped portion of the image, including cropped mask.
        translation : :class:`pybug.transform.affine.Translation`
            Translation transform that repositions the cropped image in the
            reference frame of the original.
        """
        assert(self.n_dims == len(slice_args))
        cropped_image = self.pixels[slice_args]
        translation = np.array([x.start for x in slice_args])
        return type(self)(cropped_image), Translation(translation)


class MaskImage(AbstractImage):
    r"""
    A mask image made from binary pixels.

    Parameters
    -----------
    mask_data : (M, N) ndarray
        The pixel values. Are automatically coerced in to boolean values.
    """

    def __init__(self, mask_data):
        super(MaskImage, self).__init__(mask_data)
        self.pixels = self.pixels.astype(np.bool)  # enforce boolean pixels

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
    def true_indices(self):
        r"""
        The indices of pixels that are true.

        :type: ndarray
        """
        return np.vstack(np.nonzero(self.pixels)).T

    @property
    def false_indices(self):
        r"""
        The indices of pixels that are false.

        :type: ndarray
        """
        return np.vstack(np.nonzero(~self.pixels)).T

    def true_bounding_extent(self, boundary=0):
        r"""
        Returns the maximum and minimum values along all dimensions that the
        mask includes.

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
        return np.vstack((mins, maxes)).T

    def true_bounding_extent_slicer(self, boundary=0):
        r"""
        Returns a slice object that can be used to retrieve the bounding
        extent.

        Parameters
        ----------
        boundary : int >= 0, optional
            Passed through to :meth:`true_bounding_extent`. The number of
            pixels that should be added to the extent.

            Default: 0

        Returns
        -------
        bounding_extent : slice
            Bounding extent slice object
        """
        extents = self.true_bounding_extent(boundary)
        return [slice(x[0], x[1]) for x in list(extents)]

    def mask_bounding_extent_meshgrids(self, boundary=0):
        r"""
        Returns a list of meshgrids, the ``i`` th item being the meshgrid over
        the bounding extent of the ``i`` th dimension.

        Parameters
        ----------
        boundary : int >= 0, optional
            Passed through to :meth:`true_bounding_extent`. The number of
            pixels that should be added to the extent.

            Default: 0

        Returns
        -------
        bounding_extent : list of ndarrays
            output of ``np.meshgrid``
        """
        extents = self.true_bounding_extent(boundary)
        return np.meshgrid(*[np.arange(*list(x)) for x in list(extents)])


class Image(AbstractImage):
    r"""
    Represents an 2-dimensional image with a number of channels, of size
    ``(M, N, C)``. Images can be masked in order to identify a region of
    interest. All images implicitly have a mask that is defined as the the
    entire image. Supports construction from a ``PILImage``.

    .. note::
        ``np.uint8`` pixel data is converted to ``np.float64``
        and scaled between ``0`` and ``1`` by dividing each pixel by ``255``.

    Parameters
    ----------
    image_data :  ndarray or ``PILImage``
        The pixel data for the image, where the last axis represents the
        number of channels.
    mask : (M, N, C) ``np.bool`` ndarray or :class:`MaskImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`MaskImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)
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
                raise ValueError("The mask is not of the same shape as the "
                                "image")
            if isinstance(mask, MaskImage):
                # have a MaskImage object - pull out the mask itself
                mask = mask.pixels
            self.mask = MaskImage(mask)
        else:
            self.mask = MaskImage(np.ones(self.shape))

    @property
    def shape(self):
        r"""
        Returns the shape of the image
        (with ``n_channel`` values at each point).

        :type: tuple
        """
        return self.pixels.shape[:-1]

    @property
    def n_channels(self):
        """
        The total number of channels.

        :type: int
        """
        return self.pixels.shape[-1]

    def as_vector(self, keep_channels=False):
        r"""
        Convert image to a vectorized form.

        Parameters
        ----------
        keep_channels : bool, optional
            ========== =================
            Value      Return shape
            ========== =================
            ``True``   (``n_pixels``,``n_channels``)
            ``False``  (``n_pixels``,)
            ========== =================

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

    @classmethod
    def blank(cls, shape, n_channels=1, fill=0, mask=None):
        r"""
        Returns a blank image

        Parameters
        ----------
        shape : tuple or list
            The shape of the image
        fill : int, optional
            The value to fill all pixels with

            Default: 0
        mask: (M, N) boolean ndarray or :class:`MaskImage`
            An optional mask that can be applied.

        Returns
        -------
        blank_image : :class:`Image`
            A new Image of the requested size.
        """
        pixels = np.ones(shape + (n_channels,)) * fill
        return Image(pixels, mask=mask)

    def copy(self):
        r"""
        Return a copy of this image by instantiating an image with the same
        pixel and mask data

        Returns
        -------
        copy_image : :class:`Image`
            A copy of this image
        """
        return Image(self.pixels, mask=self.mask)

    @property
    def masked_pixels(self):
        r"""
        Get the pixels covered by the ``True`` values in the mask.

        :type: (``mask.n_true``, ``n_channels``) ndarray
        """
        return self.pixels[self.mask.pixels]

    def mask_bounding_pixels(self, boundary=0):
        r"""
        Returns the pixels inside the bounding extent of the mask.

        Parameters
        ----------
        boundary : int >= 0, optional
            Passed through to :meth:`mask_bounding_extent_slicer`. The
            number of pixels that should be added to the extent.

            Default: 0

        Returns
        -------
        bounding_pixels : (M, N, C) ndarray
            Pixels inside the bounding extent of the mask
        """
        return self.pixels[self.mask.true_bounding_extent_slicer(boundary)]

    def from_vector(self, flattened, n_channels=-1):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels.

        The ``n_channels`` argument is useful for when we want to add an extra
        channel to an image but maintain the shape. For example, when
        calculating the gradient.

        Parameters
        ----------
        flattened : (``n_pixels``,)
            A flattened vector of all pixels and channels of an image.
        n_channels : int, optional
            If given, will assume that flattened is the same
            shape as this image, but with a possibly different number of
            channels

            Default: -1 (use the existing image channels)

        Returns
        -------
        image : :class:`Image`
            New image of same shape as this image and the number of
            specified channels.
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
        achieve this and so is only guaranteed to work for 3-channel 2D images.
        The output image is guaranteed to have 1 channel. If a single channel
        image is passed in, then this method returns a copy of the image.

        Returns
        -------
        greyscale : :class:`Image`
            A greyscale copy of the image

        Raises
        ------
        DimensionalityError
            Only 2-dimensional greyscale (1-channel) and RGB (3-channel)
            images are supported.
        """
        if self.n_channels == 1:
            return Image(self.pixels, self.mask.pixels)
        if self.n_channels != 3 or self.n_dims != 2:
            raise DimensionalityError("Trying to perform RGB-> greyscale "
                                      "conversion on a non-2D-RGB Image.")

        pil_image = self.as_PILImage()
        pil_bw_image = pil_image.convert('L')
        return Image(pil_bw_image, mask=self.mask.pixels)

    def as_PILImage(self):
        r"""
        Return a PIL copy of the image. Scales the image by ``255`` and
        converts to ``np.uint8``.

        Returns
        -------
        pil_image : ``PILImage``
            PIL copy of image as ``np.uint8``
        """
        return PILImage.fromarray((self.pixels * 255).astype(np.uint8))

    def crop(self, *slice_args):
        r"""
        Crops the image using the given slice objects. Expects
        ``len(args) == self.n_dims``. Maintains the cropped portion of the
        mask. Also returns the
        :class:`pybug.transform.affine.Translation` that would
        translate the cropped portion back in to the original reference
        position.

        .. note::
            Only 2D and 3D images are currently supported.

        Parameters
        ----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`Image`:
            Cropped portion of the image, including cropped mask.
        translation : :class:`pybug.transform.affine.Translation`
            Translation transform that repositions the cropped image
            in the reference frame of the original.
        """
        # crop our image
        cropped_image, translation = super(Image, self).crop(*slice_args)
        # crop our mask
        cropped_mask, mask_translation = self.mask.crop(*slice_args)
        cropped_image.mask = cropped_mask  # assign the mask
        # sanity check
        assert (translation == mask_translation)
        return cropped_image, translation
        # assert(self.n_dims == len(slice_args))
        # cropped_image = self.pixels[slice_args]
        # cropped_mask = self.mask.pixels[slice_args]
        # translation = np.array([x.start for x in slice_args])
        # return (Image(cropped_image, mask=cropped_mask),
        #         Translation(translation))

    # TODO this kwarg could be False for higher perf True for debug
    # TODO something is fishy about this method, kwarg seems to be making diff
    # TODO make a unit test for gradient of masked images (inc_masked_pixels)
    def gradient(self, inc_unmasked_pixels=False):
        r"""
        Returns an Image which is the gradient of this one. In the case of
        multiple channels, it returns the gradient over each axis over each
        channel as a flat list.

        Parameters
        ----------
        inc_unmasked_pixels : bool, optional
            If ``True``, the gradient is taken over the entire image and not
            just the masked area.

        Returns
        -------
        gradient : list of (M, N) ndarrays
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length ``2``.
            The length of a 2D, 3-channel image, will have length ``6``.
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
            gradients = [np.gradient(g) for g in np.rollaxis(self.pixels, -1)]
            gradients = list(itertools.chain.from_iterable(gradients))
            gradients = [g[..., None] for g in gradients]
            gradient_array = np.concatenate(gradients, axis=-1)

            # Erode the edge of the mask so that the gradients are not
            # affected by the outlying pixels
            diamond_structure = diamond(1)
            mask = binary_erosion(self.mask.pixels, diamond_structure)

            new_image = np.zeros((self.shape + (len(gradients),)))
            new_image[mask] = gradient_array[mask]

        return Image(new_image, mask=self.mask)
