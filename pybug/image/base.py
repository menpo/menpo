import numpy as np
from copy import deepcopy
import PIL.Image as PILImage
import scipy.linalg
from pybug.transform.affine import Translation
from pybug.landmark import Landmarkable
from pybug.base import Vectorizable
from scipy.ndimage.morphology import binary_erosion
import itertools
from pybug.visualize.base import Viewable, ImageViewer, DepthImageHeightViewer


class AbstractNDImage(Vectorizable, Landmarkable, Viewable):
    r"""
    An abstract representation of an image. All images can be
    vectorized/built from vector, viewed, all have a ``shape``,
    all are ``n_dimensional``, and all have ``n_channels``.

    Images are also :class:`pybug.landmark.Landmarkable`.

    Parameters
    -----------
    image_data: (M, N, ..., C) ndarray
        Array representing the image pixels, with the last axis being
        channels.
    """
    def __init__(self, image_data):
        Landmarkable.__init__(self)
        # asarray will pass through ndarrays unchanged
        image_data = np.asarray(image_data)
        if image_data.ndim < 3:
            raise ValueError("Abstract Images have to build from at least 3D"
                             " image data arrays (2D + n_channels) - a {} "
                             "dim array was provided".format(image_data.ndim))
        self.pixels = image_data

    @classmethod
    def _init_with_channel(cls, image_data_with_channel):
        r"""
        Constructor that always requires the image has a
        channel on the last axis. Only used by from_vector. By default,
        just calls the constructor. Subclasses with constructors that don't
        require channel axes need to overwrite this.
        """
        return cls(image_data_with_channel)

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the image. The minimum possible n_dims is
        2.

        :type: int
        """
        return len(self.shape)

    @property
    def n_pixels(self):
        r"""
        Total number of pixels in the image (``prod(shape)``)

        :type: int
        """
        return self.pixels[..., 0].size

    @property
    def n_elements(self):
        r"""
        Total number of data points in the image (``prod(shape) x
        n_channels``)

        :type: int
        """
        return self.pixels.size

    @property
    def n_channels(self):
        """
        The number of channels on each pixel in the image.

        :type: int
        """
        return self.pixels.shape[-1]

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
    def depth(self):
        r"""
        The depth of the image.

        This is the depth according to image semantics, and is thus the size
        of the **third** dimension. If the n_dim of the image is 2, this is 0.

        :type: int
        """
        if self.n_dims == 2:
            return 0
        else:
            return self.pixels.shape[0]

    @property
    def shape(self):
        r"""
        The shape of the image
        (with ``n_channel`` values at each point).

        :type: tuple
        """
        return self.pixels.shape[:-1]

    @property
    def centre(self):
        r"""
        The geometric centre of the Image - the subpixel that is in the
        middle.

        Useful for aligning shapes and images.

        :type: (D,) ndarray
        """
        return np.array(self.shape, dtype=np.double) / 2

    @property
    def _str_shape(self):
        if self.n_dims > 3:
            return reduce(lambda x, y: str(x) + ' x ' + str(y),
                          self.shape) + ' (in memory)'
        elif self.n_dims == 3:
            return (str(self.width) + 'W x ' + str(self.height) + 'H x ' +
                    str(self.depth) + 'D')
        elif self.n_dims == 2:
            return str(self.width) + 'W x ' + str(self.height) + 'H'

    def as_vector(self, keep_channels=False):
        r"""
        Convert the Image to a vectorized form.

        Parameters
        ----------
        keep_channels : bool, optional

            ========== =================
            Value      Return shape
            ========== =================
            ``True``   (``n_pixels``,``n_channels``)
            ``False``  (``n_pixels`` x ``n_channels``,)
            ========== =================

            Default: ``False``

        Returns
        -------
        vectorized_image : (shape given by ``keep_channels``) ndarray
            Vectorized image
        """
        if keep_channels:
            return self.pixels.reshape([-1, self.n_channels])
        else:
            return self.pixels.flatten()

    def _view(self, figure_id=None, new_figure=False, channel=None, **kwargs):
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
        pixels_to_view = self.pixels
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channel=channel).render(**kwargs)

    def crop_self(self, *slice_args):
        r"""
        Crops this image using the given slice objects. Expects
        ``len(args) == self.n_dims``. Landmarks are correctly adjusted so they
        maintain their position relative to the newly cropped image.

        Parameters
        -----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`self`
            This image, but cropped.
        """
        assert(self.n_dims == len(slice_args))
        self.pixels = self.pixels[slice_args]
        lm_translation = Translation(-np.array([x.start for x in slice_args]))
        # update all our landmarks
        for manager in self.landmarks.values():
            for label, landmarks in manager:
                lm_translation.apply(landmarks)
        return self

    def crop(self, *slice_args):
        r"""
        Returns a cropped version of this image using the given slice
        objects. Expects
        ``len(args) == self.n_dims``. Landmarks are correctly adjusted so they
        maintain their position relative to the newly cropped image.

        Parameters
        -----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`Image`
            A new instance of self, cropped.
        """
        cropped_image = deepcopy(self)
        return cropped_image.crop_self(*slice_args)


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


class MaskedNDImage(AbstractNDImage):
    r"""
    Represents an n-dimensional image with a number of channels, of size
    ``(M, N, ..., C)`` which has a mask. Images can be masked in order to
    identify a region of interest. All images implicitly have a mask that is
    defined as the the entire image. The mask is an instance of
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
    -------
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
            The shape of the image
        fill : int, optional
            The value to fill all pixels with

            Default: 0
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied.

        Returns
        -------
        blank_image : :class:`Image`
            A new Image of the requested size.
        """
        if fill == 0:
            pixels = np.zeros(shape + (n_channels,), dtype=dtype)
        else:
            pixels = np.ones(shape + (n_channels,), dtype=dtype) * fill
        return cls(pixels, mask=mask)

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

            ========== =================
            Value      Return shape
            ========== =================
            ``True``   (``mask.n_true``,``n_channels``)
            ``False``  (``mask.n_true`` x ``n_channels``,)
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

    def from_vector(self, flattened, n_channels=-1):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

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
        image : :class:`MaskedNDImage`
            New image of same shape as this image and the number of
            specified channels.
        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels == -1 else n_channels
        # Creates zeros of size (M x N x ... x n_channels)
        image_data = np.zeros(self.shape + (n_channels,))
        pixels_per_channel = flattened.reshape((-1, n_channels))
        image_data[self.mask.mask] = pixels_per_channel
        # call the constructor accounting for the fact that some image
        # classes expect a channel axis and some don't.
        return self.__class__._init_with_channel(image_data, mask=self.mask)

    def update_from_vector(self, flattened):
        r"""
        Takes a flattened vector and updates this image by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        Parameters
        ----------
        flattened : (``n_pixels``,)
            A flattened vector of all pixels and channels of an image.

        Returns
        -------
        image : :class:`MaskedNDImage`
            This image after being updated
        """
        self.masked_pixels = flattened.reshape((-1, self.n_channels))
        # call the constructor accounting for the fact that some image
        # classes expect a channel axis and some don't.
        return self

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

    def crop_self(self, *slice_args):
        r"""
        Crops this image using the given slice objects. Expects
        ``len(args) == self.n_dims``. Landmarks are correctly adjusted so they
        maintain their position relative to the newly cropped image.

        Parameters
        -----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`self`
            This image, but cropped.
        """
        # crop our image
        super(MaskedNDImage, self).crop_self(*slice_args)
        # crop our mask
        self.mask.crop_self(*slice_args)
        return self

    def gradient(self, nullify_values_at_mask_boundaries=False):
        r"""
        Returns a MaskedNDImage which is the gradient of this one. In the case
        of multiple channels, it returns the gradient over each axis over
        each channel as a flat list.

        Parameters
        ----------
        nullify_values_at_mask_boundaries : bool, optional
            If ``True``, the gradient is taken over the entire image and not
            just the masked area.

        Default: False

        Returns
        -------
        gradient : :class:``MaskedNDimage``
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length ``2``.
            The length of a 2D, 3-channel image, will have length ``6``.
        """
        grad_per_dim_per_channel = [np.gradient(g) for g in
                                   np.rollaxis(self.pixels, -1)]
        # Flatten out the seperate dims
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
            eroded_mask = binary_erosion(self.mask.pixels, iterations=1)

            # replace the eroded mask with the diff between the two
            # masks
            np.logical_and(~eroded_mask, self.mask.pixels, eroded_mask)
            # nullify all the boundary values in the grad image
            grad_image.pixels[eroded_mask] = 0.0

        return grad_image


class Abstract2DImage(MaskedNDImage):
    r"""
    Represents a 2-dimensional image with k number of channels, of size
    ``(M, N, C)``. ``np.uint8`` pixel data is converted to ``np.float64``
    and scaled between ``0`` and ``1`` by dividing each pixel by ``255``.
    All Image2D instances have values for channels between 0-1,
    and have a dtype of np.float.

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
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        image = np.asarray(image_data)
        if image.ndim != 3:
            raise ValueError("Trying to build a 2DImage with "
                             "{} dims".format(image.ndim - 1))
        # ensure pixels are np.float [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float64) / 255
        elif image.dtype != np.float64:
            # convert to double
            image = image.astype(np.float64)
        super(Abstract2DImage, self).__init__(image, mask=mask)

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

    def constrain_mask_to_landmarks(self, group=None, label=None):
        r"""
        Restricts this image's mask to be equal to the the convex hull
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
        """
        from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
        from pybug.transform.piecewiseaffine import TriangleContainmentError

        if len(self.landmarks) == 0:
            raise ValueError("There are no attached landmarks to "
                             "infer a mask from")
        if group is None:
            if len(self.landmarks) > 1:
                raise ValueError("no group was provided and there are "
                                 "multiple groups. Specify a group, "
                                 "e.g. {}".format(self.landmarks.keys[0]))
            else:
                group = self.landmarks.keys()[0]

        if label is None:
            pc = self.landmarks[group].all_landmarks
        else:
            pc = self.landmarks[group].with_label(label).all_landmarks

        # delaunay as no trilist provided
        pwa = PiecewiseAffineTransform(pc.points, pc.points)
        try:
            pwa.apply(self.mask.all_indices)
        except TriangleContainmentError, e:
            self.mask.update_from_vector(~e.points_outside_source_domain)


class RGBImage(Abstract2DImage):
    r"""
    Represents a 2-dimensional image with 3 channels for RBG respectively,
    of size ``(M, N, 3)``. ``np.uint8`` pixel data is converted to ``np
    .float64``, and scaled between ``0`` and ``1`` by dividing each pixel by
    ``255``.

    Parameters
    ----------
    image_data :  ndarray or ``PILImage``
        The pixel data for the image, where the last axis represents the
        three channels.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        super(RGBImage, self).__init__(image_data, mask=mask)
        if self.n_channels != 3:
            raise ValueError("Trying to build an RGBImage with {} channels -"
                             " you must provide a numpy array of size (M, N,"
                             " 3)".format(self.n_channels))

    # noinspection PyMethodOverriding
    @classmethod
    def blank(cls, shape, fill=0, mask=None):
        r"""
        Returns a blank image

        Parameters
        ----------
        shape : tuple or list
            The shape of the image
        fill : int, optional
            The value to fill all pixels with

            Default: 0
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied.

        Returns
        -------
        blank_image : :class:`RGBImage`
            A new Image of the requested size.
        """
        if fill == 0:
            pixels = np.zeros(shape + (3,), dtype=np.float64)
        elif fill < 0 or fill > 1:
            raise ValueError("RGBImage can only have values in the range "
                             "[0-1] (tried to fill with {})".format(fill))
        else:
            pixels = np.ones(shape + (3,), dtype=np.float64) * fill
        return cls(pixels, mask=mask)

    def __str__(self):
        return ('{} RGBImage. '
                'Attached mask {:.1%} true'.format(
                self._str_shape, self.n_dims, self.n_channels,
                self.mask.proportion_true))

    def as_greyscale(self, mode='average', channel=None):
        r"""
        Returns a greyscale version of the RGB image.

        Parameters
        ----------
        mode : {'average', 'luminosity', 'channel'}
            'luminosity' - Calculates the luminance using the CCIR 601 formula
                ``Y' = 0.299 R' + 0.587 G' + 0.114 B'``
            'average' - intensity is an equal average of all three channels
            'channel' - a specific channel is used

            Default 'luminosity'

        channel: int, optional
            The channel to be taken. Only used if mode is 'channel'.

            Default: None

        Returns
        -------
        greyscale_image: :class:`IntensityImage`
            A copy of this image in greyscale.
        """
        if mode == 'luminosity':
            # Invert the transformation matrix to get more precise values
            T = scipy.linalg.inv(np.array([[1.0, 0.956, 0.621],
                                           [1.0, -0.272, -0.647],
                                           [1.0, -1.106, 1.703]]))
            coef = T[0, :]
            pixels = np.dot(self.pixels, coef.T)
        elif mode == 'average':
            pixels = np.mean(self.pixels, axis=-1)
        elif mode == 'channel':
            if channel is None:
                raise ValueError("for the 'channel' mode you have to provide"
                                 " a channel index")
            elif channel < 0  or channel > 2:
                raise ValueError("channel can only be 0, 1, or 2 "
                                 "in RGB images.")
            pixels = self.pixels[..., channel]
        mask = deepcopy(self.mask)
        # TODO is this is a safe copy of the landmark dict?
        landmark_dict = deepcopy(self.landmarks)
        greyscale = IntensityImage(pixels, mask=mask)
        greyscale.landmarks = landmark_dict
        return greyscale


class IntensityImage(Abstract2DImage):
    r"""
    Represents a 2-dimensional image with 1 channel for intensity,
    of size ``(M, N, 1)``. ``np.uint8`` pixel data is converted to ``np
    .float64``, and scaled between ``0`` and ``1`` by dividing each pixel by
    ``255``.

    Parameters
    ----------
    image_data : (M, N)  ndarray or ``PILImage``
        The pixel data for the image. Note that a channel axis should not be
         provided.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    def __init__(self, image_data, mask=None):
        # add the channel axis on
        super(IntensityImage, self).__init__(image_data[..., None], mask=mask)
        if self.n_channels != 1:
            raise ValueError("Trying to build a IntensityImage with {} "
                             "channels (you shouldn't provide a channel "
                             "axis to the IntensityImage constructor)"
                             .format(self.n_channels))

    @classmethod
    def _init_with_channel(cls, image_data_with_channel, mask):
        return cls(image_data_with_channel[..., 0], mask)

    # noinspection PyMethodOverriding
    @classmethod
    def blank(cls, shape, fill=0, mask=None):
        r"""
        Returns a blank IntensityImage of the requested shape.

        Parameters
        ----------
        shape : tuple or list
            The shape of the image
        fill : int, optional
            The value to fill all pixels with

            Default: 0
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied.

        Returns
        -------
        blank_image : :class:`IntensityImage`
            A new Image of the requested size.
        """
        if fill == 0:
            pixels = np.zeros(shape, dtype=np.float64)
        elif fill < 0 or fill > 1:
            raise ValueError("IntensityImage can only have values in the "
                             "range [0-1] "
                             "(tried to fill with {})".format(fill))
        else:
            pixels = np.ones(shape, dtype=np.float64) * fill
        return cls(pixels, mask=mask)

    def __str__(self):
        return ('{} IntensityImage. '
                'Attached mask {:.1%} true'.format(
                self._str_shape, self.n_dims, self.n_channels,
                self.mask.proportion_true))


class VoxelImage(MaskedNDImage):

    def __init__(self, image_data, mask=None):
        super(VoxelImage, self).__init__(image_data, mask=mask)
        if self.n_dims != 3:
            raise ValueError("Trying to build a VoxelImage with {} channels -"
                             " you must provide a numpy array of size (X, Y,"
                             " Z, K), where K is the number of channels."
                             .format(self.n_channels))


class AbstractSpatialImage(MaskedNDImage):
    r"""
    A 2D image that represents spatial data in some fashion in it's channel
    data. As a result, it contains a :class:'pybug.shape.mesh.base.TriMesh`
    """
    def __init__(self, image_data, mask=None, texture=None,
                 tcoords=None, trilist=None):
        super(AbstractSpatialImage, self).__init__(image_data, mask=mask)
        if self.n_dims != 2:
            raise ValueError("Trying to build an AbstractSpatialImage with {} "
                             "dimensions - has to be 2 dimensional"
                             .format(self.n_dims))
        self.mesh = self._create_mesh_from_shape(trilist, tcoords,
                                                 texture)

    def _generate_points(self):
        raise NotImplementedError()

    def _create_mesh_from_shape(self, trilist, tcoords, texture):
        from pybug.shape.mesh import TriMesh, TexturedTriMesh
        from scipy.spatial import Delaunay
        points = self._generate_points()
        if trilist is None:
            # Delaunay the 2D surface.
            trilist = Delaunay(points[..., :2]).simplices
        if texture is None:
            return TriMesh(points, trilist)
        else:
            if tcoords is None:
                tcoords = self.mask.true_indices.astype(np.float64)
                # scale to [0, 1]
                tcoords = tcoords / np.array(self.shape)
                # (s,t) = (y,x)
                tcoords = np.fliplr(tcoords)
                # move origin to top left
                tcoords[:, 1] = 1.0 - tcoords[:, 1]
            return TexturedTriMesh(points, trilist, tcoords, texture)

    def _view(self, figure_id=None, new_figure=False, mode='image',
              channel=None, masked=True, **kwargs):
        r"""
        View the image using the default image viewer. Before the image is
        rendered the depth values are normalised between 0 and 1. The range
        is then shifted so that the viewable range provides a reasonable
        contrast.

        Parameters
        ----------
        mode : {'image', 'mesh', 'height'}
            The manner in which to render the depth map.

            ========== =========================
            key        description
            ========== =========================
            image      View as a greyscale image
            mesh       View as a triangulated mesh
            height     View as a height map
            ========== =========================

            Default: 'image'

        Returns
        -------
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within
        """
        pixels = self.pixels.copy()
        pixels[np.isinf(pixels)] = np.nan
        pixels = np.abs(pixels)
        pixels /= np.nanmax(pixels)

        mask = None
        if masked:
            mask = self.mask.mask

        if mode is 'image':
            return ImageViewer(figure_id, new_figure,
                               self.n_dims, pixels,
                               channel=channel, mask=mask).render(**kwargs)
        if mode is 'mesh':
            return self.mesh._view(figure_id=figure_id, new_figure=new_figure,
                                   **kwargs)
        else:
            return self._view_extra(figure_id, new_figure, mode, mask,
                                    **kwargs)

    def _view_extra(self, figure_id, new_figure, mode, mask, **kwargs):
        if mode is 'height':
            return DepthImageHeightViewer(
                figure_id, new_figure,
                self.pixels[:, :, 2], mask=mask).render(**kwargs)
        else:
            raise ValueError("Supported mode values are: 'image', 'mesh'"
                             " and 'height'")


class ShapeImage(AbstractSpatialImage):
    r"""
    An image the represents a shape image. Due to the fact a shape image has
    an implicit spatial meaning, it also contains a
    :class:'pybug.shape.mesh.base.TriMesh`. This allows the shape image to be
    treated as an image, but expose an object that represents the shape
    as a mesh.

    Has to be a 2D image, and has to have exactly 3 channels for (X,Y,
    Z) spatial values.
    """

    def __init__(self, image_data, mask=None, texture=None, tcoords=None,
                 trilist=None):
        super(ShapeImage, self).__init__(image_data, mask, texture, tcoords,
                                         trilist)
        if self.n_channels != 3:
            raise ValueError("Trying to build a ShapeImage with {} channels "
                             "- has to have exactly 3 (for X, Y, "
                             "Z)".format(self.n_channels))

    def _generate_points(self):
        return self.masked_pixels


class DepthImage(AbstractSpatialImage):
    r"""
    An image the represents a depth image. Due to the fact a depth image has
    an implicit spatial meaning, a DepthImage also contains a
    :class:'pybug.shape.mesh.base.TriMesh`. This allows the depth image to be
    treated as an image, but expose an object that represents the depth
    as a mesh.

    Will have exactly 1 channel. The numpy array used to build the
    DepthImage is of shape (M, N) - it does not include the channel axis.
    """

    def __init__(self, image_data, mask=None, texture=None, tcoords=None,
                 trilist=None):
        super(DepthImage, self).__init__(image_data[..., None], mask,
                                         texture, tcoords, trilist)
        if self.n_channels != 1:
            raise ValueError("Trying to build a DepthImage with {} channels "
                             "- has to have exactly 1 (for Z values)"
                             .format(self.n_channels))

    @classmethod
    def _init_with_channel(cls, image_data_with_channel, mask):
        return cls(image_data_with_channel[..., 0], mask)

    def _generate_points(self):
        return np.hstack((self.mask.true_indices, self.masked_pixels))

    def _view_extra(self, figure_id, new_figure, mode, mask, **kwargs):
        r"""
        View the image using the default image viewer. Before the image is
        rendered the depth values are normalised between 0 and 1. The range
        is then shifted so that the viewable range provides a reasonable
        contrast.

        Parameters
        ----------
        mode : {'image', 'mesh', 'height'}
            The manner in which to render the depth map.

            ========== =========================
            key        description
            ========== =========================
            image      View as a greyscale image
            mesh       View as a triangulated mesh
            height     View as a height map
            ========== =========================

            Default: 'image'

        Returns
        -------
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within
        """
        if mode is 'height':
            return DepthImageHeightViewer(
                figure_id, new_figure,
                self.pixels[:, :, 0], mask=mask).render(**kwargs)
        else:
            raise ValueError("Supported mode values are: 'image', 'mesh'"
                             " and 'height'")
