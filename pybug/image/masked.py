from copy import deepcopy
import itertools
import numpy as np
from scipy.ndimage import binary_erosion
from pybug.image.base import AbstractNDImage
from pybug.image.boolean import BooleanNDImage
from pybug.visualize.base import ImageViewer


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
            eroded_mask = binary_erosion(self.mask.pixels, iterations=1)

            # replace the eroded mask with the diff between the two
            # masks
            np.logical_and(~eroded_mask, self.mask.pixels, eroded_mask)
            # nullify all the boundary values in the grad image
            grad_image.pixels[eroded_mask] = 0.0

        return grad_image

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

        if self.n_dims != 2:
            raise ValueError("can only constrain mask on 2D images.")
        if len(self.landmarks) == 0:
            raise ValueError("There are no attached landmarks to "
                             "infer a mask from")
        if group is None:
            if len(self.landmarks) > 1:
                raise ValueError("no group was provided and there are "
                                 "multiple groups. Specify a group, "
                                 "e.g. {}".format(self.landmarks.keys()[0]))
            else:
                group = self.landmarks.keys()[0]

        if label is None:
            pc = self.landmarks[group].all_landmarks
        else:
            pc = self.landmarks[group].with_label(label).all_landmarks

        # Delaunay as no trilist provided
        pwa = PiecewiseAffineTransform(pc.points, pc.points)
        try:
            pwa.apply(self.mask.all_indices)
        except TriangleContainmentError, e:
            self.mask.update_from_vector(~e.points_outside_source_domain)
