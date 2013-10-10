import abc
import numpy as np
from copy import deepcopy
from pybug.base import Vectorizable
from pybug.landmark import Landmarkable
from pybug.transform.affine import Translation
from pybug.visualize.base import Viewable, ImageViewer


class ImageBoundaryError(Exception):
    r"""
    Exception that is thrown when an attempt is made to crop an image beyond
    the edge of it's boundary.

    requested_min : (d,) ndarray
        The per-dimension minimum index requested for the crop
    requested_max : (d,) ndarray
        The per-dimension maximum index requested for the crop
    snapped_min : (d,) ndarray
        The per-dimension minimum index that could be used if the crop was
        constrained to the image boundaries.
    requested_max : (d,) ndarray
        The per-dimension maximum index that could be used if the crop was
        constrained to the image boundaries.
    """
    def __init__(self, requested_min, requested_max, snapped_min,
                 snapped_max):
        super(ImageBoundaryError, self).__init__()
        self.requested_min = requested_min
        self.requested_max = requested_max
        self.snapped_min = snapped_min
        self.snapped_max = snapped_max


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

    __metaclass__ = abc.ABCMeta

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

    def blank(*args, **kwargs):
        raise NotImplementedError

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
        # noinspection PyUnresolvedReferences
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

    def from_vector_inplace(self, vector):
        r"""
        Takes a flattened vector and update this image by
        reshaping the vector to the correct dimensions.

        Parameters
        ----------
        vector : (``n_pixels``,) np.bool ndarray
            A vector vector of all the pixels of a BooleanImage.


        Notes
        -----
        For BooleanNDImage's this is rebuilding a boolean image **itself**
        from boolean values. The mask is in no way interpreted in performing
        the operation, in contrast to MaskedNDImage, where only the masked
        region is used in from_vector{_inplace}() and as_vector().
        """
        self.pixels = vector.reshape(self.pixels.shape)

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
            boundary. If False, an ImageBoundaryError will be raised if an
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
        min_indices = np.floor(min_indices)
        max_indices = np.ceil(max_indices)
        if not (min_indices.size == max_indices.size == self.n_dims):
            raise ValueError("Both min and max indices should be 1D numpy "
                             "arrays of length n_dims ({})".format(
                             self.n_dims))
        elif not np.all(max_indices > min_indices):
            raise ValueError("All max indices must be greater that the min "
                             "indices")
        min_bounded = self.constrain_points_to_bounds(min_indices)
        max_bounded = self.constrain_points_to_bounds(max_indices)
        if not constrain_to_boundary and not (
                np.all(min_bounded == min_indices) or
                np.all(max_bounded == max_indices)):
            # points have been constrained and the user didn't want this -
            raise ImageBoundaryError(min_indices, max_indices,
                                     min_bounded, max_bounded)
        # noinspection PyArgumentList
        slices = [slice(min_i, max_i)
                  for min_i, max_i in
                  zip(list(min_bounded), list(max_bounded))]
        self.pixels = self.pixels[slices]
        # update all our landmarks
        lm_translation = Translation(-min_indices)
        lm_translation.apply_inplace(self.landmarks)
        return self

    def cropped_copy(self, min_indices, max_indices,
                     constrain_to_boundary=False):
        r"""
        Return a cropped copy of this image using the given minimum and
        maximum indices. Landmarks are correctly adjusted so they maintain
        their position relative to the newly cropped image.

        Parameters
        -----------
        min_indices: (n_dims, ) ndarray
            The minimum index over each dimension

        max_indices: (n_dims, ) ndarray
            The maximum index over each dimension

        constrain_to_boundary: boolean, optional
            If True the crop will be snapped to not go beyond this images
            boundary. If False, an ImageBoundaryError will be raised if an
            attempt is made to go beyond the edge of the image.

            Default: True

        Returns
        -------
        cropped_image : :class:`type(self)`
            A new instance of self, but cropped.


        Raises
        ------
        ValueError
            min_indices and max_indices both have to be of length n_dims.
            All max_indices must be greater than min_indices.

        ImageBoundaryError
            Raised if constrain_to_boundary is False, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        cropped_image = deepcopy(self)
        return cropped_image.crop(min_indices, max_indices,
                                  constrain_to_boundary=constrain_to_boundary)

    def crop_to_landmarks(self, group=None, label=None, boundary=0,
                          constrain_to_boundary=True):
        r"""
        Crop this image to be bounded just around a set of landmarks

        Parameters
        ----------
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
             all landmarks in the group are used.

            Default: None

        boundary: int, Optional
            An extra padding to be added all around the landmarks bounds.

            Default: 0

        constrain_to_boundary: boolean, optional
            If True the crop will be snapped to not go beyond this images
            boundary. If False, an ImageBoundaryError will be raised if an
            attempt is made to go beyond the edge of the image.

            Default: True

        Raises
        ------
        ImageBoundaryError
            Raised if constrain_to_boundary is False, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        pc = self.landmarks[group][label].lms
        min_indices, max_indices = pc.bounds(boundary=boundary)
        self.crop(min_indices, max_indices,
                  constrain_to_boundary=constrain_to_boundary)

    def constrain_points_to_bounds(self, points):
        r"""
        Constrains the points provided to be within the bounds of this
        image.

        Parameters
        ----------

        points: (d,) ndarray
            points to be snapped to the image boundaries

        Returns
        -------

        bounded_points: (d,) ndarray
            points snapped to not stray outside the image edges

        """
        bounded_points = points.copy()
        # check we don't stray under any edges
        bounded_points[bounded_points < 0] = 0
        # check we don't stray over any edges
        shape = np.array(self.shape)
        over_image = (shape - bounded_points) < 0
        bounded_points[over_image] = shape[over_image]
        return bounded_points

    def warp_to(self, template_mask, transform, warp_landmarks=False,
                interpolator='scipy', **kwargs):
        r"""
        Warps this image into a different reference space.

        Parameters
        ----------
        template_mask : :class:`pybug.image.boolean.BooleanNDImage`
            Defines the shape of the result, and what pixels should be
            sampled.
        transform : :class:`pybug.transform.base.Transform`
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
            Passed through to the interpolator. See `pybug.interpolation`
            for details.

        Returns
        -------
        warped_image : type(self)
            A copy of this image, warped.
        """
        from pybug.interpolation import c_interpolation, scipy_interpolation
        # configure the interpolator we are going to use for the warp
        if interpolator == 'scipy':
            _interpolator = scipy_interpolation
        elif interpolator == 'c':
            _interpolator = c_interpolation
        else:
            raise ValueError("Don't understand interpolator '{}': needs to "
                             "be either 'scipy' or 'c'".format(interpolator))

        if self.n_dims != transform.n_dims:
            raise ValueError(
                "Trying to warp a {}D image with a {}D transform "
                "(they must match)".format(self.n_dims, transform.n_dims))

        template_points = template_mask.true_indices
        points_to_sample = transform.apply(template_points).T
        # we want to sample each channel in turn, returning a vector of sampled
        # pixels. Store those in a (n_pixels, n_channels) array.
        sampled_pixel_values = _interpolator(self.pixels, points_to_sample,
                                             **kwargs)

        # Set all NaN pixels to 0
        sampled_pixel_values = np.nan_to_num(sampled_pixel_values)
        # build a warped version of the image
        warped_image = self._build_warped_image(template_mask,
                                                sampled_pixel_values)

        if warp_landmarks:
            raise Exception("Warp Landmarks is not supported until Transform"
                            ".pseudoinverse is")
            warped_image.landmarks = self.landmarks
            transform.pseudoinverse.apply_inplace(warped_image.landmarks)
        return warped_image

    def _build_warped_image(self, template_mask, sampled_pixel_values):
        r"""
        Builds the warped image from the template mask and
        sampled pixel values. Overridden for BooleanNDImage as we can't use
        the usual from_vector_inplace method. All other Image classes share
        the MaskedNDImage implementation.
        """
        raise NotImplementedError
