from __future__ import division
import abc
import numpy as np
from copy import deepcopy
from skimage.transform import pyramid_gaussian
from skimage.transform.pyramids import _smooth
import scipy.linalg
import PIL.Image as PILImage
from scipy.misc import imrotate

from menpo.base import Vectorizable
from menpo.landmark import Landmarkable
from menpo.transform import Translation, NonUniformScale, UniformScale, \
    AlignmentUniformScale
from menpo.visualize.base import Viewable, ImageViewer
from menpo.image.feature import FeatureExtraction


class ImageBoundaryError(ValueError):
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


class Image(Vectorizable, Landmarkable, Viewable):
    r"""
    An n-dimensional image.

    Images are n-dimensional homogeneous regular arrays of data. Each
    spatially distinct location in the array is referred to as a `pixel`.
    At a pixel, ``k`` distinct pieces of information can be stored. Each
    datum at a pixel is refereed to as being in a `channel`. All pixels in
    the image have the  same number of channels, and all channels have the
    same data-type (float).


    Parameters
    -----------
    image_data: (M, N ..., Q, C) ndarray
        Array representing the image pixels, with the last axis being
        channels.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, image_data):
        Landmarkable.__init__(self)
        image_data = np.array(image_data, copy=True, order='C')
        # This is the degenerate case whereby we can just put the extra axis
        # on ourselves
        if image_data.ndim == 2:
            image_data = image_data[..., None]
        if image_data.ndim < 2:
            raise ValueError("Pixel array has to be 2D (2D shape, implicitly "
                             "1 channel) or 3D+ (2D+ shape, n_channels) "
                             " - a {}D array "
                             "was provided".format(image_data.ndim))
        self.pixels = image_data
        # add FeatureExtraction functionality
        self.features = FeatureExtraction(self)

    @classmethod
    def _init_with_channel(cls, image_data_with_channel, **kwargs):
        r"""
        Constructor that always requires the image has a
        channel on the last axis. Only used by from_vector. By default,
        just calls the constructor. Subclasses with constructors that don't
        require channel axes need to overwrite this.
        """
        return cls(image_data_with_channel, **kwargs)

    @classmethod
    def blank(cls, shape, n_channels=1, fill=0, dtype=np.float, **kwargs):
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
        Subclasses of `Image` need to overwrite this method and
        explicitly call this superclass method:

            super(SubClass, cls).blank(shape,**kwargs)

        in order to appropriately propagate the SubClass type to cls.

        Returns
        -------
        blank_image : :class:`Image`
            A new image of the requested size.
        """
        # Ensure that the '+' operator means concatenate tuples
        shape = tuple(np.ceil(shape))
        if fill == 0:
            pixels = np.zeros(shape + (n_channels,), dtype=dtype)
        else:
            pixels = np.ones(shape + (n_channels,), dtype=dtype) * fill
        return cls._init_with_channel(pixels, **kwargs)

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

        :type: (n_dims,) ndarray
        """
        # noinspection PyUnresolvedReferences
        return np.array(self.shape, dtype=np.double) / 2

    @property
    def _str_shape(self):
        if self.n_dims > 2:
            return ' x '.join(str(dim) for dim in self.shape)
        elif self.n_dims == 2:
            return '{}W x {}H'.format(self.width, self.height)

    def as_vector(self, keep_channels=False):
        r"""
        The vectorized form of this image.

        Parameters
        ----------
        keep_channels : bool, optional

            ========== =================================
            Value      Return shape
            ========== =================================
            ``False``  (``n_pixels``  x ``n_channels``,)
            ``True``   (``n_pixels``, ``n_channels``)
            ========== =================================

            Default: ``False``

        Returns
        -------
        (shape given by keep_channels) ndarray
            Flattened representation of this image, containing all pixel
            and channel information
        """
        if keep_channels:
            return self.pixels.reshape([-1, self.n_channels])
        else:
            return self.pixels.flatten()

    def as_histogram(self, keep_channels=True, bins='unique'):
        r"""
        Histogram binning of the values of this image.

        Parameters
        ----------
        keep_channels : bool, optional
            If set to ``False``, it returns a single histogram for all the
            channels of the image. If set to ``True``, it returns a list of
            histograms, one for each channel.

            Default: ``True``
        bins : 'unique', positive int or sequence of scalars, optional
            If set equal to 'unique', the bins of the histograms are centered
            on the unique values of each channel. If set equal to a positive
            integer, then this is the number of bins. If set equal to a
            sequence of scalars, these will be used as bins centres.

            Default: 'unique'

        Returns
        -------
        hist : array or list with n_channels arrays
            The histogram(s). If keep_channels=False, then hist is an array. If
            keep_channels=True, then hist is a list with len(hist)=n_channels.
        bin_edges : array or list with n_channels arrays
            An array or a list of arrays corresponding to the above histograms
            that store the bins' edges.
            The result in the case of list of arrays can be visualized as:
                for k in range(len(hist)):
                    plt.subplot(1,len(hist),k)
                    width = 0.7 * (bin_edges[k][1] - bin_edges[k][0])
                    center = (bin_edges[k][:-1] + bin_edges[k][1:]) / 2
                    plt.bar(center, hist[k], align='center', width=width)

        Raises
        ------
        ValueError
            Bins can be either 'unique', positive int or a sequence of scalars.
        """
        # parse options
        if isinstance(bins, str):
            if bins == 'unique':
                bins = 0
            else:
                raise ValueError("Bins can be either 'unique', positive int or"
                                 "a sequence of scalars.")
        elif isinstance(bins, int) and bins < 1:
            raise ValueError("Bins can be either 'unique', positive int or a "
                             "sequence of scalars.")
        # compute histogram
        vec = self.as_vector(keep_channels=keep_channels)
        if len(vec.shape) == 1 or vec.shape[1] == 1:
            if bins == 0:
                bins = np.unique(vec)
            hist, bin_edges = np.histogram(vec, bins=bins)
        else:
            hist = []
            bin_edges = []
            num_bins = bins
            for ch in range(vec.shape[1]):
                if bins == 0:
                    num_bins = np.unique(vec[:, ch])
                h_tmp, c_tmp = np.histogram(vec[:, ch], bins=num_bins)
                hist.append(h_tmp)
                bin_edges.append(c_tmp)
        return hist, bin_edges

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
        For BooleanImage's this is rebuilding a boolean image **itself**
        from boolean values. The mask is in no way interpreted in performing
        the operation, in contrast to MaskedImage, where only the masked
        region is used in from_vector{_inplace}() and as_vector().
        """
        self.pixels = vector.reshape(self.pixels.shape)

    def _view(self, figure_id=None, new_figure=False, channels=None,
              **kwargs):
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
        pixels_to_view = self.pixels
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channels=channels).render(**kwargs)

    def glyph(self, vectors_block_size=10, use_negative=False, channels=None):
        r"""
        Create glyph of a feature image. If feature_data has negative values,
        the use_negative flag controls whether there will be created a glyph of
        both positive and negative values concatenated the one on top of the
        other.

        Parameters
        ----------
        vectors_block_size: int
            Defines the size of each block with vectors of the glyph image.
        use_negative: bool
            Defines whether to take into account possible negative values of
            feature_data.
        """
        # first, choose the appropriate channels
        if channels is None:
            pixels = self.pixels[..., :4]
        elif channels != 'all':
            pixels = self.pixels[..., channels]
        else:
            pixels = self.pixels
        # compute the glyph
        negative_weights = -pixels
        scale = np.maximum(pixels.max(), negative_weights.max())
        pos = _create_feature_glyph(pixels, vectors_block_size)
        pos = pos * 255 / scale
        glyph_image = pos
        if use_negative and pixels.min() < 0:
            neg = _create_feature_glyph(negative_weights, vectors_block_size)
            neg = neg * 255 / scale
            glyph_image = np.concatenate((pos, neg))
        glyph = Image(glyph_image)
        # correct landmarks
        from menpo.transform import NonUniformScale
        image_shape = np.array(self.shape, dtype=np.double)
        glyph_shape = np.array(glyph.shape, dtype=np.double)
        nus = NonUniformScale(glyph_shape / image_shape)
        glyph.landmarks = self.landmarks
        nus.apply_inplace(glyph.landmarks)
        return glyph

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
            raise ValueError(
                "Both min and max indices should be 1D numpy arrays of"
                " length n_dims ({})".format(self.n_dims))
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
        slices = [slice(min_i, max_i)
                  for min_i, max_i in
                  zip(list(min_bounded), list(max_bounded))]
        self.pixels = self.pixels[slices].copy()
        # update all our landmarks
        lm_translation = Translation(-min_bounded)
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

    def crop_to_landmarks(self, group=None, label='all', boundary=0,
                          constrain_to_boundary=True):
        r"""
        Crop this image to be bounded around a set of landmarks with an
        optional n_pixel boundary

        Parameters
        ----------
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If
            'all' all landmarks in the group are used.

            Default: 'all'

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

    def crop_to_landmarks_proportion(self, boundary_proportion, group=None,
                                     label='all', minimum=True,
                                     constrain_to_boundary=True):
        r"""
        Crop this image to be bounded around a set of landmarks with a
        border proportional to the landmark spread or range.

        Parameters
        ----------
        boundary_proportion: float
            Additional padding to be added all around the landmarks
            bounds defined as a proportion of the landmarks' range. See
            minimum for a definition of how the range is calculated.
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If
            'all' all landmarks in the group are used.

            Default: 'all'

        minimum: bool, Optional
            If True the specified proportion is relative to the minimum
            value of the landmarks' per-dimension range; if False wrt the
            maximum value of the landmarks' per-dimension range.

            Default: True

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
        if minimum:
            boundary = boundary_proportion * np.min(pc.range())
        else:
            boundary = boundary_proportion * np.max(pc.range())
        self.crop_to_landmarks(group=group, label=label, boundary=boundary,
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
        Return a copy of this image warped into a different reference space.

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
        from menpo.interpolation import scipy_interpolation
        # configure the interpolator we are going to use for the warp
        # currently only scipy is supported but in the future we may have CUDA
        if interpolator == 'scipy':
            _interpolator = scipy_interpolation
        else:
            raise ValueError("Don't understand interpolator '{}': needs to "
                             "be 'scipy'".format(interpolator))

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
        # set any nan values to 0
        sampled_pixel_values[np.isnan(sampled_pixel_values)] = 0
        # build a warped version of the image
        warped_image = self._build_warped_image(template_mask,
                                                sampled_pixel_values)

        if warp_landmarks:
            warped_image.landmarks = self.landmarks
            transform.pseudoinverse.apply_inplace(warped_image.landmarks)
        return warped_image

    def _build_warped_image(self, template_mask, sampled_pixel_values,
                            **kwargs):
        r"""
        Builds the warped image from the template mask and
        sampled pixel values. Overridden for BooleanImage as we can't use
        the usual from_vector_inplace method. All other Image classes share
        the Image implementation.
        """
        warped_image = self.blank(template_mask.shape,
                                  n_channels=self.n_channels, **kwargs)
        warped_image.from_vector_inplace(sampled_pixel_values.ravel())
        return warped_image

    def rescale(self, scale, interpolator='scipy', round='ceil', **kwargs):
        r"""
        Return a copy of this image, rescaled by a given factor.
        All image information (landmarks) are rescaled appropriately.

        Parameters
        ----------
        scale : float or tuple
            The scale factor. If a tuple, the scale to apply to each dimension.
            If a single float, the scale will be applied uniformly across
            each dimension.
        interpolator : 'scipy', optional
            The interpolator that should be used to perform the warp.

            Default: 'scipy'
        round: {'ceil', 'floor', 'round'}
            Rounding function to be applied to floating point shapes.

            Default: 'ceil'
        kwargs : dict
            Passed through to the interpolator. See `menpo.interpolation`
            for details. Note that mode is set to nearest to avoid numerical
            issues, and cannot be changed here by the user.

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
        # Pythonic way of converting to list if we are passed a single float
        try:
            if len(scale) < self.n_dims:
                raise ValueError(
                    'Must provide a scale per dimension.'
                    '{} scales were provided, {} were expected.'.format(
                        len(scale), self.n_dims
                    )
                )
        except TypeError:  # Thrown when len() is called on a float
            scale = [scale] * self.n_dims

        # Make sure we have a numpy array
        scale = np.asarray(scale)
        for s in scale:
            if s <= 0:
                raise ValueError('Scales must be positive floats.')

        transform = NonUniformScale(scale)
        from menpo.image.boolean import BooleanImage
        # use the scale factor to make the template mask bigger
        template_mask = BooleanImage.blank(transform.apply(self.shape),
                                           round=round)
        # due to image indexing, we can't just apply the pseduoinverse
        # transform to achieve the scaling we want though!
        # Consider a 3x rescale on a 2x4 image. Looking at each dimension:
        #    H 2 -> 6 so [0-1] -> [0-5] = 5/1 = 5x
        #    W 4 -> 12 [0-3] -> [0-11] = 11/3 = 3.67x
        # => need to make the correct scale per dimension!
        shape = np.array(self.shape, dtype=np.float)
        # scale factors = max_index_after / current_max_index
        # (note that max_index = length - 1, as 0 based)
        scale_factors = (scale * shape - 1) / (shape - 1)
        inverse_transform = NonUniformScale(scale_factors).pseudoinverse
        # for rescaling we enforce that mode is nearest to avoid num. errors
        if 'mode' in kwargs:
            raise ValueError("Cannot set 'mode' kwarg on rescale - set to "
                             "'nearest' to avoid numerical errors")
        kwargs['mode'] = 'nearest'
        # Note here we pass warp_mask to warp_to. In the case of
        # Images that aren't MaskedImages this kwarg will
        # harmlessly fall through so we are fine.
        return self.warp_to(template_mask, inverse_transform,
                            warp_landmarks=True,
                            interpolator=interpolator, **kwargs)

    def rescale_to_reference_shape(self, reference_shape, group=None,
                                       label='all', interpolator='scipy',
                                       round='ceil', **kwargs):
        r"""
        Return a copy of this image, rescaled so that the scale of a
        particular group of landmarks matches the scale of the passed
        reference landmarks.

        Parameters
        ----------
        reference_shape: :class:`menpo.shape.pointcloud`
            The reference shape to which the landmarks scale will be matched
            against.
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None
        label: string, Optional
            The label of of the landmark manager that you wish to use. If
            'all' all landmarks in the group are used.

            Default: 'all'
        interpolator : 'scipy' or 'c', optional
            The interpolator that should be used to perform the warp.

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
        """
        pc = self.landmarks[group][label].lms
        scale = AlignmentUniformScale(pc, reference_shape).as_vector()
        return self.rescale(scale, interpolator=interpolator,
                            round=round, **kwargs)

    def rescale_landmarks_to_diagonal_range(self, diagonal_range, group=None,
                                            label='all', interpolator='scipy',
                                            round='ceil', **kwargs):
        r"""
        Return a copy of this image, rescaled so that the diagonal_range of the
        bounding box containing its landmarks matches the specified diagonal_range
        range.

        Parameters
        ----------
        diagonal_range: :class:`menpo.shape.pointcloud`
            The diagonal_range range that we want the landmarks of the returned
            image to have.
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None
        label: string, Optional
            The label of of the landmark manager that you wish to use. If
            'all' all landmarks in the group are used.

            Default: 'all'
        interpolator : 'scipy', optional
            The interpolator that should be used to perform the warp.

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
        """
        x, y = self.landmarks[group][label].lms.range()
        scale = diagonal_range / np.sqrt(x**2 + y**2)
        return self.rescale(scale, interpolator=interpolator,
                            round=round, **kwargs)

    def resize(self, shape, interpolator='scipy', **kwargs):
        r"""
        Return a copy of this image, resized to a particular shape.
        All image information (landmarks, the mask in the case of
        :class:`MaskedImage`) is resized appropriately.

        Parameters
        ----------
        shape : tuple
            The new shape to resize to.
        interpolator : 'scipy' or 'c', optional
            The interpolator that should be used to perform the warp.

            Default: 'scipy'
        kwargs : dict
            Passed through to the interpolator. See `menpo.interpolation`
            for details.

        Returns
        -------
        resized_image : type(self)
            A copy of this image, resized.

        Raises
        ------
        ValueError:
            If the number of dimensions of the new shape does not match
            the number of dimensions of the image.
        """
        shape = np.asarray(shape)
        if len(shape) != self.n_dims:
            raise ValueError(
                'Dimensions must match.'
                '{} dimensions provided, {} were expected.'.format(
                    shape.shape, self.n_dims))
        scales = shape.astype(np.float) / self.shape
        # Have to round the shape when scaling to deal with floating point
        # errors. For example, if we want (250, 250), we need to ensure that
        # we get (250, 250) even if the number we obtain is 250 to some
        # floating point inaccuracy.
        return self.rescale(scales, interpolator=interpolator,
                            round='round', **kwargs)

    def gaussian_pyramid(self, n_levels=3, downscale=2, sigma=None,
                         order=1, mode='reflect', cval=0):
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
        max_layer = n_levels - 1
        pyramid = pyramid_gaussian(self.pixels, max_layer=max_layer,
                                   downscale=downscale, sigma=sigma,
                                   order=order, mode=mode, cval=cval)

        for j, image_data in enumerate(pyramid):
            image = self.__class__(image_data)

            # rescale and reassign existent landmark
            image.landmarks = self.landmarks
            transform = UniformScale(downscale ** j, self.n_dims)
            transform.pseudoinverse.apply_inplace(image.landmarks)
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
        for j in range(n_levels):
            if j is 0:
                yield self
            else:
                if sigma is None:
                    sigma_aux = 2 * downscale**j / 6.0
                else:
                    sigma_aux = sigma

                image_data = _smooth(self.pixels, sigma=sigma_aux,
                                     mode=mode, cval=cval)
                image = self.__class__(image_data)

                # rescale and reassign existent landmark
                image.landmarks = self.landmarks
                yield image

    def as_greyscale(self, mode='luminosity', channel=None):
        r"""
        Returns a greyscale version of the image. If the image does *not*
        represent a 2D RGB image, then the 'luminosity' mode will fail.

        Parameters
        ----------
        mode : {'average', 'luminosity', 'channel'}
            'luminosity' - Calculates the luminance using the CCIR 601 formula
                ``Y' = 0.2989 R' + 0.5870 G' + 0.1140 B'``
            'average' - intensity is an equal average of all three channels
            'channel' - a specific channel is used

            Default 'luminosity'

        channel: int, optional
            The channel to be taken. Only used if mode is 'channel'.

            Default: None

        Returns
        -------
        greyscale_image: :class:`MaskedImage`
            A copy of this image in greyscale.
        """
        greyscale = deepcopy(self)
        if mode == 'luminosity':
            if self.n_dims != 2:
                raise ValueError("The 'luminosity' mode only works on 2D RGB"
                                 "images. {} dimensions found, "
                                 "2 expected.".format(self.n_dims))
            elif self.n_channels != 3:
                raise ValueError("The 'luminosity' mode only works on RGB"
                                 "images. {} channels found, "
                                 "3 expected.".format(self.n_channels))

            # Invert the transformation matrix to get more precise values
            T = scipy.linalg.inv(np.array([[1.0, 0.956, 0.621],
                                           [1.0, -0.272, -0.647],
                                           [1.0, -1.106, 1.703]]))
            coef = T[0, :]
            pixels = np.dot(greyscale.pixels, coef.T)
        elif mode == 'average':
            pixels = np.mean(greyscale.pixels, axis=-1)
        elif mode == 'channel':
            if channel is None:
                raise ValueError("For the 'channel' mode you have to provide"
                                 " a channel index")
            pixels = greyscale.pixels[..., channel].copy()
        else:
            raise ValueError("Unknown mode {} - expected 'luminosity', "
                             "'average' or 'channel'.".format(mode))

        greyscale.pixels = pixels[..., None]
        return greyscale

    def as_PILImage(self):
        r"""
        Return a PIL copy of the image. Scales the image by ``255`` and
        converts to ``np.uint8``. Image must only have 1 or 3 channels and
        be two dimensional.

        Returns
        -------
        pil_image : ``PILImage``
            PIL copy of image as ``np.uint8``

        Raises
        ------
        ValueError if image is not 2D and 1 channel or 3 channels.
        """
        if self.n_dims != 2 or self.n_channels not in [1, 3]:
            raise ValueError('Can only convert greyscale or RGB 2D images. '
                             'Received a {} channel {}D image.'.format(
                self.n_channels, self.ndims))
        return PILImage.fromarray((self.pixels * 255).astype(np.uint8))

    def __str__(self):
        return ('{} {}D Image with {} channels'.format(
            self._str_shape, self.n_dims, self.n_channels))

    @property
    def has_landmarks_outside_bounds(self):
        """
        Indicates whether there are landmarks located outside the image bounds.

        :type: bool
        """
        if self.landmarks.has_landmarks:
            for l_group in self.landmarks:
                pc = l_group[1].lms.points
                if np.any(np.logical_or(self.shape - pc < 1, pc < 0)):
                    return True
        return False

    def constrain_landmarks_to_bounds(self):
        r"""
        Move landmarks that are located outside the image bounds on the bounds.
        """
        if self.has_landmarks_outside_bounds:
            for l_group in self.landmarks:
                l = self.landmarks[l_group[0]]
                for k in range(l.lms.points.shape[1]):
                    tmp = l.lms.points[:, k]
                    tmp[tmp < 0] = 0
                    tmp[tmp > self.shape[k] - 1] = self.shape[k] - 1
                    l.lms.points[:, k] = tmp
                self.landmarks[l_group[0]] = l


def _create_feature_glyph(features, vbs):
    r"""
    Create glyph of feature pixels.

    Parameters
    ----------
    feature_type : (N, D) ndarray
        The feature pixels to use.
    vbs: int
        Defines the size of each block with vectors of the glyph image.
    """
    # vbs = Vector block size
    num_bins = features.shape[2]
    # construct a "glyph" for each orientation
    block_image_temp = np.zeros((vbs, vbs))
    # Create a vertical line of ones, to be the first vector
    block_image_temp[:, round(vbs / 2) - 1:round(vbs / 2) + 1] = 1
    block_im = np.zeros((block_image_temp.shape[0],
                         block_image_temp.shape[1],
                         num_bins))
    # First vector as calculated above
    block_im[:, :, 0] = block_image_temp
    # Number of bins rotations to create an 'asterisk' shape
    for i in range(1, num_bins):
        block_im[:, :, i] = imrotate(block_image_temp, -i * vbs)

    # make pictures of positive feature_data by adding up weighted glyphs
    features[features < 0] = 0
    glyph_im = np.sum(block_im[None, None, :, :, :] *
                      features[:, :, None, None, :], axis=-1)
    glyph_im = np.bmat(glyph_im.tolist())
    return glyph_im
