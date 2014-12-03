from __future__ import division
import abc
from warnings import warn

import numpy as np
import scipy.linalg
import PIL.Image as PILImage

from menpo.base import Vectorizable
from menpo.landmark import LandmarkableViewable
from menpo.transform import (Translation, NonUniformScale,
                             AlignmentUniformScale, Affine, Rotation)
from menpo.visualize.base import ImageViewer
from .interpolation import scipy_interpolation, cython_interpolation
from .extract_patches import extract_patches_cython


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


def indices_for_image_of_shape(shape):
    r"""
    The indices of all pixels in an image with a given shape (without
    channel information).

    :type: (`n_dims`, `n_pixels`) ndarray

    """
    return np.indices(shape).reshape([len(shape), -1]).T


class Image(Vectorizable, LandmarkableViewable):
    r"""
    An n-dimensional image.

    Images are n-dimensional homogeneous regular arrays of data. Each
    spatially distinct location in the array is referred to as a `pixel`.
    At a pixel, `k` distinct pieces of information can be stored. Each
    datum at a pixel is refereed to as being in a `channel`. All pixels in
    the image have the same number of channels, and all channels have the
    same data-type (`float`).

    Parameters
    -----------
    image_data : ``(M, N ..., Q, C)`` `ndarray`
        Array representing the image pixels, with the last axis being
        channels.

    copy : `bool`, optional
        If ``False``, the ``image_data`` will not be copied on assignment.
        Note that this will miss out on additional checks. Further note that we
        still demand that the array is C-contiguous - if it isn't, a copy will
        be generated anyway.
        In general, this should only be used if you know what you are doing.

    Raises
    ------
    Warning
        If ``copy=False`` cannot be honoured
    ValueError
        If the pixel array is malformed
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, image_data, copy=True):
        super(Image, self).__init__()
        if not copy:
            if not image_data.flags.c_contiguous:
                image_data = np.array(image_data, copy=True, order='C')
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
        else:
            image_data = np.array(image_data, copy=True, order='C')
            # Degenerate case whereby we can just put the extra axis
            # on ourselves
            if image_data.ndim == 2:
                image_data = image_data[..., None]
            if image_data.ndim < 2:
                raise ValueError(
                    "Pixel array has to be 2D (2D shape, implicitly "
                    "1 channel) or 3D+ (2D+ shape, n_channels) "
                    " - a {}D array "
                    "was provided".format(image_data.ndim))
        self.pixels = image_data

    def as_masked(self, mask=None, copy=True):
        r"""
        Return a copy of this image with an attached mask behavior.

        A custom mask may be provided, or None. See the :map:`MaskedImage`
        constructor for details of how the kwargs will be handled.

        Parameters
        ----------
        mask : `ndarray` with shape of ``self.shape`` or :map:`BooleanImage`
            A mask to attach to the newly generated masked image.
        copy : `bool`, optional
            If ``False``, the produced :map:`MaskedImage` will share pixels with
            ``self``. Only suggested to be used for performance.

        Returns
        -------
        masked_image : :map:`MaskedImage`
            An image with the same pixels and landmarks as this one, but with
            a mask.
        """
        img = MaskedImage(self.pixels, mask=mask, copy=copy)
        img.landmarks = self.landmarks
        return img

    @classmethod
    def blank(cls, shape, n_channels=1, fill=0, dtype=np.float):
        r"""
        Returns a blank image.

        Parameters
        ----------
        shape : `tuple` or `list`
            The shape of the image. Any floating point values are rounded up
            to the nearest integer.
        n_channels : `int`, optional
            The number of channels to create the image with.
        fill : `int`, optional
            The value to fill all pixels with.
        dtype : numpy data type, optional
            The data type of the image.

        Returns
        -------
        blank_image : :map:`Image`
            A new image of the requested size.
        """
        # Ensure that the '+' operator means concatenate tuples
        shape = tuple(np.ceil(shape).astype(np.int))
        if fill == 0:
            pixels = np.zeros(shape + (n_channels,), dtype=dtype)
        else:
            pixels = np.ones(shape + (n_channels,), dtype=dtype) * fill
        # We know there is no need to copy...
        return cls(pixels, copy=False)

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the image. The minimum possible ``n_dims``
        is 2.

        :type: `int`
        """
        return len(self.shape)

    @property
    def n_pixels(self):
        r"""
        Total number of pixels in the image (``prod(shape)``,)

        :type: `int`
        """
        return self.pixels[..., 0].size

    @property
    def n_elements(self):
        r"""
        Total number of data points in the image
        (``prod(shape)``, ``n_channels``)

        :type: `int`
        """
        return self.pixels.size

    @property
    def n_channels(self):
        """
        The number of channels on each pixel in the image.

        :type: `int`
        """
        return self.pixels.shape[-1]

    @property
    def width(self):
        r"""
        The width of the image.

        This is the width according to image semantics, and is thus the size
        of the **second** dimension.

        :type: `int`
        """
        return self.pixels.shape[1]

    @property
    def height(self):
        r"""
        The height of the image.

        This is the height according to image semantics, and is thus the size
        of the **first** dimension.

        :type: `int`
        """
        return self.pixels.shape[0]

    @property
    def shape(self):
        r"""
        The shape of the image
        (with ``n_channel`` values at each point).

        :type: `tuple`
        """
        return self.pixels.shape[:-1]

    @property
    def diagonal(self):
        r"""
        The diagonal size of this image

        :type: `float`
        """
        return np.sqrt(np.sum(np.array(self.shape) ** 2))

    @property
    def centre(self):
        r"""
        The geometric centre of the Image - the subpixel that is in the
        middle.

        Useful for aligning shapes and images.

        :type: (``n_dims``,) `ndarray`
        """
        # noinspection PyUnresolvedReferences
        return np.array(self.shape, dtype=np.double) / 2

    @property
    def _str_shape(self):
        if self.n_dims > 2:
            return ' x '.join(str(dim) for dim in self.shape)
        elif self.n_dims == 2:
            return '{}W x {}H'.format(self.width, self.height)

    def indices(self):
        r"""
        Return the indices of all pixels in this image.

        :type: (``n_dims``, ``n_pixels``) ndarray

        """
        return indices_for_image_of_shape(self.shape)

    def _as_vector(self, keep_channels=False):
        r"""
        The vectorized form of this image.

        Parameters
        ----------
        keep_channels : `bool`, optional

            ========== =============================
            Value      Return shape
            ========== =============================
            `False`    (`n_pixels` * `n_channels`,)
            `True`     (`n_pixels`, `n_channels`)
            ========== =============================

        Returns
        -------
        vec : (shape given by ``keep_channels``) `ndarray`
            Flattened representation of this image, containing all pixel
            and channel information
        """
        if keep_channels:
            return self.pixels.reshape([-1, self.n_channels])
        else:
            return self.pixels.ravel()

    def from_vector(self, vector, n_channels=None, copy=True):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels.

        The `n_channels` argument is useful for when we want to add an extra
        channel to an image but maintain the shape. For example, when
        calculating the gradient.

        Note that landmarks are transferred in the process.

        Parameters
        ----------
        vector : (`n_parameters`,)
            A flattened vector of all pixels and channels of an image.

        n_channels : int, optional
            If given, will assume that vector is the same shape as this image,
            but with a possibly different number of channels

            Default: Use the existing image channels

        copy : bool, optional
            If False the vector will not be copied in creating the new image.

            Default: True

        Returns
        -------
        image : :map:`Image`
            New image of same shape as this image and the number of
            specified channels.

        Raises
        ------

        Warning : If the copy=False flag cannot be honored

        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels is None else n_channels
        image_data = vector.reshape(self.shape + (n_channels,))
        new_image = Image(image_data, copy=copy)
        new_image.landmarks = self.landmarks
        return new_image

    def from_vector_inplace(self, vector, copy=True):
        r"""
        Takes a flattened vector and update this image by
        reshaping the vector to the correct dimensions.

        Parameters
        ----------
        vector : (`n_pixels`,) np.bool ndarray
            A vector vector of all the pixels of a BooleanImage.

        copy: bool, optional
            If False, the vector will be set as the pixels. If True a copy of
            the vector is taken.

            Default: True

        Raises
        ------
        Warning : If copy=False flag cannot be honored

        Notes
        -----
        For BooleanImage's this is rebuilding a boolean image **itself**
        from boolean values. The mask is in no way interpreted in performing
        the operation, in contrast to MaskedImage, where only the masked
        region is used in from_vector{_inplace}() and as_vector().
        """
        image_data = vector.reshape(self.pixels.shape)
        if not copy:
            if not image_data.flags.c_contiguous:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
                image_data = np.array(image_data, copy=True, order='C')
        else:
            image_data = np.array(image_data, copy=True, order='C')
        self.pixels = image_data

    def extract_channels(self, channels):
        r"""
        A copy of this image with only the specified channels.

        Parameters
        ----------
        channels : `int` or `[int]`
            The channel index or list of channel indices to retain

        Returns
        -------
        image : type(self)
            A copy of this image with only the channels requested.
        """
        copy = self.copy()
        if not isinstance(channels, list):
            channels = [channels]  # ensure we don't remove the channel axis
        copy.pixels = self.pixels[..., channels]
        return copy

    def as_histogram(self, keep_channels=True, bins='unique'):
        r"""
        Histogram binning of the values of this image.

        Parameters
        ----------
        keep_channels : bool, optional
            If set to `False`, it returns a single histogram for all the
            channels of the image. If set to `True`, it returns a list of
            histograms, one for each channel.

            Default: `True`
        bins : 'unique', positive int or sequence of scalars, optional
            If set equal to 'unique', the bins of the histograms are centred
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
                    centre = (bin_edges[k][:-1] + bin_edges[k][1:]) / 2
                    plt.bar(centre, hist[k], align='center', width=width)

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

    def view(self, figure_id=None, new_figure=False, channels=None,
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

    def view_widget(self, popup=False):
        r"""
        Visualizes the image object using the
        menpo.visualize.widgets.visualize_images widget.

        Parameters
        -----------
        popup : `boolean`, optional
            If enabled, the widget will appear as a popup window.
        """
        from menpo.visualize import visualize_images
        visualize_images(self, figure_size=(7, 7), popup=popup)

    def gradient(self, **kwargs):
        r"""
        Returns an :map:`Image` which is the gradient of this one. In the case
        of multiple channels, it returns the gradient over each axis over
        each channel as a flat list.

        Returns
        -------
        gradient : :map:`Image`
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length `2`.
            The length of a 2D, 3-channel image, will have length `6`.
        """
        from menpo.feature import gradient
        return gradient(self)

    def crop_inplace(self, min_indices, max_indices,
                     constrain_to_boundary=True):
        r"""
        Crops this image using the given minimum and maximum indices.
        Landmarks are correctly adjusted so they maintain their position
        relative to the newly cropped image.

        Parameters
        -----------
        min_indices : (n_dims, ) ndarray
            The minimum index over each dimension

        max_indices : (n_dims, ) ndarray
            The maximum index over each dimension

        constrain_to_boundary : boolean, optional
            If `True` the crop will be snapped to not go beyond this images
            boundary. If `False`, an :map:`ImageBoundaryError` will be raised if
            an attempt is made to go beyond the edge of the image.

            Default: `True`

        Returns
        -------
        cropped_image : `type(self)`
            This image, cropped.

        Raises
        ------
        `ValueError`
            `min_indices` and `max_indices` both have to be of length `n_dims`.
            All `max_indices` must be greater than `min_indices`.

        `ImageBoundaryError`
            Raised if `constrain_to_boundary` is `False`, and an attempt is made
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
        slices = [slice(int(min_i), int(max_i))
                  for min_i, max_i in
                  zip(list(min_bounded), list(max_bounded))]
        self.pixels = self.pixels[slices].copy()
        # update all our landmarks
        lm_translation = Translation(-min_bounded)
        lm_translation.apply_inplace(self.landmarks)
        return self

    def crop(self, min_indices, max_indices,
             constrain_to_boundary=False):
        r"""
        Return a cropped copy of this image using the given minimum and
        maximum indices. Landmarks are correctly adjusted so they maintain
        their position relative to the newly cropped image.

        Parameters
        -----------
        min_indices : (n_dims, ) ndarray
            The minimum index over each dimension

        max_indices : (n_dims, ) ndarray
            The maximum index over each dimension

        constrain_to_boundary : boolean, optional
            If `True` the crop will be snapped to not go beyond this images
            boundary. If `False`, an :map:`ImageBoundaryError` will be raised if
            an attempt is made to go beyond the edge of the image.

            Default: `True`

        Returns
        -------
        cropped_image : :class:`type(self)`
            A new instance of self, but cropped.


        Raises
        ------
        ValueError
            `min_indices` and `max_indices` both have to be of length `n_dims`.
            All `max_indices` must be greater than `min_indices`.

        ImageBoundaryError
            Raised if `constrain_to_boundary` is `False`, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        cropped_image = self.copy()
        return cropped_image.crop_inplace(
            min_indices, max_indices,
            constrain_to_boundary=constrain_to_boundary)

    def crop_to_landmarks_inplace(self, group=None, label=None, boundary=0,
                                  constrain_to_boundary=True):
        r"""
        Crop this image to be bounded around a set of landmarks with an
        optional `n_pixel` boundary

        Parameters
        ----------
        group : string, Optional
            The key of the landmark set that should be used. If `None`,
            and if there is only one set of landmarks, this set will be used.

            Default: `None`
        label : string, Optional
            The label of of the landmark manager that you wish to use. If
            `None` all landmarks in the group are used.

            Default: `None`
        boundary : int, Optional
            An extra padding to be added all around the landmarks bounds.

            Default: `0`
        constrain_to_boundary : boolean, optional
            If `True` the crop will be snapped to not go beyond this images
            boundary. If `False`, an :map`ImageBoundaryError` will be raised if
            an attempt is made to go beyond the edge of the image.

            Default: `True`

        Returns
        -------
        image : :map:`Image`
            This image, cropped to it's landmarks.

        Raises
        ------
        ImageBoundaryError
            Raised if `constrain_to_boundary` is `False`, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        pc = self.landmarks[group][label]
        bb = pc.bounds(boundary=boundary)
        return self.crop_inplace(bb.min, bb.max,
                                 constrain_to_boundary=constrain_to_boundary)

    def crop_to_landmarks_proportion_inplace(self, boundary_proportion,
                                             group=None, label=None,
                                             minimum=True,
                                             constrain_to_boundary=True):
        r"""
        Crop this image to be bounded around a set of landmarks with a
        border proportional to the landmark spread or range.

        Parameters
        ----------
        boundary_proportion : float
            Additional padding to be added all around the landmarks
            bounds defined as a proportion of the landmarks' range. See
            the minimum parameter for a definition of how the range is
            calculated.
        group : string, Optional
            The key of the landmark set that should be used. If `None`,
            and if there is only one set of landmarks, this set will be used.

            Default: `None`
        label : string, Optional
            The label of of the landmark manager that you wish to use. If
            `None` all landmarks in the group are used.

            Default: `None`
        minimum : bool, Optional
            If `True` the specified proportion is relative to the minimum
            value of the landmarks' per-dimension range; if `False` w.r.t. the
            maximum value of the landmarks' per-dimension range.

            Default: `True`
        constrain_to_boundary : boolean, optional
            If `True`, the crop will be snapped to not go beyond this images
            boundary. If `False`, an :map:`ImageBoundaryError` will be raised if
            an attempt is made to go beyond the edge of the image.

            Default: `True`

        Returns
        -------
        image : :map:`Image`
            This image, cropped to it's landmarks with a border proportional to
            the landmark spread or range.

        Raises
        ------
        ImageBoundaryError
            Raised if `constrain_to_boundary` is `False`, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        pc = self.landmarks[group][label]
        if minimum:
            boundary = boundary_proportion * np.min(pc.range())
        else:
            boundary = boundary_proportion * np.max(pc.range())
        return self.crop_to_landmarks_inplace(
            group=group, label=label, boundary=boundary,
            constrain_to_boundary=constrain_to_boundary)

    def constrain_points_to_bounds(self, points):
        r"""
        Constrains the points provided to be within the bounds of this
        image.

        Parameters
        ----------
        points : (d,) ndarray
            points to be snapped to the image boundaries

        Returns
        -------
        bounded_points : (d,) ndarray
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

    def extract_patches(self, patch_centers, patch_size=(16, 16),
                        sample_offsets=None, as_single_array=False):
        r"""
        Extract a set of patches from an image. Given a set of patch centers and
        a patch size, patches are extracted from within the image, centred
        on the given coordinates. Sample offsets denote a set of offsets to
        extract from within a patch. This is very useful if you want to extract
        a dense set of features around a set of landmarks and simply sample the
        same grid of patches around the landmarks.

        If sample offsets are used, to access the offsets for each patch you
        need to slice the resulting list. So for 2 offsets, the first centers
        offset patches would be ``patches[:2]``.

        Currently only 2D images are supported.

        Parameters
        ----------
        patch_centers : :map:`PointCloud`
            The centers to extract patches around.
        patch_size : tuple or ndarray, optional
            The size of the patch to extract
        sample_offsets : :map:`PointCloud`, optional
            The offsets to sample from within a patch. So (0,0) is the centre
            of the patch (no offset) and (1, 0) would be sampling the patch
            from 1 pixel up the first axis away from the centre.
        as_single_array : (n_center * n_offset, self.shape...) ndarray, optional
            If ``True``, a single numpy array is returned containing each patch.
            If ``False``, a list of images is returned representing each patch.

        Returns
        -------
        patches : list or ndarray
            Returns the extracted patches. Returns a list if ``as_single_array``
            is ``True`` and an ndarray if ``as_single_array`` is ``False``.

        Raises
        ------
        ValueError
            If image is not 2D
        """
        if self.n_dims != 2:
            raise ValueError('Only two dimensional patch extraction is '
                             'currently supported.')

        if sample_offsets is None:
            sample_offsets_arr = np.zeros([1, 2], dtype=np.int64)
        else:
            sample_offsets_arr = np.require(sample_offsets.points,
                                            dtype=np.int64)

        single_array = extract_patches_cython(self.pixels,
                                              patch_centers.points,
                                              np.asarray(patch_size,
                                                         dtype=np.int64),
                                              sample_offsets_arr)

        if as_single_array:
            return single_array
        else:
            return [Image(p, copy=False) for p in single_array]

    def extract_patches_around_landmarks(
            self, group=None, label=None, patch_size=(16, 16),
            sample_offsets=None, as_single_array=False):
        r"""
        Extract patches around landmarks existing on this image. Provided the
        group label and optionally the landmark label extract a set of patches.

        See `extract_patches` for more information.

        Currently only 2D images are supported.

        Parameters
        ----------
        group : `str` or `None`, optional
            The landmark group to use as patch centres.
        label : `str` or `None`, optional
            The landmark label within the group to use as centres.
        patch_size : tuple or ndarray, optional
            The size of the patch to extract
        sample_offsets : :map:`PointCloud`, optional
            The offsets to sample from within a patch. So (0,0) is the centre
            of the patch (no offset) and (1, 0) would be sampling the patch
            from 1 pixel up the first axis away from the centre.
        as_single_array : (n_center * n_offset, self.shape...) ndarray, optional
            If ``True``, a single numpy array is returned containing each patch.
            If ``False``, a list of images is returned representing each patch.

        Returns
        -------
        patches : list or ndarray
            Returns the extracted patches. Returns a list if ``as_single_array``
            is ``True`` and an ndarray if ``as_single_array`` is ``False``.

        Raises
        ------
        ValueError
            If image is not 2D
        """
        return self.extract_patches(self.landmarks[group][label],
                                    patch_size=patch_size,
                                    sample_offsets=sample_offsets,
                                    as_single_array=as_single_array)

    def warp_to_mask(self, template_mask, transform, warp_landmarks=False,
             order=1, mode='constant', cval=0.):
        r"""
        Return a copy of this image warped into a different reference space.

        Note that warping into a mask is slower than warping into a full image.
        If you don't need a non-linear mask, consider warp_to_shape instead.

        Parameters
        ----------
        template_mask : :map:`BooleanImage`
            Defines the shape of the result, and what pixels should be
            sampled.

        transform : :map:`Transform`
            Transform **from the template space back to this image**.
            Defines, for each pixel location on the template, which pixel
            location should be sampled from on this image.

        warp_landmarks : `bool`, optional
            If `True`, warped_image will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        mode : `str`, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').

        cval : `float`, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        warped_image : :map:`MaskedImage`
            A copy of this image, warped.

        """
        if self.n_dims != transform.n_dims:
            raise ValueError(
                "Trying to warp a {}D image with a {}D transform "
                "(they must match)".format(self.n_dims, transform.n_dims))
        template_points = template_mask.true_indices()
        points_to_sample = transform.apply(template_points)
        # we want to sample each channel in turn, returning a vector of
        # sampled pixels. Store those in a (n_pixels, n_channels) array.
        sampled_pixel_values = scipy_interpolation(
            self.pixels, points_to_sample, order=order, mode=mode, cval=cval)
        # set any nan values to 0
        sampled_pixel_values[np.isnan(sampled_pixel_values)] = 0
        # build a warped version of the image
        warped_image = self._build_warped_to_mask(template_mask,
                                                  sampled_pixel_values)
        if warp_landmarks and self.has_landmarks:
            warped_image.landmarks = self.landmarks
            transform.pseudoinverse().apply_inplace(warped_image.landmarks)
        if hasattr(self, 'path'):
            warped_image.path = self.path
        return warped_image

    def _build_warped_to_mask(self, template_mask, sampled_pixel_values):
        r"""
        Builds the warped image from the template mask and
        sampled pixel values. Overridden for BooleanImage as we can't use
        the usual from_vector_inplace method. All other Image classes share
        the Image implementation.
        """
        warped_image = MaskedImage.blank(template_mask.shape,
                                         n_channels=self.n_channels,
                                         mask=template_mask)
        warped_image.from_vector_inplace(sampled_pixel_values.ravel())
        return warped_image

    def warp_to_shape(self, template_shape, transform, warp_landmarks=False,
                      order=1, mode='constant', cval=0.):
        """
        Return a copy of this image warped into a different reference space.

        Parameters
        ----------
        template_shape : tuple or ndarray
            Defines the shape of the result, and what pixel indices should be
            sampled (all of them).

        transform : :map:`Transform`
            Transform **from the template_shape space back to this image**.
            Defines, for each index on template_shape, which pixel location
            should be sampled from on this image.

        warp_landmarks : `bool`, optional
            If `True`, ``warped_image`` will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        mode : `str`, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').

        cval : `float`, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        warped_image : ``type(self)``
            A copy of this image, warped.

        """
        if (isinstance(transform, Affine) and order in range(4) and
            self.n_dims == 2):
            # skimage has an optimised Cython interpolation for 2D affine
            # warps
            sampled = cython_interpolation(self.pixels, template_shape,
                                           transform, order=order,
                                           mode=mode, cval=cval)
        else:
            template_points = indices_for_image_of_shape(template_shape)
            points_to_sample = transform.apply(template_points)
            # we want to sample each channel in turn, returning a vector of
            # sampled pixels. Store those in a (n_pixels, n_channels) array.
            sampled = scipy_interpolation(self.pixels, points_to_sample,
                                          order=order, mode=mode, cval=cval)
        # set any nan values to 0
        sampled[np.isnan(sampled)] = 0
        # build a warped version of the image
        warped_pixels = sampled.reshape(template_shape + (self.n_channels,))
        warped_image = Image(warped_pixels, copy=False)

        # warp landmarks if requested.
        if warp_landmarks and self.has_landmarks:
            warped_image.landmarks = self.landmarks
            transform.pseudoinverse().apply_inplace(warped_image.landmarks)
        if hasattr(self, 'path'):
            warped_image.path = self.path
        return warped_image

    def rescale(self, scale, round='ceil', order=1):
        r"""
        Return a copy of this image, rescaled by a given factor.
        Landmarks are rescaled appropriately.

        Parameters
        ----------
        scale : `float` or `tuple` of `floats`
            The scale factor. If a tuple, the scale to apply to each dimension.
            If a single float, the scale will be applied uniformly across
            each dimension.
        round: {'ceil', 'floor', 'round'}
            Rounding function to be applied to floating point shapes.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        Returns
        -------
        rescaled_image : ``type(self)``
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
        # use the scale factor to make the template mask bigger
        # while respecting the users rounding preference.
        template_shape = round_image_shape(transform.apply(self.shape),
                                           round)
        # due to image indexing, we can't just apply the pseudoinverse
        # transform to achieve the scaling we want though!
        # Consider a 3x rescale on a 2x4 image. Looking at each dimension:
        #    H 2 -> 6 so [0-1] -> [0-5] = 5/1 = 5x
        #    W 4 -> 12 [0-3] -> [0-11] = 11/3 = 3.67x
        # => need to make the correct scale per dimension!
        shape = np.array(self.shape, dtype=np.float)
        # scale factors = max_index_after / current_max_index
        # (note that max_index = length - 1, as 0 based)
        scale_factors = (scale * shape - 1) / (shape - 1)
        inverse_transform = NonUniformScale(scale_factors).pseudoinverse()
        # for rescaling we enforce that mode is nearest to avoid num. errors
        return self.warp_to_shape(template_shape, inverse_transform,
                                  warp_landmarks=True, order=order,
                                  mode='nearest')

    def rescale_to_diagonal(self, diagonal, round='ceil'):
        r"""
        Return a copy of this image, rescaled so that the it's diagonal is a
        new size.

        Parameters
        ----------
        diagonal: int
            The diagonal size of the new image

        Returns
        -------
        rescaled_image : type(self)
            A copy of this image, rescaled.

        """
        return self.rescale(diagonal / self.diagonal, round=round)

    def rescale_to_reference_shape(self, reference_shape, group=None,
                                   label=None, round='ceil', order=1):
        r"""
        Return a copy of this image, rescaled so that the scale of a
        particular group of landmarks matches the scale of the passed
        reference landmarks.

        Parameters
        ----------
        reference_shape: :map:`PointCloud`
            The reference shape to which the landmarks scale will be matched
            against.
        group : `str`, optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.
        label : `str`, optional
            The label of of the landmark manager that you wish to use. If
            `None` all landmarks in the group are used.
        round : {'ceil', 'floor', 'round'}, optional
            Rounding function to be applied to floating point shapes.
        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        Returns
        -------
        rescaled_image : ``type(self)``
            A copy of this image, rescaled.
        """
        pc = self.landmarks[group][label]
        scale = AlignmentUniformScale(pc, reference_shape).as_vector().copy()
        return self.rescale(scale, round=round, order=order)

    def rescale_landmarks_to_diagonal_range(self, diagonal_range, group=None,
                                            label=None, round='ceil', order=1):
        r"""
        Return a copy of this image, rescaled so that the diagonal_range of the
        bounding box containing its landmarks matches the specified
        diagonal_range range.

        Parameters
        ----------
        diagonal_range: ``(n_dims,)`` `ndarray`
            The diagonal_range range that we want the landmarks of the returned
            image to have.

        group : `str`, optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

        label: `str`, optional
            The label of of the landmark manager that you wish to use. If
            `None` all landmarks in the group are used.

        round: {'ceil', 'floor', 'round'}, optional
            Rounding function to be applied to floating point shapes.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        Returns
        -------
        rescaled_image : ``type(self)``
            A copy of this image, rescaled.
        """
        x, y = self.landmarks[group][label].range()
        scale = diagonal_range / np.sqrt(x ** 2 + y ** 2)
        return self.rescale(scale, round=round, order=order)

    def resize(self, shape, order=1):
        r"""
        Return a copy of this image, resized to a particular shape.
        All image information (landmarks, and mask in the case of
        :map:`MaskedImage`) is resized appropriately.

        Parameters
        ----------
        shape : tuple
            The new shape to resize to.

        order : `int`, optional
            The order of interpolation. The order has to be in the range 0-5:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic

        Returns
        -------
        resized_image : ``type(self)``
            A copy of this image, resized.

        Raises
        ------
        ValueError:
            If the number of dimensions of the new shape does not match
            the number of dimensions of the image.
        """
        shape = np.asarray(shape, dtype=np.float)
        if len(shape) != self.n_dims:
            raise ValueError(
                'Dimensions must match.'
                '{} dimensions provided, {} were expected.'.format(
                    shape.shape, self.n_dims))
        scales = shape / self.shape
        # Have to round the shape when scaling to deal with floating point
        # errors. For example, if we want (250, 250), we need to ensure that
        # we get (250, 250) even if the number we obtain is 250 to some
        # floating point inaccuracy.
        return self.rescale(scales, round='round', order=order)

    def rotate_ccw_about_centre(self, theta, degrees=True, cval=0):
        r"""
        Return a rotation of this image clockwise about it's centre.

        Parameters
        ----------
        theta : ``float``
            The angle of rotation about the origin
        degrees : ``bool``, optional
            If true theta is interpreted as a degree. If False, theta is
            interpreted as radians.
        cval : ``float``, optional
            The value to be set outside the rotated image boundaries
        """
        if self.n_dims != 2:
            raise ValueError('Image rotation is presently only supported on '
                             '2D images')
        # create a translation that moves the centre of the image to the origin
        t = Translation(self.centre)
        r = Rotation.from_2d_ccw_angle(theta, degrees=degrees)
        r_about_centre = t.pseudoinverse().compose_before(r).compose_before(t)
        return self.warp_to_shape(self.shape, r_about_centre.pseudoinverse(),
                                  warp_landmarks=True, cval=cval)

    def pyramid(self, n_levels=3, downscale=2):
        image = self
        yield image
        for _ in range(n_levels - 1):
            image = image.rescale(1.0 / downscale)
            yield image

    def gaussian_pyramid(self, n_levels=3, downscale=2, sigma=None):
        r"""
        Return the gaussian pyramid of this image. The first image of the
        pyramid will be the original, unmodified, image, and counts as level 1.

        Parameters
        ----------
        n_levels : int, optional
            Total number of levels in the pyramid, including the original
            unmodified image

        downscale : float, optional
            Downscale factor.

        sigma : float, optional
            Sigma for gaussian filter. Default is `downscale / 3.` which
            corresponds to a filter mask twice the size of the scale factor
            that covers more than 99% of the gaussian distribution.

        Returns
        -------
        image_pyramid:
            Generator yielding pyramid layers as menpo image objects.
        """
        from menpo.feature import gaussian_filter
        if sigma is None:
            sigma = downscale / 3.
        image = self
        yield image
        for level in range(n_levels - 1):
            image = gaussian_filter(image, sigma).rescale(1.0 / downscale)
            yield image

    def as_greyscale(self, mode='luminosity', channel=None):
        r"""
        Returns a greyscale version of the image. If the image does *not*
        represent a 2D RGB image, then the 'luminosity' mode will fail.

        Parameters
        ----------
        mode : {'average', 'luminosity', 'channel'}
            'luminosity' - Calculates the luminance using the CCIR 601 formula

                `Y' = 0.2989 R' + 0.5870 G' + 0.1140 B'`

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
        greyscale = self.copy()
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
        Return a PIL copy of the image. Depending on the image data type,
        different operations are performed:

        ========= ===========================================
        dtype     Processing
        ========= ===========================================
        uint8     No processing, directly converted to PIL
        bool      Scale by 255, convert to uint8
        float32   Scale by 255, convert to uint8
        float64   Scale by 255, convert to uint8
        OTHER     Raise ValueError
        ========= ===========================================

        Image must only have 1 or 3 channels and be 2 dimensional.
        Non `uint8` images must be in the rage ``[0, 1]`` to be converted.

        Returns
        -------
        pil_image : `PILImage`
            PIL copy of image

        Raises
        ------
        ValueError
            If image is not 2D and 1 channel or 3 channels.
        ValueError
            If pixels data type is not float32, float64, bool or uint8
        ValueError
            If pixels data type is float32 or float64 and the pixel
            range is outside of ``[0, 1]``
        """
        if self.n_dims != 2 or self.n_channels not in [1, 3]:
            raise ValueError('Can only convert greyscale or RGB 2D images. '
                             'Received a {} channel {}D image.'.format(
                self.n_channels, self.n_dims))

        # Slice off the channel for greyscale images
        pixels = self.pixels[..., 0] if self.n_channels == 1 else self.pixels
        if pixels.dtype in [np.float64, np.float32, np.bool]:  # Type check
            if np.any((self.pixels < 0) | (self.pixels > 1)):  # Range check
                raise ValueError('Pixel values are outside the range '
                                 '[0, 1] - ({}, {}).'.format(self.pixels.min(),
                                                             self.pixels.max()))
            else:
                pixels = (pixels * 255).astype(np.uint8)
        if pixels.dtype != np.uint8:
            raise ValueError('Unexpected data type - {}.'.format(pixels.dtype))
        return PILImage.fromarray(pixels)

    def __str__(self):
        return ('{} {}D Image with {} channel{}'.format(
            self._str_shape, self.n_dims, self.n_channels,
            's' * (self.n_channels > 1)))

    @property
    def has_landmarks_outside_bounds(self):
        """
        Indicates whether there are landmarks located outside the image bounds.

        :type: bool
        """
        if self.landmarks.has_landmarks:
            for l_group in self.landmarks:
                pc = self.landmarks[l_group].lms.points
                if np.any(np.logical_or(self.shape - pc < 1, pc < 0)):
                    return True
        return False

    def constrain_landmarks_to_bounds(self):
        r"""
        Move landmarks that are located outside the image bounds on the bounds.
        """
        if self.has_landmarks_outside_bounds:
            for l_group in self.landmarks:
                l = self.landmarks[l_group]
                for k in range(l.lms.points.shape[1]):
                    tmp = l.lms.points[:, k]
                    tmp[tmp < 0] = 0
                    tmp[tmp > self.shape[k] - 1] = self.shape[k] - 1
                    l.lms.points[:, k] = tmp
                self.landmarks[l_group] = l

    def normalize_std_inplace(self, mode='all', **kwargs):
        r"""
        Normalizes this image such that its pixel values have zero mean and
        unit variance.

        Parameters
        ----------

        mode : {'all', 'per_channel'}
            If 'all', the normalization is over all channels. If
            'per_channel', each channel individually is mean centred and
            normalized in variance.
        """
        self._normalize_inplace(np.std, mode=mode)

    def normalize_norm_inplace(self, mode='all', **kwargs):
        r"""
        Normalizes this image such that its pixel values have zero mean and
        its norm equals 1.

        Parameters
        ----------

        mode : {'all', 'per_channel'}
            If 'all', the normalization is over all channels. If
            'per_channel', each channel individually is mean centred and
            normalized in variance.
        """
        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        self._normalize_inplace(scale_func, mode=mode)

    def _normalize_inplace(self, scale_func, mode='all'):
        pixels = self.as_vector(keep_channels=True)
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
            self.from_vector_inplace(centered_pixels / scale_factor)


def round_image_shape(shape, round):
    if round not in ['ceil', 'round', 'floor']:
        raise ValueError('round must be either ceil, round or floor')
    # Ensure that the '+' operator means concatenate tuples
    return tuple(getattr(np, round)(shape).astype(np.int))


from .masked import MaskedImage
