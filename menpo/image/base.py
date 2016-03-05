from __future__ import division
from warnings import warn
from collections import Iterable

import numpy as np
import PIL.Image as PILImage

from menpo.compatibility import basestring
from menpo.base import Vectorizable, MenpoDeprecationWarning
from menpo.shape import PointCloud, bounding_box
from menpo.landmark import Landmarkable
from menpo.transform import (Translation, NonUniformScale,
                             AlignmentUniformScale, Affine, scale_about_centre,
                             rotate_ccw_about_centre, Rotation)
from menpo.visualize.base import ImageViewer, LandmarkableViewable, Viewable
from .interpolation import scipy_interpolation, cython_interpolation
from .patches import extract_patches, set_patches


# Cache the greyscale luminosity coefficients as they are invariant.
_greyscale_luminosity_coef = None


class ImageBoundaryError(ValueError):
    r"""
    Exception that is thrown when an attempt is made to crop an image beyond
    the edge of it's boundary.

    Parameters
    ----------
    requested_min : ``(d,)`` `ndarray`
        The per-dimension minimum index requested for the crop
    requested_max : ``(d,)`` `ndarray`
        The per-dimension maximum index requested for the crop
    snapped_min : ``(d,)`` `ndarray`
        The per-dimension minimum index that could be used if the crop was
        constrained to the image boundaries.
    requested_max : ``(d,)`` `ndarray`
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

    Parameters
    ----------
    shape : ``(n_dims, n_pixels)`` `ndarray`
        The shape of the image.

    Returns
    -------
    indices : `ndarray`
        The indices of all the pixels in the image.
    """
    return np.indices(shape).reshape([len(shape), -1]).T


def normalise_pixels_range(pixels, error_on_unknown_type=True):
    r"""
    Normalise the given pixels to the Menpo valid floating point range, [0, 1].
    This is a single place to handle normalising pixels ranges. At the moment
    the supported types are uint8 and uint16.

    Parameters
    ----------
    pixels : `ndarray`
        The pixels to normalise in the floating point range.
    error_on_unknown_type : `bool`, optional
        If ``True``, this method throws a ``ValueError`` if the given pixels
        array is an unknown type. If ``False``, this method performs no
        operation.

    Returns
    -------
    normalised_pixels : `ndarray`
        The normalised pixels in the range [0, 1].

    Raises
    ------
    ValueError
        If ``pixels`` is an unknown type and ``error_on_unknown_type==True``
    """
    dtype = pixels.dtype
    if dtype == np.uint8:
        max_range = 255.0
    elif dtype == np.uint16:
        max_range = 65535.0
    else:
        if error_on_unknown_type:
            raise ValueError('Unexpected dtype ({}) - normalisation range '
                             'is unknown'.format(dtype))
        else:
            # Do nothing
            return pixels
    # This multiplication is quite a bit faster than just dividing - will
    # automatically cast it up to float64
    return pixels * (1.0 / max_range)


def denormalise_pixels_range(pixels, out_dtype):
    """
    Denormalise the given pixels array into the range of the given out dtype.
    If the given pixels are floating point or boolean then the values
    are scaled appropriately and cast to the output dtype. If the pixels
    are already the correct dtype they are immediately returned.
    Floating point pixels must be in the range [0, 1].
    Currently uint8 and uint16 output dtypes are supported.

    Parameters
    ----------
    pixels : `ndarray`
        The pixels to denormalise.
    out_dtype : `np.dtype`
        The numpy data type to output and scale the values into.

    Returns
    -------
    out_pixels : `ndarray`
        Will be in the correct range and will have type ``out_dtype``.

    Raises
    ------
    ValueError
        Pixels are floating point and range outside [0, 1]
    ValueError
        Input pixels dtype not in the set {float32, float64, bool}.
    ValueError
        Output dtype not in the set {uint8, uint16}
    """
    in_dtype = pixels.dtype
    if in_dtype == out_dtype:
        return pixels

    if np.issubclass_(in_dtype.type, np.floating) or in_dtype == np.float:
        if np.issubclass_(out_dtype, np.floating) or out_dtype == np.float:
            return pixels.astype(out_dtype)
        else:
            p_min = pixels.min()
            p_max = pixels.max()
            if p_min < 0.0 or p_max > 1.0:
                raise ValueError('Unexpected input range [{}, {}] - pixels '
                                 'must be in the range [0, 1]'.format(p_min,
                                                                      p_max))
    elif in_dtype != np.bool:
        raise ValueError('Unexpected input dtype ({}) - only float32, float64 '
                         'and bool supported'.format(in_dtype))

    if out_dtype == np.uint8:
        max_range = 255.0
    elif out_dtype == np.uint16:
        max_range = 65535.0
    else:
        raise ValueError('Unexpected output dtype ({}) - normalisation range '
                         'is unknown'.format(out_dtype))

    return (pixels * max_range).astype(out_dtype)


def channels_to_back(pixels):
    r"""
    Roll the channels from the front to the back for an image. If the image
    that is passed is already a numpy array, then that is also fine.

    Always returns a numpy array because our :map:`Image` containers do not
    support channels at the back.

    Parameters
    ----------
    image : `ndarray`
        The pixels or image to roll the channel back for.

    Returns
    -------
    rolled_pixels : `ndarray`
        The numpy array of pixels with the channels on the last axis.
    """
    return np.require(np.rollaxis(pixels, 0, pixels.ndim), dtype=pixels.dtype,
                      requirements=['C'])


def channels_to_front(pixels):
    r"""
    Convert the given pixels array (channels assumed to be at the last axis
    as is common in other imaging packages) into a numpy array.

    Parameters
    ----------
    pixels : ``(H, W, C)`` `buffer`
        The pixels to convert to the Menpo channels at axis 0.

    Returns
    -------
    pixels : ``(C, H, W)`` `ndarray`
        Numpy array, channels as axis 0.
    """
    if not isinstance(pixels, np.ndarray):
        pixels = np.array(pixels)
    # Channels to axis 0
    if pixels.ndim == 3:
        pixels = np.require(np.rollaxis(pixels, -1), dtype=pixels.dtype,
                            requirements=['C'])
    return pixels


class Image(Vectorizable, Landmarkable, Viewable, LandmarkableViewable):
    r"""
    An n-dimensional image.

    Images are n-dimensional homogeneous regular arrays of data. Each
    spatially distinct location in the array is referred to as a `pixel`.
    At a pixel, ``k`` distinct pieces of information can be stored. Each
    datum at a pixel is refereed to as being in a `channel`. All pixels in
    the image have the same number of channels, and all channels have the
    same data-type (`float64`).

    Parameters
    ----------
    image_data : ``(C, M, N ..., Q)`` `ndarray`
        Array representing the image pixels, with the first axis being
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
            # Ensures that the data STAYS C-contiguous
            image_data = image_data.reshape((1,) + image_data.shape)

        if image_data.ndim < 2:
            raise ValueError(
                "Pixel array has to be 2D (implicitly 1 channel, "
                "2D shape) or 3D+ (n_channels, 2D+ shape) "
                " - a {}D array "
                "was provided".format(image_data.ndim))
        self.pixels = image_data

    @classmethod
    def init_blank(cls, shape, n_channels=1, fill=0, dtype=np.float):
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
            pixels = np.zeros((n_channels,) + shape, dtype=dtype)
        else:
            pixels = np.ones((n_channels,) + shape, dtype=dtype) * fill
        # We know there is no need to copy...
        return cls(pixels, copy=False)

    @classmethod
    def init_from_rolled_channels(cls, pixels):
        r"""
        Create an Image from a set of pixels where the channels axis is on
        the last axis (the back). This is common in other frameworks, and
        therefore this method provides a convenient means of creating a menpo
        Image from such data. Note that a copy is always created due to the
        need to rearrange the data.

        Parameters
        ----------
        pixels : ``(M, N ..., Q, C)`` `ndarray`
            Array representing the image pixels, with the last axis being
            channels.

        Returns
        -------
        image : :map:`Image`
            A new image from the given pixels, with the FIRST axis as the
            channels.
        """
        return cls(np.rollaxis(pixels, -1))

    def as_masked(self, mask=None, copy=True):
        r"""
        Return a copy of this image with an attached mask behavior.

        A custom mask may be provided, or ``None``. See the :map:`MaskedImage`
        constructor for details of how the kwargs will be handled.

        Parameters
        ----------
        mask : ``(self.shape)`` `ndarray` or :map:`BooleanImage`
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
        from menpo.image import MaskedImage
        img = MaskedImage(self.pixels, mask=mask, copy=copy)
        img.landmarks = self.landmarks
        if hasattr(self, 'path'):
            img.path = self.path
        return img

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
        Total number of pixels in the image ``(prod(shape),)``

        :type: `int`
        """
        return self.pixels[0, ...].size

    @property
    def n_elements(self):
        r"""
        Total number of data points in the image
        ``(prod(shape), n_channels)``

        :type: `int`
        """
        return self.pixels.size

    @property
    def n_channels(self):
        """
        The number of channels on each pixel in the image.

        :type: `int`
        """
        return self.pixels.shape[0]

    @property
    def width(self):
        r"""
        The width of the image.

        This is the width according to image semantics, and is thus the size
        of the **last** dimension.

        :type: `int`
        """
        return self.pixels.shape[-1]

    @property
    def height(self):
        r"""
        The height of the image.

        This is the height according to image semantics, and is thus the size
        of the **second to last** dimension.

        :type: `int`
        """
        return self.pixels.shape[-2]

    @property
    def shape(self):
        r"""
        The shape of the image
        (with ``n_channel`` values at each point).

        :type: `tuple`
        """
        return self.pixels.shape[1:]

    def diagonal(self):
        r"""
        The diagonal size of this image

        :type: `float`
        """
        return np.sqrt(np.sum(np.array(self.shape) ** 2))

    def centre(self):
        r"""
        The geometric centre of the Image - the subpixel that is in the
        middle.

        Useful for aligning shapes and images.

        :type: (``n_dims``,) `ndarray`
        """
        # noinspection PyUnresolvedReferences
        return np.array(self.shape, dtype=np.double) / 2

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
            `False`    ``(n_channels * n_pixels,)``
            `True`     ``(n_channels, n_pixels)``
            ========== =============================

        Returns
        -------
        vec : (See ``keep_channels`` above) `ndarray`
            Flattened representation of this image, containing all pixel
            and channel information.
        """
        if keep_channels:
            return self.pixels.reshape([self.n_channels, -1])
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
        vector : ``(n_parameters,)`` `ndarray`
            A flattened vector of all pixels and channels of an image.
        n_channels : `int`, optional
            If given, will assume that vector is the same shape as this image,
            but with a possibly different number of channels.
        copy : `bool`, optional
            If ``False``, the vector will not be copied in creating the new
            image.

        Returns
        -------
        image : :map:`Image`
            New image of same shape as this image and the number of
            specified channels.

        Raises
        ------
        Warning
            If the ``copy=False`` flag cannot be honored
        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels is None else n_channels
        image_data = vector.reshape((n_channels,) + self.shape)
        new_image = Image(image_data, copy=copy)
        new_image.landmarks = self.landmarks
        return new_image

    def _from_vector_inplace(self, vector, copy=True):
        r"""
        Takes a flattened vector and update this image by
        reshaping the vector to the correct dimensions.

        Parameters
        ----------
        vector : ``(n_pixels,)`` `bool ndarray`
            A vector vector of all the pixels of a :map:`BooleanImage`.
        copy: `bool`, optional
            If ``False``, the vector will be set as the pixels. If ``True``, a
            copy of the vector is taken.

        Raises
        ------
        Warning
            If ``copy=False`` flag cannot be honored

        Note
        ----
        For :map:`BooleanImage` this is rebuilding a boolean image **itself**
        from boolean values. The mask is in no way interpreted in performing
        the operation, in contrast to :map:`MaskedImage`, where only the masked
        region is used in :meth:`from_vector_inplace` and :meth:`as_vector`.
        """
        image_data = vector.reshape(self.pixels.shape)
        if not copy:
            if not image_data.flags.c_contiguous:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
                image_data = np.array(image_data, copy=True, order='C',
                                      dtype=image_data.dtype)
        else:
            image_data = np.array(image_data, copy=True, order='C',
                                  dtype=image_data.dtype)
        self.pixels = image_data

    def extract_channels(self, channels):
        r"""
        A copy of this image with only the specified channels.

        Parameters
        ----------
        channels : `int` or `[int]`
            The channel index or `list` of channel indices to retain.

        Returns
        -------
        image : `type(self)`
            A copy of this image with only the channels requested.
        """
        copy = self.copy()
        if not isinstance(channels, list):
            channels = [channels]  # ensure we don't remove the channel axis
        copy.pixels = self.pixels[channels]
        return copy

    def as_histogram(self, keep_channels=True, bins='unique'):
        r"""
        Histogram binning of the values of this image.

        Parameters
        ----------
        keep_channels : `bool`, optional
            If set to ``False``, it returns a single histogram for all the
            channels of the image. If set to ``True``, it returns a `list` of
            histograms, one for each channel.
        bins : ``{unique}``, positive `int` or sequence of scalars, optional
            If set equal to ``'unique'``, the bins of the histograms are centred
            on the unique values of each channel. If set equal to a positive
            `int`, then this is the number of bins. If set equal to a
            sequence of scalars, these will be used as bins centres.

        Returns
        -------
        hist : `ndarray` or `list` with ``n_channels`` `ndarrays` inside
            The histogram(s). If ``keep_channels=False``, then hist is an
            `ndarray`. If ``keep_channels=True``, then hist is a `list` with
            ``len(hist)=n_channels``.
        bin_edges : `ndarray` or `list` with `n_channels` `ndarrays` inside
            An array or a list of arrays corresponding to the above histograms
            that store the bins' edges.

        Raises
        ------
        ValueError
            Bins can be either 'unique', positive int or a sequence of scalars.

        Examples
        --------
        Visualizing the histogram when a list of array bin edges is provided:

        >>> hist, bin_edges = image.as_histogram()
        >>> for k in range(len(hist)):
        >>>     plt.subplot(1,len(hist),k)
        >>>     width = 0.7 * (bin_edges[k][1] - bin_edges[k][0])
        >>>     centre = (bin_edges[k][:-1] + bin_edges[k][1:]) / 2
        >>>     plt.bar(centre, hist[k], align='center', width=width)
        """
        # parse options
        if isinstance(bins, basestring):
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
        if len(vec.shape) == 1 or vec.shape[0] == 1:
            if bins == 0:
                bins = np.unique(vec)
            hist, bin_edges = np.histogram(vec, bins=bins)
        else:
            hist = []
            bin_edges = []
            num_bins = bins
            for ch in range(vec.shape[0]):
                if bins == 0:
                    num_bins = np.unique(vec[ch, :])
                h_tmp, c_tmp = np.histogram(vec[ch, :], bins=num_bins)
                hist.append(h_tmp)
                bin_edges.append(c_tmp)
        return hist, bin_edges

    def _view_2d(self, figure_id=None, new_figure=False, channels=None,
                 interpolation='bilinear', cmap_name=None, alpha=1.,
                 render_axes=False, axes_font_name='sans-serif',
                 axes_font_size=10, axes_font_style='normal',
                 axes_font_weight='normal', axes_x_limits=None,
                 axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
                 figure_size=(10, 8)):
        r"""
        View the image using the default image viewer. This method will appear 
        on the Image as ``view`` if the Image is 2D.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36,
                hanning, hamming, hermite, kaiser, quadric, catrom, gaussian,
                bessel, mitchell, sinc, lanczos}
        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::
            
                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::
            
                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.

        Returns
        -------
        viewer : `ImageViewer`
            The image viewing object.
        """
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           self.pixels, channels=channels).render(
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualizes the image object using an interactive widget. Currently
        only supports the rendering of 2D images.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the images will have the form of
            plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_images
            visualize_images(self, figure_size=figure_size, style=style,
                             browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def _view_landmarks_2d(self, channels=None, group=None,
                           with_labels=None, without_labels=None,
                           figure_id=None, new_figure=False,
                           interpolation='bilinear', cmap_name=None, alpha=1.,
                           render_lines=True, line_colour=None, line_style='-',
                           line_width=1, render_markers=True, marker_style='o',
                           marker_size=5, marker_face_colour=None,
                           marker_edge_colour=None, marker_edge_width=1.,
                           render_numbering=False,
                           numbers_horizontal_align='center',
                           numbers_vertical_align='bottom',
                           numbers_font_name='sans-serif', numbers_font_size=10,
                           numbers_font_style='normal',
                           numbers_font_weight='normal',
                           numbers_font_colour='k', render_legend=False,
                           legend_title='', legend_font_name='sans-serif',
                           legend_font_style='normal', legend_font_size=10,
                           legend_font_weight='normal',
                           legend_marker_scale=None,
                           legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                           legend_border_axes_pad=None, legend_n_columns=1,
                           legend_horizontal_spacing=None,
                           legend_vertical_spacing=None, legend_border=True,
                           legend_border_padding=None, legend_shadow=False,
                           legend_rounded_corners=False, render_axes=False,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           axes_x_limits=None, axes_y_limits=None,
                           axes_x_ticks=None, axes_y_ticks=None,
                           figure_size=(10, 8)):
        """
        Visualize the landmarks. This method will appear on the Image as
        ``view_landmarks`` if the Image is 2D.

        Parameters
        ----------
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        group : `str` or``None`` optional
            The landmark group to be visualized. If ``None`` and there are more
            than one landmark groups, an error is raised.
        with_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated. Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{-, --, -., :}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::
            
                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::
            
                {r, g, b, c, m, k, w}
                or 
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : ``{center, right, left}``, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : See Below, optional
            The font of the numbers. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : See Below, optional
            The font weight of the numbers.
            Example options ::
            
                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend. Example options ::
            
                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : ``{normal, italic, oblique}``, optional
            The font style of the legend.
        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : See Below, optional
            The font weight of the legend.
            Example options ::
            
                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::
            
                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        ValueError
            If the landmark manager doesn't contain the provided group label.
        """
        from menpo.visualize import view_image_landmarks
        return view_image_landmarks(
            self, channels, False, group, with_labels, without_labels,
            figure_id, new_figure, interpolation, cmap_name, alpha,
            render_lines, line_colour, line_style, line_width,
            render_markers, marker_style, marker_size, marker_face_colour,
            marker_edge_colour, marker_edge_width, render_numbering,
            numbers_horizontal_align, numbers_vertical_align,
            numbers_font_name, numbers_font_size, numbers_font_style,
            numbers_font_weight, numbers_font_colour, render_legend,
            legend_title, legend_font_name, legend_font_style,
            legend_font_size, legend_font_weight, legend_marker_scale,
            legend_location, legend_bbox_to_anchor, legend_border_axes_pad,
            legend_n_columns, legend_horizontal_spacing,
            legend_vertical_spacing, legend_border, legend_border_padding,
            legend_shadow, legend_rounded_corners, render_axes, axes_font_name,
            axes_font_size, axes_font_style, axes_font_weight, axes_x_limits,
            axes_y_limits, axes_x_ticks, axes_y_ticks, figure_size)

    def crop(self, min_indices, max_indices, constrain_to_boundary=False,
             return_transform=False):
        r"""
        Return a cropped copy of this image using the given minimum and
        maximum indices. Landmarks are correctly adjusted so they maintain
        their position relative to the newly cropped image.

        Parameters
        ----------
        min_indices : ``(n_dims,)`` `ndarray`
            The minimum index over each dimension.
        max_indices : ``(n_dims,)`` `ndarray`
            The maximum index over each dimension.
        constrain_to_boundary : `bool`, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map:`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the cropping is also returned.

        Returns
        -------
        cropped_image : `type(self)`
            A new instance of self, but cropped.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ValueError
            ``min_indices`` and ``max_indices`` both have to be of length
            ``n_dims``. All ``max_indices`` must be greater than
            ``min_indices``.
        ImageBoundaryError
            Raised if ``constrain_to_boundary=False``, and an attempt is made
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
        all_max_bounded = np.all(min_bounded == min_indices)
        all_min_bounded = np.all(max_bounded == max_indices)
        if not (constrain_to_boundary or all_max_bounded or all_min_bounded):
            # points have been constrained and the user didn't want this -
            raise ImageBoundaryError(min_indices, max_indices,
                                     min_bounded, max_bounded)

        new_shape = max_bounded - min_bounded
        return self.warp_to_shape(new_shape, Translation(min_bounded), order=0,
                                  warp_landmarks=True,
                                  return_transform=return_transform)

    def crop_to_pointcloud(self, pointcloud, boundary=0,
                           constrain_to_boundary=True,
                           return_transform=False):
        r"""
        Return a copy of this image cropped so that it is bounded around a
        pointcloud with an optional ``n_pixel`` boundary.

        Parameters
        ----------
        pointcloud : :map:`PointCloud`
            The pointcloud to crop around.
        boundary : `int`, optional
            An extra padding to be added all around the landmarks bounds.
        constrain_to_boundary : `bool`, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the cropping is also returned.

        Returns
        -------
        image : :map:`Image`
            A copy of this image cropped to the bounds of the pointcloud.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ImageBoundaryError
            Raised if ``constrain_to_boundary=False``, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        min_indices, max_indices = pointcloud.bounds(boundary=boundary)
        return self.crop(min_indices, max_indices,
                         constrain_to_boundary=constrain_to_boundary,
                         return_transform=return_transform)

    def crop_to_landmarks(self, group=None, boundary=0,
                          constrain_to_boundary=True,
                          return_transform=False):
        r"""
        Return a copy of this image cropped so that it is bounded around a set
        of landmarks with an optional ``n_pixel`` boundary

        Parameters
        ----------
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``
            and if there is only one set of landmarks, this set will be used.
        boundary : `int`, optional
            An extra padding to be added all around the landmarks bounds.
        constrain_to_boundary : `bool`, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the cropping is also returned.

        Returns
        -------
        image : :map:`Image`
            A copy of this image cropped to its landmarks.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ImageBoundaryError
            Raised if ``constrain_to_boundary=False``, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        pc = self.landmarks[group].lms
        return self.crop_to_pointcloud(
            pc, boundary=boundary, constrain_to_boundary=constrain_to_boundary,
            return_transform=return_transform)

    def crop_to_pointcloud_proportion(self, pointcloud, boundary_proportion,
                                      minimum=True,
                                      constrain_to_boundary=True,
                                      return_transform=False):
        r"""
        Return a copy of this image cropped so that it is bounded around a
        pointcloud with an optional ``n_pixel`` boundary.

        Parameters
        ----------
        boundary_proportion : `float`
            Additional padding to be added all around the landmarks
            bounds defined as a proportion of the landmarks range. See
            the minimum parameter for a definition of how the range is
            calculated.
        pointcloud : :map:`PointCloud`
            The pointcloud to crop around.
        minimum : `bool`, optional
            If ``True`` the specified proportion is relative to the minimum
            value of the pointclouds' per-dimension range; if ``False`` w.r.t.
            the maximum value of the pointclouds' per-dimension range.
        constrain_to_boundary : `bool`, optional
            If ``True``, the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map:`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the cropping is also returned.

        Returns
        -------
        image : :map:`Image`
            A copy of this image cropped to the border proportional to
            the pointcloud spread or range.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ImageBoundaryError
            Raised if ``constrain_to_boundary=False``, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        if minimum:
            boundary = boundary_proportion * np.min(pointcloud.range())
        else:
            boundary = boundary_proportion * np.max(pointcloud.range())
        return self.crop_to_pointcloud(
            pointcloud, boundary=boundary,
            constrain_to_boundary=constrain_to_boundary,
            return_transform=return_transform)

    def crop_to_landmarks_proportion(self, boundary_proportion,
                                     group=None, minimum=True,
                                     constrain_to_boundary=True,
                                     return_transform=False):
        r"""
        Crop this image to be bounded around a set of landmarks with a
        border proportional to the landmark spread or range.

        Parameters
        ----------
        boundary_proportion : `float`
            Additional padding to be added all around the landmarks
            bounds defined as a proportion of the landmarks range. See
            the minimum parameter for a definition of how the range is
            calculated.
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``
            and if there is only one set of landmarks, this set will be used.
        minimum : `bool`, optional
            If ``True`` the specified proportion is relative to the minimum
            value of the landmarks' per-dimension range; if ``False`` w.r.t. the
            maximum value of the landmarks' per-dimension range.
        constrain_to_boundary : `bool`, optional
            If ``True``, the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map:`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the cropping is also returned.

        Returns
        -------
        image : :map:`Image`
            This image, cropped to its landmarks with a border proportional to
            the landmark spread or range.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ImageBoundaryError
            Raised if ``constrain_to_boundary=False``, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        pc = self.landmarks[group].lms
        return self.crop_to_pointcloud_proportion(
            pc, boundary_proportion, minimum=minimum,
            constrain_to_boundary=constrain_to_boundary,
            return_transform=return_transform)

    def _propagate_crop_to_inplace(self, cropped):
        # helper method that sets self's state to the result of a crop call.
        # only needed for the deprecation period of the inplace crop methods.
        self.pixels = cropped.pixels
        self.landmarks = cropped.landmarks
        if hasattr(self, 'mask'):
            self.mask = cropped.mask
        return self

    def constrain_points_to_bounds(self, points):
        r"""
        Constrains the points provided to be within the bounds of this image.

        Parameters
        ----------
        points : ``(d,)`` `ndarray`
            Points to be snapped to the image boundaries.

        Returns
        -------
        bounded_points : ``(d,)`` `ndarray`
            Points snapped to not stray outside the image edges.
        """
        bounded_points = points.copy()
        # check we don't stray under any edges
        bounded_points[bounded_points < 0] = 0
        # check we don't stray over any edges
        shape = np.array(self.shape)
        over_image = (shape - bounded_points) < 0
        bounded_points[over_image] = shape[over_image]
        return bounded_points

    def extract_patches(self, patch_centers, patch_shape=(16, 16),
                        sample_offsets=None, as_single_array=True):
        r"""
        Extract a set of patches from an image. Given a set of patch centers
        and a patch size, patches are extracted from within the image, centred
        on the given coordinates. Sample offsets denote a set of offsets to
        extract from within a patch. This is very useful if you want to extract
        a dense set of features around a set of landmarks and simply sample the
        same grid of patches around the landmarks.

        If sample offsets are used, to access the offsets for each patch you
        need to slice the resulting `list`. So for 2 offsets, the first centers
        offset patches would be ``patches[:2]``.

        Currently only 2D images are supported.

        Parameters
        ----------
        patch_centers : :map:`PointCloud`
            The centers to extract patches around.
        patch_shape : ``(1, n_dims)`` `tuple` or `ndarray`, optional
            The size of the patch to extract
        sample_offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
            The offsets to sample from within a patch. So ``(0, 0)`` is the
            centre of the patch (no offset) and ``(1, 0)`` would be sampling the
            patch from 1 pixel up the first axis away from the centre.
            If ``None``, then no offsets are applied.
        as_single_array : `bool`, optional
            If ``True``, an ``(n_center, n_offset, n_channels, patch_shape)``
            `ndarray`, thus a single numpy array is returned containing each
            patch. If ``False``, a `list` of ``n_center * n_offset``
            :map:`Image` objects is returned representing each patch.

        Returns
        -------
        patches : `list` or `ndarray`
            Returns the extracted patches. Returns a list if
            ``as_single_array=True`` and an `ndarray` if
            ``as_single_array=False``.

        Raises
        ------
        ValueError
            If image is not 2D
        """
        if self.n_dims != 2:
            raise ValueError('Only two dimensional patch extraction is '
                             'currently supported.')

        if sample_offsets is None:
            sample_offsets = np.zeros([1, 2], dtype=np.intp)
        else:
            sample_offsets = np.require(sample_offsets, dtype=np.intp)

        patch_centers = np.require(patch_centers.points, dtype=np.float,
                                   requirements=['C'])
        single_array = extract_patches(self.pixels, patch_centers,
                                       np.asarray(patch_shape, dtype=np.intp),
                                       sample_offsets)

        if as_single_array:
            return single_array
        else:
            return [Image(o, copy=False) for p in single_array for o in p]

    def extract_patches_around_landmarks(
            self, group=None, patch_shape=(16, 16),
            sample_offsets=None, as_single_array=True):
        r"""
        Extract patches around landmarks existing on this image. Provided the
        group label and optionally the landmark label extract a set of patches.

        See `extract_patches` for more information.

        Currently only 2D images are supported.

        Parameters
        ----------
        group : `str` or ``None``, optional
            The landmark group to use as patch centres.
        patch_shape : `tuple` or `ndarray`, optional
            The size of the patch to extract
        sample_offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
            The offsets to sample from within a patch. So ``(0, 0)`` is the
            centre of the patch (no offset) and ``(1, 0)`` would be sampling the
            patch from 1 pixel up the first axis away from the centre.
            If ``None``, then no offsets are applied.
        as_single_array : `bool`, optional
            If ``True``, an ``(n_center, n_offset, n_channels, patch_shape)``
            `ndarray`, thus a single numpy array is returned containing each
            patch. If ``False``, a `list` of ``n_center * n_offset``
            :map:`Image` objects is returned representing each patch.

        Returns
        -------
        patches : `list` or `ndarray`
            Returns the extracted patches. Returns a list if
            ``as_single_array=True`` and an `ndarray` if
            ``as_single_array=False``.

        Raises
        ------
        ValueError
            If image is not 2D
        """
        return self.extract_patches(self.landmarks[group].lms,
                                    patch_shape=patch_shape,
                                    sample_offsets=sample_offsets,
                                    as_single_array=as_single_array)

    def set_patches(self, patches, patch_centers, offset=None,
                    offset_index=None):
        r"""
        Set the values of a group of patches into the correct regions of
        **this** image. Given an array of patches and a set of patch centers,
        the patches' values are copied in the regions of the image that are
        centred on the coordinates of the given centers.

        The patches argument can have any of the two formats that are returned
        from the `extract_patches()` and `extract_patches_around_landmarks()`
        methods. Specifically it can be:

            1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
            2. `list` of ``n_center * n_offset`` :map:`Image` objects

        Currently only 2D images are supported.

        Parameters
        ----------
        patches : `ndarray` or `list`
            The values of the patches. It can have any of the two formats that
            are returned from the `extract_patches()` and
            `extract_patches_around_landmarks()` methods. Specifically, it can
            either be an ``(n_center, n_offset, self.n_channels, patch_shape)``
            `ndarray` or a `list` of ``n_center * n_offset`` :map:`Image`
            objects.
        patch_centers : :map:`PointCloud`
            The centers to set the patches around.
        offset : `list` or `tuple` or ``(1, 2)`` `ndarray` or ``None``, optional
            The offset to apply on the patch centers within the image.
            If ``None``, then ``(0, 0)`` is used.
        offset_index : `int` or ``None``, optional
            The offset index within the provided `patches` argument, thus the
            index of the second dimension from which to sample. If ``None``,
            then ``0`` is used.

        Raises
        ------
        ValueError
            If image is not 2D
        ValueError
            If offset does not have shape (1, 2)
        """
        # parse arguments
        if self.n_dims != 2:
            raise ValueError('Only two dimensional patch insertion is '
                             'currently supported.')
        if offset is None:
            offset = np.zeros([1, 2], dtype=np.intp)
        elif isinstance(offset, tuple) or isinstance(offset, list):
            offset = np.asarray([offset])
        offset = np.require(offset, dtype=np.intp)
        if not offset.shape == (1, 2):
            raise ValueError('The offset must be a tuple, a list or a '
                             'numpy.array with shape (1, 2).')
        if offset_index is None:
            offset_index = 0

        # if patches is a list, convert it to array
        if isinstance(patches, list):
            patches = _convert_patches_list_to_single_array(
                patches, patch_centers.n_points)

        # set patches
        set_patches(patches, self.pixels, patch_centers.points, offset,
                    offset_index)

    def set_patches_around_landmarks(self, patches, group=None,
                                     offset=None, offset_index=None):
        r"""
        Set the values of a group of patches around the landmarks existing in
        **this** image. Given an array of patches, a group and a label, the
        patches' values are copied in the regions of the image that are
        centred on the coordinates of corresponding landmarks.

        The patches argument can have any of the two formats that are returned
        from the `extract_patches()` and `extract_patches_around_landmarks()`
        methods. Specifically it can be:

            1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
            2. `list` of ``n_center * n_offset`` :map:`Image` objects

        Currently only 2D images are supported.

        Parameters
        ----------
        patches : `ndarray` or `list`
            The values of the patches. It can have any of the two formats that
            are returned from the `extract_patches()` and
            `extract_patches_around_landmarks()` methods. Specifically, it can
            either be an ``(n_center, n_offset, self.n_channels, patch_shape)``
            `ndarray` or a `list` of ``n_center * n_offset`` :map:`Image`
            objects.
        group : `str` or ``None`` optional
            The landmark group to use as patch centres.
        offset : `list` or `tuple` or ``(1, 2)`` `ndarray` or ``None``, optional
            The offset to apply on the patch centers within the image.
            If ``None``, then ``(0, 0)`` is used.
        offset_index : `int` or ``None``, optional
            The offset index within the provided `patches` argument, thus the
            index of the second dimension from which to sample. If ``None``,
            then ``0`` is used.

        Raises
        ------
        ValueError
            If image is not 2D
        ValueError
            If offset does not have shape (1, 2)
        """
        return self.set_patches(patches, self.landmarks[group].lms,
                                offset=offset, offset_index=offset_index)

    def warp_to_mask(self, template_mask, transform, warp_landmarks=True,
                     order=1, mode='constant', cval=0.0, batch_size=None,
                     return_transform=False):
        r"""
        Return a copy of this image warped into a different reference space.

        Note that warping into a mask is slower than warping into a full image.
        If you don't need a non-linear mask, consider :meth:``warp_to_shape``
        instead.

        Parameters
        ----------
        template_mask : :map:`BooleanImage`
            Defines the shape of the result, and what pixels should be sampled.
        transform : :map:`Transform`
            Transform **from the template space back to this image**.
            Defines, for each pixel location on the template, which pixel
            location should be sampled from on this image.
        warp_landmarks : `bool`, optional
            If ``True``, result will have the same landmark dictionary
            as ``self``, but with each landmark updated to the warped position.
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5]

            ========= =====================
            Order     Interpolation
            ========= =====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= =====================

        mode : ``{constant, nearest, reflect, wrap}``, optional
            Points outside the boundaries of the input are filled according
            to the given mode.
        cval : `float`, optional
            Used in conjunction with mode ``constant``, the value outside
            the image boundaries.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this
            value can cause warping to become much slower, particular for
            cached warps such as Piecewise Affine. This size indicates
            how many points in the image should be warped at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are warped at once.
        return_transform : `bool`, optional
            This argument is for internal use only. If ``True``, then the
            :map:`Transform` object is also returned.

        Returns
        -------
        warped_image : :map:`MaskedImage`
            A copy of this image, warped.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        if self.n_dims != transform.n_dims:
            raise ValueError(
                "Trying to warp a {}D image with a {}D transform "
                "(they must match)".format(self.n_dims, transform.n_dims))
        template_points = template_mask.true_indices()
        points_to_sample = transform.apply(template_points,
                                           batch_size=batch_size)
        sampled = self.sample(points_to_sample,
                              order=order, mode=mode, cval=cval)

        # set any nan values to 0
        sampled[np.isnan(sampled)] = 0
        # build a warped version of the image
        warped_image = self._build_warp_to_mask(template_mask, sampled)
        if warp_landmarks and self.has_landmarks:
            warped_image.landmarks = self.landmarks
            transform.pseudoinverse()._apply_inplace(warped_image.landmarks)
        if hasattr(self, 'path'):
            warped_image.path = self.path
        # optionally return the transform
        if return_transform:
            return warped_image, transform
        else:
            return warped_image

    def _build_warp_to_mask(self, template_mask, sampled_pixel_values):
        r"""
        Builds the warped image from the template mask and sampled pixel values.
        Overridden for :map:`BooleanImage` as we can't use the usual
        :meth:`from_vector_inplace` method. All other :map:`Image` classes
        share the :map:`Image` implementation.

        Parameters
        ----------
        template_mask : :map:`BooleanImage` or 2D `bool ndarray`
            Mask for warping.
        sampled_pixel_values : ``(n_true_pixels_in_mask,)`` `ndarray`
            Sampled value to rebuild the masked image from.
        """
        from menpo.image import MaskedImage
        warped_image = MaskedImage.init_blank(template_mask.shape,
                                              n_channels=self.n_channels,
                                              mask=template_mask)
        warped_image._from_vector_inplace(sampled_pixel_values.ravel())
        return warped_image

    def sample(self, points_to_sample, order=1, mode='constant', cval=0.0):
        r"""
        Sample this image at the given sub-pixel accurate points. The input
        PointCloud should have the same number of dimensions as the image e.g.
        a 2D PointCloud for a 2D multi-channel image. A numpy array will be
        returned the has the values for every given point across each channel
        of the image.

        Parameters
        ----------
        points_to_sample : :map:`PointCloud`
            Array of points to sample from the image. Should be
            `(n_points, n_dims)`
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5].
            See warp_to_shape for more information.
        mode : ``{constant, nearest, reflect, wrap}``, optional
            Points outside the boundaries of the input are filled according
            to the given mode.
        cval : `float`, optional
            Used in conjunction with mode ``constant``, the value outside
            the image boundaries.

        Returns
        -------
        sampled_pixels : (`n_points`, `n_channels`) `ndarray`
            The interpolated values taken across every channel of the image.
        """
        # The public interface is a PointCloud, but when this is used internally
        # a numpy array is passed. So let's just treat the PointCloud as a
        # 'special case' and not document the ndarray ability.
        if isinstance(points_to_sample, PointCloud):
            points_to_sample = points_to_sample.points
        return scipy_interpolation(self.pixels, points_to_sample,
                                   order=order,  mode=mode, cval=cval)

    def warp_to_shape(self, template_shape, transform, warp_landmarks=True,
                      order=1, mode='constant', cval=0.0, batch_size=None,
                      return_transform=False):
        """
        Return a copy of this image warped into a different reference space.

        Parameters
        ----------
        template_shape : `tuple` or `ndarray`
            Defines the shape of the result, and what pixel indices should be
            sampled (all of them).
        transform : :map:`Transform`
            Transform **from the template_shape space back to this image**.
            Defines, for each index on template_shape, which pixel location
            should be sampled from on this image.
        warp_landmarks : `bool`, optional
            If ``True``, result will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5]
            
            ========= ====================
            Order     Interpolation
            ========= ====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= ====================

        mode : ``{constant, nearest, reflect, wrap}``, optional
            Points outside the boundaries of the input are filled according
            to the given mode.
        cval : `float`, optional
            Used in conjunction with mode ``constant``, the value outside
            the image boundaries.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this
            value can cause warping to become much slower, particular for
            cached warps such as Piecewise Affine. This size indicates
            how many points in the image should be warped at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are warped at once.
        return_transform : `bool`, optional
            This argument is for internal use only. If ``True``, then the
            :map:`Transform` object is also returned.

        Returns
        -------
        warped_image : `type(self)`
            A copy of this image, warped.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        template_shape = np.array(template_shape, dtype=np.int)
        if (isinstance(transform, Affine) and order in range(4) and
            self.n_dims == 2):

            # we are going to be able to go fast.

            if isinstance(transform, Translation) and order == 0:
                # an integer translation (e.g. a crop) If this lies entirely
                # in the bounds then we can just do a copy. We need to match
                # the behavior of cython_interpolation exactly, which means
                # matching its rounding behavior too:
                t = transform.translation_component.copy()
                pos_t = t > 0.0
                t[pos_t] += 0.5
                t[~pos_t] -= 0.5
                min_ = t.astype(np.int)
                max_ = template_shape + min_
                if np.all(max_ <= np.array(self.shape)) and np.all(min_ >= 0):
                    # we have a crop - slice the pixels.
                    warped_pixels = self.pixels[:,
                                    int(min_[0]):int(max_[0]),
                                    int(min_[1]):int(max_[1])].copy()
                    return self._build_warp_to_shape(warped_pixels, transform,
                                                     warp_landmarks,
                                                     return_transform)
            # we couldn't do the crop, but skimage has an optimised Cython
            # interpolation for 2D affine warps - let's use that
            sampled = cython_interpolation(self.pixels, template_shape,
                                           transform, order=order,
                                           mode=mode, cval=cval)
        else:
            template_points = indices_for_image_of_shape(template_shape)
            points_to_sample = transform.apply(template_points,
                                               batch_size=batch_size)
            sampled = self.sample(points_to_sample,
                                  order=order, mode=mode, cval=cval)

        # set any nan values to 0
        sampled[np.isnan(sampled)] = 0
        # build a warped version of the image
        warped_pixels = sampled.reshape(
            (self.n_channels,) + tuple(template_shape))

        return self._build_warp_to_shape(warped_pixels, transform,
                                         warp_landmarks, return_transform)

    def _build_warp_to_shape(self, warped_pixels, transform, warp_landmarks,
                             return_transform):
        # factored out common logic from the different paths we can take in
        # warp_to_shape. Rebuilds an image post-warp, adjusting landmarks
        # as necessary.
        warped_image = Image(warped_pixels, copy=False)

        # warp landmarks if requested.
        if warp_landmarks and self.has_landmarks:
            warped_image.landmarks = self.landmarks
            transform.pseudoinverse()._apply_inplace(warped_image.landmarks)
        if hasattr(self, 'path'):
            warped_image.path = self.path

        # optionally return the transform
        if return_transform:
            return warped_image, transform
        else:
            return warped_image

    def rescale(self, scale, round='ceil', order=1,
                return_transform=False):
        r"""
        Return a copy of this image, rescaled by a given factor.
        Landmarks are rescaled appropriately.

        Parameters
        ----------
        scale : `float` or `tuple` of `floats`
            The scale factor. If a tuple, the scale to apply to each dimension.
            If a single `float`, the scale will be applied uniformly across
            each dimension.
        round: ``{ceil, floor, round}``, optional
            Rounding function to be applied to floating point shapes.
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5]

            ========= ====================
            Order     Interpolation
            ========= ====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= ====================

        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the rescale is also returned.

        Returns
        -------
        rescaled_image : ``type(self)``
            A copy of this image, rescaled.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

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
                                  mode='nearest',
                                  return_transform=return_transform)

    def rescale_to_diagonal(self, diagonal, round='ceil',
                            return_transform=False):
        r"""
        Return a copy of this image, rescaled so that the it's diagonal is a
        new size.

        Parameters
        ----------
        diagonal: `int`
            The diagonal size of the new image.
        round: ``{ceil, floor, round}``, optional
            Rounding function to be applied to floating point shapes.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the rescale is also returned.

        Returns
        -------
        rescaled_image : type(self)
            A copy of this image, rescaled.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        return self.rescale(diagonal / self.diagonal(), round=round,
                            return_transform=return_transform)

    def rescale_to_pointcloud(self, pointcloud, group=None,
                              round='ceil', order=1,
                              return_transform=False):
        r"""
        Return a copy of this image, rescaled so that the scale of a
        particular group of landmarks matches the scale of the passed
        reference pointcloud.

        Parameters
        ----------
        pointcloud: :map:`PointCloud`
            The reference pointcloud to which the landmarks specified by
            ``group`` will be scaled to match.
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        round: ``{ceil, floor, round}``, optional
            Rounding function to be applied to floating point shapes.
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5]

            ========= ====================
            Order     Interpolation
            ========= ====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= ====================

        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the rescale is also returned.

        Returns
        -------
        rescaled_image : ``type(self)``
            A copy of this image, rescaled.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        pc = self.landmarks[group].lms
        scale = AlignmentUniformScale(pc, pointcloud).as_vector().copy()
        return self.rescale(scale, round=round, order=order,
                            return_transform=return_transform)

    def rescale_landmarks_to_diagonal_range(self, diagonal_range, group=None,
                                            round='ceil', order=1,
                                            return_transform=False):
        r"""
        Return a copy of this image, rescaled so that the ``diagonal_range`` of
        the bounding box containing its landmarks matches the specified
        ``diagonal_range`` range.

        Parameters
        ----------
        diagonal_range: ``(n_dims,)`` `ndarray`
            The diagonal_range range that we want the landmarks of the returned
            image to have.
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``
            and if there is only one set of landmarks, this set will be used.
        round : ``{ceil, floor, round}``, optional
            Rounding function to be applied to floating point shapes.
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5]

            ========= =====================
            Order     Interpolation
            ========= =====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= =====================

        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the rescale is also returned.

        Returns
        -------
        rescaled_image : ``type(self)``
            A copy of this image, rescaled.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        x, y = self.landmarks[group].lms.range()
        scale = diagonal_range / np.sqrt(x ** 2 + y ** 2)
        return self.rescale(scale, round=round, order=order,
                            return_transform=return_transform)

    def resize(self, shape, order=1, return_transform=False):
        r"""
        Return a copy of this image, resized to a particular shape.
        All image information (landmarks, and mask in the case of
        :map:`MaskedImage`) is resized appropriately.

        Parameters
        ----------
        shape : `tuple`
            The new shape to resize to.
        order : `int`, optional
            The order of interpolation. The order has to be in the range [0,5]

            ========= =====================
            Order     Interpolation
            ========= =====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= =====================

        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the resize is also returned.

        Returns
        -------
        resized_image : ``type(self)``
            A copy of this image, resized.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

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
        return self.rescale(scales, round='round', order=order,
                            return_transform=return_transform)

    def zoom(self, scale, cval=0.0, return_transform=False):
        r"""
        Return a copy of this image, zoomed about the centre point. ``scale``
        values greater than 1.0 denote zooming **in** to the image and values
        less than 1.0 denote zooming **out** of the image. The size of the
        image will not change, if you wish to scale an image, please see
        :meth:`rescale`.

        Parameters
        ----------
        scale : `float`
            ``scale > 1.0`` denotes zooming in. Thus the image will appear
            larger and areas at the edge of the zoom will be 'cropped' out.
            ``scale < 1.0`` denotes zooming out. The image will be padded
            by the value of ``cval``.
        cval : ``float``, optional
            The value to be set outside the rotated image boundaries.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the zooming is also returned.

        Returns
        -------
        zoomed_image : ``type(self)``
            A copy of this image, zoomed.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        t = scale_about_centre(self, 1.0 / scale)
        return self.warp_to_shape(self.shape, t, cval=cval,
                                  return_transform=return_transform)

    def rotate_ccw_about_centre(self, theta, degrees=True, retain_shape=False,
                                cval=0.0, round='round', order=1,
                                return_transform=False):
        r"""
        Return a copy of this image, rotated counter-clockwise about its centre.

        Note that the `retain_shape` argument defines the shape of the rotated
        image. If ``retain_shape=True``, then the shape of the rotated image
        will be the same as the one of current image, so some regions will
        probably be cropped. If ``retain_shape=False``, then the returned image
        has the correct size so that the whole area of the current image is
        included.

        Parameters
        ----------
        theta : `float`
            The angle of rotation about the centre.
        degrees : `bool`, optional
            If ``True``, `theta` is interpreted in degrees. If ``False``,
            ``theta`` is interpreted as radians.
        retain_shape : `bool`, optional
            If ``True``, then the shape of the rotated image will be the same as
            the one of current image, so some regions will probably be cropped.
            If ``False``, then the returned image has the correct size so that
            the whole area of the current image is included.
        cval : `float`, optional
            The value to be set outside the rotated image boundaries.
        round : ``{'ceil', 'floor', 'round'}``, optional
            Rounding function to be applied to floating point shapes. This is
            only used in case ``retain_shape=True``.
        order : `int`, optional
            The order of interpolation. The order has to be in the range
            ``[0,5]``. This is only used in case ``retain_shape=True``.

            ========= ====================
            Order     Interpolation
            ========= ====================
            0         Nearest-neighbor
            1         Bi-linear *(default)*
            2         Bi-quadratic
            3         Bi-cubic
            4         Bi-quartic
            5         Bi-quintic
            ========= ====================

        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the rotation is also returned.

        Returns
        -------
        rotated_image : ``type(self)``
            The rotated image.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ValueError
            Image rotation is presently only supported on 2D images
        """
        if self.n_dims != 2:
            raise ValueError('Image rotation is presently only supported on '
                             '2D images')

        if retain_shape:
            # Rotate the image about its centre
            trans = rotate_ccw_about_centre(self, theta, degrees=degrees)
            # Output image's shape must be the same as the original one
            shape = self.shape
        else:
            # Get image's bounding box coordinates
            bbox = bounding_box((0, 0), [self.shape[0] - 1, self.shape[1] - 1])
            # Translate to origin and rotate counter-clockwise
            trans = Translation(-self.centre(),
                                skip_checks=True).compose_before(
                Rotation.init_from_2d_ccw_angle(theta, degrees=degrees))
            rotated_bbox = trans.apply(bbox)
            # Create new translation so that min bbox values go to 0
            t = Translation(-rotated_bbox.bounds()[0])
            trans.compose_before_inplace(t)
            rotated_bbox = trans.apply(bbox)
            # Output image's shape is the range of the rotated bounding box
            # while respecting the users rounding preference.
            shape = round_image_shape(rotated_bbox.range() + 1, round)

        # Warp image
        return self.warp_to_shape(
            shape, trans.pseudoinverse(), order=order, warp_landmarks=True,
            cval=cval, return_transform=return_transform)

    def mirror(self, axis=1, return_transform=False):
        r"""
        Return a copy of this image, mirrored/flipped about a certain axis.

        Parameters
        ----------
        axis : `int`, optional
            The axis about which to mirror the image.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the mirroring is also returned.

        Returns
        -------
        mirrored_image : ``type(self)``
            The mirrored image.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ValueError
            axis cannot be negative
        ValueError
            axis={} but the image has {} dimensions
        """
        # Check axis argument
        if axis < 0:
            raise ValueError('axis cannot be negative')
        elif axis >= self.n_dims:
            raise ValueError("axis={} but the image has {} "
                             "dimensions".format(axis, self.n_dims))

        # Create transform that includes ...
        # ... flipping about the selected axis ...
        rot_matrix = np.eye(self.n_dims)
        rot_matrix[axis, axis] = -1
        # ... and translating back to the image's bbox
        tr_matrix = np.zeros(self.n_dims)
        tr_matrix[axis] = self.shape[axis] - 1

        # Create transform object
        trans = Rotation(rot_matrix, skip_checks=True).compose_before(
            Translation(tr_matrix, skip_checks=True))

        # Warp image
        return self.warp_to_shape(self.shape, trans.pseudoinverse(),
                                  warp_landmarks=True,
                                  return_transform=return_transform)

    def pyramid(self, n_levels=3, downscale=2):
        r"""
        Return a rescaled pyramid of this image. The first image of the
        pyramid will be the original, unmodified, image, and counts as level 1.

        Parameters
        ----------
        n_levels : `int`, optional
            Total number of levels in the pyramid, including the original
            unmodified image
        downscale : `float`, optional
            Downscale factor.

        Yields
        ------
        image_pyramid: `generator`
            Generator yielding pyramid layers as :map:`Image` objects.
        """
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
        n_levels : `int`, optional
            Total number of levels in the pyramid, including the original
            unmodified image
        downscale : `float`, optional
            Downscale factor.
        sigma : `float`, optional
            Sigma for gaussian filter. Default is ``downscale / 3.`` which
            corresponds to a filter mask twice the size of the scale factor
            that covers more than 99% of the gaussian distribution.

        Yields
        ------
        image_pyramid: `generator`
            Generator yielding pyramid layers as :map:`Image` objects.
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
        represent a 2D RGB image, then the ``luminosity`` mode will fail.

        Parameters
        ----------
        mode : ``{average, luminosity, channel}``, optional
            ============== =====================================================
            mode           Greyscale Algorithm
            ============== =====================================================
            average        Equal average of all channels
            luminosity     Calculates the luminance using the CCIR 601 formula:
            |              .. math:: Y' = 0.2989 R' + 0.5870 G' + 0.1140 B'
            channel        A specific channel is chosen as the intensity value.
            ============== =====================================================

        channel: `int`, optional
            The channel to be taken. Only used if mode is ``channel``.

        Returns
        -------
        greyscale_image : :map:`MaskedImage`
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
            # Only compute the coefficients once.
            global _greyscale_luminosity_coef
            if _greyscale_luminosity_coef is None:
                _greyscale_luminosity_coef = np.linalg.inv(
                    np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.106, 1.703]]))[0, :]
            # Compute greyscale via dot product
            pixels = np.dot(_greyscale_luminosity_coef,
                            greyscale.pixels.reshape(3, -1))
            # Reshape image back to original shape (with 1 channel)
            pixels = pixels.reshape(greyscale.shape)
        elif mode == 'average':
            pixels = np.mean(greyscale.pixels, axis=0)
        elif mode == 'channel':
            if channel is None:
                raise ValueError("For the 'channel' mode you have to provide"
                                 " a channel index")
            pixels = greyscale.pixels[channel]
        else:
            raise ValueError("Unknown mode {} - expected 'luminosity', "
                             "'average' or 'channel'.".format(mode))

        # Set new pixels - ensure channel axis and maintain
        greyscale.pixels = pixels[None, ...].astype(greyscale.pixels.dtype,
                                                    copy=False)
        return greyscale

    def as_PILImage(self, out_dtype=np.uint8):
        r"""
        Return a PIL copy of the image scaled and cast to the correct
        values for the provided ``out_dtype``.

        Image must only have 1 or 3 channels and be 2 dimensional.
        Non `uint8` floating point images must be in the range ``[0, 1]`` to be
        converted.

        Parameters
        ----------
        out_dtype : `np.dtype`, optional
            The dtype the output array should be.

        Returns
        -------
        pil_image : `PILImage`
            PIL copy of image

        Raises
        ------
        ValueError
            If image is not 2D and has 1 channel or 3 channels.
        ValueError
            If pixels data type is `float32` or `float64` and the pixel
            range is outside of ``[0, 1]``
        ValueError
            If the output dtype is unsupported. Currently uint8 is supported.
        """
        if self.n_dims != 2 or (self.n_channels != 1 and self.n_channels != 3):
            raise ValueError(
                'Can only convert greyscale or RGB 2D images. '
                'Received a {} channel {}D image.'.format(self.n_channels,
                                                          self.n_dims))

        # Slice off the channel for greyscale images
        if self.n_channels == 1:
            pixels = self.pixels[0]
        else:
            pixels = channels_to_back(self.pixels)
        pixels = denormalise_pixels_range(pixels, out_dtype)
        return PILImage.fromarray(pixels)

    def as_imageio(self, out_dtype=np.uint8):
        r"""
        Return an Imageio copy of the image scaled and cast to the correct
        values for the provided ``out_dtype``.

        Image must only have 1 or 3 channels and be 2 dimensional.
        Non `uint8` floating point images must be in the range ``[0, 1]`` to be
        converted.

        Parameters
        ----------
        out_dtype : `np.dtype`, optional
            The dtype the output array should be.

        Returns
        -------
        imageio_image : `ndarray`
            Imageio image (which is just a numpy ndarray with the channels
            as the last axis).

        Raises
        ------
        ValueError
            If image is not 2D and has 1 channel or 3 channels.
        ValueError
            If pixels data type is `float32` or `float64` and the pixel
            range is outside of ``[0, 1]``
        ValueError
            If the output dtype is unsupported. Currently uint8 and uint16
            are supported.
        """
        if self.n_dims != 2 or (self.n_channels != 1 and self.n_channels != 3):
            raise ValueError(
                'Can only convert greyscale or RGB 2D images. '
                'Received a {} channel {}D image.'.format(self.n_channels,
                                                          self.n_dims))

        # Slice off the channel for greyscale images
        if self.n_channels == 1:
            pixels = self.pixels[0]
        else:
            pixels = channels_to_back(self.pixels)
        return denormalise_pixels_range(pixels, out_dtype)

    def pixels_range(self):
        r"""
        The range of the pixel values (min and max pixel values).

        Returns
        -------
        min_max : ``(dtype, dtype)``
            The minimum and maximum value of the pixels array.
        """
        return self.pixels.min(), self.pixels.max()

    def rolled_channels(self):
        r"""
        Returns the pixels matrix, with the channels rolled to the back axis.
        This may be required for interacting with external code bases that
        require images to have channels as the last axis, rather than the
        menpo convention of channels as the first axis.

        Returns
        -------
        rolled_channels : `ndarray`
            Pixels with channels as the back (last) axis.
        """
        return channels_to_back(self.pixels)

    def __str__(self):
        return ('{} {}D Image with {} channel{}'.format(
            self._str_shape(), self.n_dims, self.n_channels,
            's' * (self.n_channels > 1)))

    @property
    def has_landmarks_outside_bounds(self):
        """
        Indicates whether there are landmarks located outside the image bounds.

        :type: `bool`
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
        Deprecated. See the non-mutating API, `normalize_std()`.
        """
        warn('the public API for inplace operations is deprecated '
             'and will be removed in a future version of Menpo. '
             'Use .normalize_std() instead.', MenpoDeprecationWarning)
        self._normalize_inplace(np.std, mode=mode)

    def normalize_std(self, mode='all', **kwargs):
        r"""
        Returns a copy of this image normalized such that its
        pixel values have zero mean and unit variance.

        Parameters
        ----------
        mode : ``{all, per_channel}``, optional
            If ``all``, the normalization is over all channels. If
            ``per_channel``, each channel individually is mean centred and
            normalized in variance.
        """
        return self._normalize(np.std, mode=mode)

    def normalize_norm_inplace(self, mode='all', **kwargs):
        r"""
        Deprecated. See the non-mutating API, `normalize_norm()`.
        """
        warn('the public API for inplace operations is deprecated '
             'and will be removed in a future version of Menpo. '
             'Use .normalize_norm() instead.', MenpoDeprecationWarning)

        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        self._normalize_inplace(scale_func, mode=mode)

    def normalize_norm(self, mode='all', **kwargs):
        r"""
        Returns a copy of this image normalized such that its pixel values
        have zero mean and its norm equals 1.

        Parameters
        ----------
        mode : ``{all, per_channel}``, optional
            If ``all``, the normalization is over all channels. If
            ``per_channel``, each channel individually is mean centred and
            normalized in variance.

        Returns
        -------
        image : ``type(self)``
            A copy of this image, normalized.
        """
        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        return self._normalize(scale_func, mode=mode)

    def _normalize(self, scale_func, mode='all'):
        new = self.copy()
        new._normalize_inplace(scale_func, mode=mode)
        return new

    def _normalize_inplace(self, scale_func, mode='all'):
        pixels = self.as_vector(keep_channels=True)
        if mode == 'all':
            centered_pixels = pixels - np.mean(pixels)
            scale_factor = scale_func(centered_pixels)

        elif mode == 'per_channel':
            centered_pixels = pixels - np.mean(pixels, axis=1)[..., None]
            scale_factor = scale_func(centered_pixels, axis=1)[..., None]
        else:
            raise ValueError("mode has to be 'all' or 'per_channel' - '{}' "
                             "was provided instead".format(mode))

        if np.any(scale_factor == 0):
            raise ValueError("Image has 0 variance - can't be "
                             "normalized")
        else:
            self._from_vector_inplace(centered_pixels / scale_factor)

    def rescale_pixels(self, minimum, maximum, per_channel=True):
        r"""A copy of this image with pixels linearly rescaled to fit a range.

        Note that the only pixels that will considered and rescaled are those
        that feature in the vectorized form of this image. If you want to use
        this routine on all the pixels in a :map:`MaskedImage`, consider
        using `as_unmasked()` prior to this call.

        Parameters
        ----------
        minimum: `float`
            The minimal value of the rescaled pixels
        maximum: `float`
            The maximal value of the rescaled pixels
        per_channel: `boolean`, optional
            If ``True``, each channel will be rescaled independently. If
            ``False``, the scaling will be over all channels.

        Returns
        -------
        rescaled_image: ``type(self)``
            A copy of this image with pixels linearly rescaled to fit in the
            range provided.
        """
        v = self.as_vector(keep_channels=True).T
        if per_channel:
            min_, max_ = v.min(axis=0), v.max(axis=0)
        else:
            min_, max_ = v.min(), v.max()
        sf = ((maximum - minimum) * 1.0) / (max_ - min_)
        v_new = ((v - min_) * sf) + minimum
        return self.from_vector(v_new.T.ravel())


def round_image_shape(shape, round):
    if round not in ['ceil', 'round', 'floor']:
        raise ValueError('round must be either ceil, round or floor')
    # Ensure that the '+' operator means concatenate tuples
    return tuple(getattr(np, round)(shape).astype(np.int))


def _convert_patches_list_to_single_array(patches_list, n_center):
    r"""
    Converts patches from a `list` of :map:`Image` objects to a single `ndarray`
    with shape ``(n_center, n_offset, self.n_channels, patch_shape)``.

    Note that these two are the formats returned by the `extract_patches()`
    and `extract_patches_around_landmarks()` methods of :map:`Image` class.

    Parameters
    ----------
    patches_list : `list` of `n_center * n_offset` :map:`Image` objects
        A `list` that contains all the patches as :map:`Image` objects.
    n_center : `int`
        The number of centers from which the patches are extracted.

    Returns
    -------
    patches_array : `ndarray` ``(n_center, n_offset, n_channels, patch_shape)``
        The numpy array that contains all the patches.
    """
    n_offsets = np.int(len(patches_list) / n_center)
    n_channels = patches_list[0].n_channels
    height = patches_list[0].height
    width = patches_list[0].width
    patches_array = np.empty((n_center, n_offsets, n_channels, height, width),
                             dtype=patches_list[0].pixels.dtype)
    total_index = 0
    for p in range(n_center):
        for o in range(n_offsets):
            patches_array[p, o, ...] = patches_list[total_index].pixels
            total_index += 1
    return patches_array


def _create_patches_image(patches, patch_centers, patches_indices=None,
                          offset_index=None, background='black'):
    r"""
    Creates an :map:`Image` object in which the patches are located on the
    correct regions based on the centers. Thus, the image is a block-sparse
    matrix. It has also two attached :map:`LandmarkGroup` objects. The
    `all_patch_centers` one contains all the patch centers, while the
    `selected_patch_centers` one contains only the centers that correspond to
    the patches that the user selected to set.

    The patches argument can have any of the two formats that are returned
    from the `extract_patches()` and `extract_patches_around_landmarks()`
    methods of the :map:`Image` class. Specifically it can be:

        1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
        2. `list` of ``n_center * n_offset`` :map:`Image` objects

    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches. It can have any of the two formats that are
        returned from the `extract_patches()` and
        `extract_patches_around_landmarks()` methods. Specifically, it can
        either be an ``(n_center, n_offset, self.n_channels, patch_shape)``
        `ndarray` or a `list` of ``n_center * n_offset`` :map:`Image` objects.
    patch_centers : :map:`PointCloud`
        The centers to set the patches around.
    patches_indices : `int` or `list` of `int` or ``None``, optional
        Defines the patches that will be set (copied) to the image. If ``None``,
        then all the patches are copied.
    offset_index : `int` or ``None``, optional
        The offset index within the provided `patches` argument, thus the index
        of the second dimension from which to sample. If ``None``, then ``0`` is
        used.
    background : ``{'black', 'white'}``, optional
        If ``'black'``, then the background is set equal to the minimum value
        of `patches`. If ``'white'``, then the background is set equal to the
        maximum value of `patches`.

    Returns
    -------
    patches_image : :map:`Image`
        The output patches image object.

    Raises
    ------
    ValueError
        Background must be either ''black'' or ''white''.
    """
    # If patches is a list, convert it to array
    if isinstance(patches, list):
        patches = _convert_patches_list_to_single_array(patches,
                                                        patch_centers.n_points)

    # Parse inputs
    if offset_index is None:
        offset_index = 0
    if patches_indices is None:
        patches_indices = np.arange(patches.shape[0])
    elif not isinstance(patches_indices, Iterable):
        patches_indices = [patches_indices]

    # Compute patches image's shape
    n_channels = patches.shape[2]
    patch_shape0 = patches.shape[3]
    patch_shape1 = patches.shape[4]
    top, left = np.min(patch_centers.points, 0)
    bottom, right = np.max(patch_centers.points, 0)
    min_0 = np.floor(top - patch_shape0)
    min_1 = np.floor(left - patch_shape1)
    max_0 = np.ceil(bottom + patch_shape0)
    max_1 = np.ceil(right + patch_shape1)
    height = max_0 - min_0 + 1
    width = max_1 - min_1 + 1

    # Translate the patch centers to fit in the new image
    new_patch_centers = patch_centers.copy()
    new_patch_centers.points = patch_centers.points - np.array([[min_0, min_1]])

    # Create temporary pointcloud with the selected patch centers
    tmp_centers = PointCloud(new_patch_centers.points[patches_indices])

    # Create new image with the correct background values
    if background == 'black':
        patches_image = Image.init_blank(
            (height, width), n_channels,
            fill=np.min(patches[patches_indices]),
            dtype=patches.dtype)
    elif background == 'white':
        patches_image = Image.init_blank(
            (height, width), n_channels,
            fill=np.max(patches[patches_indices]),
            dtype=patches.dtype)
    else:
        raise ValueError('Background must be either ''black'' or ''white''.')

    # Attach the corrected patch centers
    patches_image.landmarks['all_patch_centers'] = new_patch_centers
    patches_image.landmarks['selected_patch_centers'] = tmp_centers

    # Set the patches
    patches_image.set_patches_around_landmarks(patches[patches_indices],
                                               group='selected_patch_centers',
                                               offset_index=offset_index)

    return patches_image
