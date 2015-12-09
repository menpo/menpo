from __future__ import division
from warnings import warn
import numpy as np
binary_erosion = None  # expensive, from scipy.ndimage
binary_dilation = None  # expensive, from scipy.ndimage

from menpo.base import MenpoDeprecationWarning
from menpo.visualize.base import ImageViewer

from .base import Image
from .boolean import BooleanImage


class OutOfMaskSampleError(ValueError):
    r"""
    Exception that is thrown when an attempt is made to sample an MaskedImage
    in an area that is masked out (where the mask is ``False``).

    Parameters
    ----------
    sampled_mask : `bool ndarray`
        The sampled mask, ``True`` where the image's mask was ``True`` and
        ``False`` otherwise. Useful for masking out the sampling array.
    sampled_values : `ndarray`
        The sampled values, no attempt at masking is made.
    """
    def __init__(self, sampled_mask, sampled_values):
        super(OutOfMaskSampleError, self).__init__()
        self.sampled_mask = sampled_mask
        self.sampled_values = sampled_values


class MaskedImage(Image):
    r"""
    Represents an `n`-dimensional `k`-channel image, which has a mask.
    Images can be masked in order to identify a region of interest. All
    images implicitly have a mask that is defined as the the entire image.
    The mask is an instance of :map:`BooleanImage`.

    Parameters
    ----------
    image_data :  ``(C, M, N ..., Q)`` `ndarray`
        The pixel data for the image, where the first axis represents the
        number of channels.
    mask : ``(M, N)`` `bool ndarray` or :map:`BooleanImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).
    copy: `bool`, optional
        If ``False``, the ``image_data`` will not be copied on assignment. If a
        mask is provided, this also won't be copied. In general this should only
        be used if you know what you are doing.

    Raises
    ------
    ValueError
        Mask is not the same shape as the image
    """
    def __init__(self, image_data, mask=None, copy=True):
        super(MaskedImage, self).__init__(image_data, copy=copy)
        if mask is not None:
            # Check if we need to create a BooleanImage or not
            if not isinstance(mask, BooleanImage):
                # So it's a numpy array.
                mask_image = BooleanImage(mask, copy=copy)
            else:
                # It's a BooleanImage object.
                if copy:
                    mask = mask.copy()
                mask_image = mask
            if mask_image.shape == self.shape:
                self.mask = mask_image
            else:
                raise ValueError("Trying to construct a Masked Image of "
                                 "shape {} with a Mask of differing "
                                 "shape {}".format(self.shape,
                                                   mask.shape))
        else:
            # no mask provided - make the default.
            self.mask = BooleanImage.init_blank(self.shape, fill=True)

    @classmethod
    def init_blank(cls, shape, n_channels=1, fill=0, dtype=np.float, mask=None):
        r"""Generate a blank masked image

        Parameters
        ----------
        shape : `tuple` or `list`
            The shape of the image. Any floating point values are rounded up
            to the nearest integer.
        n_channels: `int`, optional
            The number of channels to create the image with.
        fill : `int`, optional
            The value to fill all pixels with.
        dtype: `numpy datatype`, optional
            The datatype of the image.
        mask: ``(M, N)`` `bool ndarray` or :map:`BooleanImage`
            An optional mask that can be applied to the image. Has to have a
            shape equal to that of the image.

        Notes
        -----
        Subclasses of :map:`MaskedImage` need to overwrite this method and
        explicitly call this superclass method

        ::

            super(SubClass, cls).init_blank(shape,**kwargs)

        in order to appropriately propagate the subclass type to ``cls``.

        Returns
        -------
        blank_image : :map:`MaskedImage`
            A new masked image of the requested size.
        """
        # Ensure that the '+' operator means concatenate tuples
        shape = tuple(np.ceil(shape).astype(np.int))
        if fill == 0:
            pixels = np.zeros((n_channels,) + shape, dtype=dtype)
        else:
            pixels = np.ones((n_channels,) + shape, dtype=dtype) * fill
        return cls(pixels, copy=False, mask=mask)

    @classmethod
    def init_from_rolled_channels(cls, pixels, mask=None):
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
        mask : ``(M, N)`` `bool ndarray` or :map:`BooleanImage`, optional
            A binary array representing the mask. Must be the same
            shape as the image. Only one mask is supported for an image (so the
            mask is applied to every channel equally).

        Returns
        -------
        image : :map:`Image`
            A new image from the given pixels, with the FIRST axis as the
            channels.
        """
        return cls(np.rollaxis(pixels, -1), mask=mask)

    def as_unmasked(self, copy=True, fill=None):
        r"""
        Return a copy of this image without the masking behavior.

        By default the mask is simply discarded. However, there is an optional
        kwarg, ``fill``, that can be set which will fill the **non-masked**
        areas with the given value.

        Parameters
        ----------
        copy : `bool`, optional
            If ``False``, the produced :map:`Image` will share pixels with
            ``self``. Only suggested to be used for performance.
        fill : `float` or ``None``, optional
            If ``None`` the mask is simply discarded. If a number, the
             *unmasked* regions are filled with the given value.

        Returns
        -------
        image : :map:`Image`
            An image with the same pixels and landmarks as this one, but with
            no mask.
        """
        img = Image(self.pixels, copy=copy)
        if fill is not None:
            img.pixels[..., ~self.mask.mask] = fill

        if self.has_landmarks:
            img.landmarks = self.landmarks
        if hasattr(self, 'path'):
            img.path = self.path
        return img

    def n_true_pixels(self):
        r"""
        The number of ``True`` values in the mask.

        :type: `int`
        """
        return self.mask.n_true()

    def n_false_pixels(self):
        r"""
        The number of ``False`` values in the mask.

        :type: `int`
        """
        return self.mask.n_false()

    def n_true_elements(self):
        r"""
        The number of ``True`` elements of the image over all the channels.

        :type: `int`
        """
        return self.n_true_pixels() * self.n_channels

    def n_false_elements(self):
        r"""
        The number of ``False`` elements of the image over all the channels.

        :type: `int`
        """
        return self.n_false_pixels() * self.n_channels

    def indices(self):
        r"""
        Return the indices of all true pixels in this image.

        :type: ``(n_dims, n_true_pixels)`` `ndarray`
        """
        return self.mask.true_indices()

    def masked_pixels(self):
        r"""
        Get the pixels covered by the `True` values in the mask.

        :type: ``(n_channels, mask.n_true)`` `ndarray`
        """
        if self.mask.all_true():
            return self.pixels
        return self.pixels[..., self.mask.mask]

    def set_masked_pixels(self, pixels, copy=True):
        r"""
        Update the masked pixels only to new values.

        Parameters
        ----------
        pixels: `ndarray`
            The new pixels to set.
        copy: `bool`, optional
            If ``False`` a copy will be avoided in assignment. This can only
            happen if the mask is all ``True`` - in all other cases it will
            raise a warning.

        Raises
        ------
        Warning
            If the ``copy=False`` flag cannot be honored.
        """
        if self.mask.all_true():
            # reshape the vector into the image again
            pixels = pixels.reshape((self.n_channels,) + self.shape)
            if not copy:
                if not pixels.flags.c_contiguous:
                    warn('The copy flag was NOT honoured. A copy HAS been '
                         'made. Copy can only be avoided if MaskedImage has '
                         'an all_true mask and the pixels provided are '
                         'C-contiguous.')
                    pixels = pixels.copy()
            else:
                pixels = pixels.copy()
            self.pixels = pixels
        else:
            self.pixels[..., self.mask.mask] = pixels
            # oh dear, couldn't avoid a copy. Did the user try to?
            if not copy:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'copy can only be avoided if MaskedImage has an all_true '
                     'mask.')

    def __str__(self):
        return ('{} {}D MaskedImage with {} channels. '
                'Attached mask {:.1%} true'.format(
            self._str_shape(), self.n_dims, self.n_channels,
            self.mask.proportion_true()))

    def _as_vector(self, keep_channels=False):
        r"""
        Convert image to a vectorized form. Note that the only pixels
        returned here are from the masked region on the image.

        Parameters
        ----------
        keep_channels : `bool`, optional

            ========== =================================
            Value      Return shape
            ========== =================================
            ``True``     ``(mask.n_true, n_channels)``
            ``False``    ``(mask.n_true * n_channels,)``
            ========== =================================

        Returns
        -------
        vectorized_image : (shape given by ``keep_channels``) `ndarray`
            Vectorized image
        """
        if keep_channels:
            return self.masked_pixels().reshape([self.n_channels, -1])
        else:
            return self.masked_pixels().ravel()

    def from_vector(self, vector, n_channels=None):
        r"""
        Takes a flattened vector and returns a new image formed by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        On masked images, the vector is always copied.

        The ``n_channels`` argument is useful for when we want to add an extra
        channel to an image but maintain the shape. For example, when
        calculating the gradient.

        Note that landmarks are transferred in the process.

        Parameters
        ----------
        vector : ``(n_pixels,)``
            A flattened vector of all pixels and channels of an image.
        n_channels : `int`, optional
            If given, will assume that vector is the same shape as this image,
            but with a possibly different number of channels.

        Returns
        -------
        image : :class:`MaskedImage`
            New image of same shape as this image and the number of
            specified channels.
        """
        # This is useful for when we want to add an extra channel to an image
        # but maintain the shape. For example, when calculating the gradient
        n_channels = self.n_channels if n_channels is None else n_channels
        # Creates zeros of size (n_channels x M x N x ...)
        if self.mask.all_true():
            # we can just reshape the array!
            image_data = vector.reshape(((n_channels,) + self.shape))
        else:
            image_data = np.zeros((n_channels,) + self.shape,
                                  dtype=vector.dtype)
            pixels_per_channel = vector.reshape((n_channels, -1))
            image_data[..., self.mask.mask] = pixels_per_channel
        new_image = MaskedImage(image_data, mask=self.mask)
        new_image.landmarks = self.landmarks
        return new_image

    def _from_vector_inplace(self, vector, copy=True):
        r"""
        Takes a flattened vector and updates this image by reshaping
        the vector to the correct pixels and channels. Note that the only
        region of the image that will be filled is the masked region.

        Parameters
        ----------
        vector : ``(n_parameters,)``
            A flattened vector of all pixels and channels of an image.
        copy : `bool`, optional
            If ``False``, the vector will be set as the pixels with no copy
            made.
            If ``True`` a copy of the vector is taken.

        Raises
        ------
        Warning
            If ``copy=False`` cannot be honored.
        """
        self.set_masked_pixels(vector.reshape((self.n_channels, -1)),
                               copy=copy)

    def _view_2d(self, figure_id=None, new_figure=False, channels=None,
                 masked=True, interpolation='bilinear', cmap_name=None,
                 alpha=1., render_axes=False, axes_font_name='sans-serif',
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
        masked : `bool`, optional
            If ``True``, only the masked pixels will be rendered.
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

        Raises
        ------
        ValueError
            If Image is not 2D
        """
        mask = self.mask.mask if masked else None
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           self.pixels, channels=channels,
                           mask=mask).render(
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def _view_landmarks_2d(self, channels=None, masked=True, group=None,
                           with_labels=None, without_labels=None,
                           figure_id=None, new_figure=False,
                           interpolation='bilinear', cmap_name=None, alpha=1.,
                           render_lines=True, line_colour=None, line_style='-',
                           line_width=1, render_markers=True, marker_style='o',
                           marker_size=20, marker_face_colour=None,
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
        masked : `bool`, optional
            If ``True``, only the masked pixels will be rendered.
        group : `str` or``None`` optionals
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
            The size of the markers in points^2.
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
            self, channels, masked, group, with_labels, without_labels,
            figure_id, new_figure, interpolation, cmap_name, alpha,
            render_lines, line_colour, line_style, line_width, render_markers,
            marker_style, marker_size, marker_face_colour, marker_edge_colour,
            marker_edge_width, render_numbering, numbers_horizontal_align,
            numbers_vertical_align, numbers_font_name, numbers_font_size,
            numbers_font_style, numbers_font_weight, numbers_font_colour,
            render_legend, legend_title, legend_font_name, legend_font_style,
            legend_font_size, legend_font_weight, legend_marker_scale,
            legend_location, legend_bbox_to_anchor, legend_border_axes_pad,
            legend_n_columns, legend_horizontal_spacing,
            legend_vertical_spacing, legend_border, legend_border_padding,
            legend_shadow, legend_rounded_corners, render_axes, axes_font_name,
            axes_font_size, axes_font_style, axes_font_weight, axes_x_limits,
            axes_y_limits, axes_x_ticks, axes_y_ticks, figure_size)

    def crop_to_true_mask(self, boundary=0, constrain_to_boundary=True,
                          return_transform=False):
        r"""
        Crop this image to be bounded just the `True` values of it's mask.

        Parameters
        ----------
        boundary : `int`, optional
            An extra padding to be added all around the true mask region.
        constrain_to_boundary : `bool`, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map:`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image. Note that
            is only possible if ``boundary != 0``.
        return_transform : `bool`, optional
            If ``True``, then the :map:`Transform` object that was used to
            perform the cropping is also returned.

        Returns
        -------
        cropped_image : ``type(self)``
            A copy of this image, cropped to the true mask.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.

        Raises
        ------
        ImageBoundaryError
            Raised if 11constrain_to_boundary=False`1, and an attempt is
            made to crop the image in a way that violates the image bounds.
        """
        min_indices, max_indices = self.mask.bounds_true(
            boundary=boundary, constrain_to_bounds=False)
        # no point doing the bounds check twice - let the crop do it only.
        return self.crop(min_indices, max_indices,
                         constrain_to_boundary=constrain_to_boundary,
                         return_transform=return_transform)

    def sample(self, points_to_sample, order=1, mode='constant', cval=0.0):
        r"""
        Sample this image at the given sub-pixel accurate points. The input
        PointCloud should have the same number of dimensions as the image e.g.
        a 2D PointCloud for a 2D multi-channel image. A numpy array will be
        returned the has the values for every given point across each channel
        of the image.

        If the points to sample are *outside* of the mask (fall on a ``False``
        value in the mask), an exception is raised. This exception contains
        the information of which points were outside of the mask (``False``)
        and *also* returns the sampled points.

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

        Raises
        ------
        OutOfMaskSampleError
            One of the points to sample was outside of the valid area of the
            mask (``False`` in the mask). This exception contains both the
            mask of valid sample points, **as well as** the sampled points
            themselves, in case you want to ignore the error.
        """
        sampled_mask = self.mask.sample(points_to_sample, mode=mode, cval=cval)
        sampled_values = Image.sample(self, points_to_sample, order=order,
                                      mode=mode, cval=cval)
        if not np.all(sampled_mask):
            raise OutOfMaskSampleError(sampled_mask, sampled_values)
        return sampled_values

    # noinspection PyMethodOverriding
    def warp_to_mask(self, template_mask, transform, warp_landmarks=False,
                     order=1, mode='constant', cval=0., batch_size=None,
                     return_transform=False):
        r"""
        Warps this image into a different reference space.

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
        warped_image : ``type(self)``
            A copy of this image, warped.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        # call the super variant and get ourselves a MaskedImage back
        # with a blank mask
        warped_image = Image.warp_to_mask(self, template_mask, transform,
                                          warp_landmarks=warp_landmarks,
                                          order=order, mode=mode, cval=cval,
                                          batch_size=batch_size)
        # Set the template mask as our mask
        warped_image.mask = template_mask
        # optionally return the transform
        if return_transform:
            return warped_image, transform
        else:
            return warped_image

    # noinspection PyMethodOverriding
    def warp_to_shape(self, template_shape, transform, warp_landmarks=False,
                      order=1, mode='constant', cval=0., batch_size=None,
                      return_transform=False):
        """
        Return a copy of this :map:`MaskedImage` warped into a different
        reference space.

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
        # call the super variant and get ourselves an Image back
        warped_image = Image.warp_to_shape(self, template_shape, transform,
                                           warp_landmarks=warp_landmarks,
                                           order=order, mode=mode, cval=cval,
                                           batch_size=batch_size)
        # warp the mask separately and reattach.
        mask = self.mask.warp_to_shape(template_shape, transform,
                                       warp_landmarks=warp_landmarks,
                                       mode=mode, cval=cval)
        # efficiently turn the Image into a MaskedImage, attaching the
        # landmarks
        masked_warped_image = warped_image.as_masked(mask=mask, copy=False)
        if hasattr(warped_image, 'path'):
            masked_warped_image.path = warped_image.path
        # optionally return the transform
        if return_transform:
            return masked_warped_image, transform
        else:
            return masked_warped_image

    def normalize_std_inplace(self, mode='all', limit_to_mask=True):
        r"""
        Normalizes this image such that it's pixel values have zero mean and
        unit variance.

        Parameters
        ----------
        mode : ``{all, per_channel}``, optional
            If ``all``, the normalization is over all channels. If
            ``per_channel``, each channel individually is mean centred and
            normalized in variance.
        limit_to_mask : `bool`, optional
            If ``True``, the normalization is only performed wrt the masked
            pixels.
            If ``False``, the normalization is wrt all pixels, regardless of
            their masking value.
        """
        warn('the public API for inplace operations is deprecated '
             'and will be removed in a future version of Menpo. '
             'Use .normalize_std() instead.', MenpoDeprecationWarning)
        self._normalize_inplace(np.std, mode=mode,
                                limit_to_mask=limit_to_mask)

    def normalize_std(self, mode='all', limit_to_mask=True):
        r"""
        Returns a copy of this image normalized such that it's pixel values
        have zero mean and unit variance.

        Parameters
        ----------
        mode : ``{all, per_channel}``, optional
            If ``all``, the normalization is over all channels. If
            ``per_channel``, each channel individually is mean centred and
            normalized in variance.
        limit_to_mask : `bool`, optional
            If ``True``, the normalization is only performed wrt the masked
            pixels.
            If ``False``, the normalization is wrt all pixels, regardless of
            their masking value.
        """
        return self._normalize(np.std, mode=mode,
                               limit_to_mask=limit_to_mask)

    def normalize_norm_inplace(self, mode='all', limit_to_mask=True,
                               **kwargs):
        r"""
        Normalizes this image such that it's pixel values have zero mean and
        its norm equals 1.

        Parameters
        ----------
        mode : ``{all, per_channel}``, optional
            If ``all``, the normalization is over all channels. If
            ``per_channel``, each channel individually is mean centred and
            normalized in variance.
        limit_to_mask : `bool`, optional
            If ``True``, the normalization is only performed wrt the masked
            pixels.
            If ``False``, the normalization is wrt all pixels, regardless of
            their masking value.
        """
        warn('the public API for inplace operations is deprecated '
             'and will be removed in a future version of Menpo. '
             'Use .normalize_norm() instead.', MenpoDeprecationWarning)

        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        self._normalize_inplace(scale_func, mode=mode,
                                limit_to_mask=limit_to_mask)

    def normalize_norm(self, mode='all', limit_to_mask=True, **kwargs):
        r"""
        Returns a copy of this imaage normalized such that it's pixel values
        have zero mean and its norm equals 1.

        Parameters
        ----------
        mode : ``{all, per_channel}``, optional
            If ``all``, the normalization is over all channels. If
            ``per_channel``, each channel individually is mean centred and
            normalized in variance.
        limit_to_mask : `bool`, optional
            If ``True``, the normalization is only performed wrt the masked
            pixels.
            If ``False``, the normalization is wrt all pixels, regardless of
            their masking value.
        """

        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        return self._normalize(scale_func, mode=mode,
                               limit_to_mask=limit_to_mask)

    def _normalize(self, scale_func, mode='all', limit_to_mask=True):
        new = self.copy()
        new._normalize_inplace(scale_func, mode=mode,
                               limit_to_mask=limit_to_mask)
        return new

    def _normalize_inplace(self, scale_func, mode='all', limit_to_mask=True):
        if limit_to_mask:
            pixels = self.as_vector(keep_channels=True)
        else:
            pixels = Image.as_vector(self, keep_channels=True)
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
            normalized_pixels = centered_pixels / scale_factor

        if limit_to_mask:
            self._from_vector_inplace(normalized_pixels.flatten())
        else:
            Image._from_vector_inplace(self,
                                       normalized_pixels.flatten())

    def constrain_mask_to_landmarks(self, group=None,
                                    batch_size=None,
                                    point_in_pointcloud='pwa'):
        r"""
        Restricts this mask to be equal to the convex hull around the chosen
        landmarks.

        The choice of whether a pixel is inside or outside of the pointcloud
        is determined by the ``point_in_pointcloud`` parameter. By default
        a Piecewise Affine transform is used to test for containment, which
        is useful when building efficiently aligning images. For large images,
        a faster and pixel-accurate method can be used ('convex_hull').
        Alternatively, a callable can be provided to override the test. By
        default, the provided implementations are only valid for 2D images.

        Parameters
        ----------
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
            If the landmarks in question are an instance of :map:`TriMesh`,
            the triangulation of the landmarks will be used in the convex
            hull caculation. If the landmarks are an instance of
            :map:`PointCloud`, Delaunay triangulation will be used to
            create a triangulation.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this value
            will cause constraining to become much slower. This size indicates
            how many points in the image should be checked at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are checked at once. By default, this is only used for
            the 'pwa' point_in_pointcloud choice.
        point_in_pointcloud : {'pwa', 'convex_hull'} or `callable`
            The method used to check if pixels in the image fall inside the
            pointcloud or not. Can be accurate to a Piecewise Affine transform,
            a pixel accurate convex hull or any arbitrary callable.
            If a callable is passed, it should take two parameters,
            the :map:`PointCloud` to constrain with and the pixel locations
            ((d, n_dims) ndarray) to test and should return a (d, 1) boolean
            ndarray of whether the pixels were inside (True) or outside (False)
            of the :map:`PointCloud`.
        """
        self.mask.constrain_to_pointcloud(
            self.landmarks[group].lms, batch_size=batch_size,
            point_in_pointcloud=point_in_pointcloud)

    def build_mask_around_landmarks(self, patch_shape, group=None):
        r"""
        Restricts this images mask to be patches around each landmark in
        the chosen landmark group. This is useful for visualizing patch
        based methods.

        Parameters
        ----------
        patch_shape : `tuple`
            The size of the patch.
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        """
        # get the selected pointcloud
        pc = self.landmarks[group].lms
        # temporarily set all mask values to False
        self.mask.pixels[:] = False
        # create a patches array of the correct size, full of True values
        patches = np.ones((pc.n_points, 1, 1, int(patch_shape[0]),
                           int(patch_shape[1])), dtype=np.bool)
        # set True patches around pointcloud centers
        self.mask.set_patches(patches, pc)

    def set_boundary_pixels(self, value=0.0, n_pixels=1):
        r"""
        Returns a copy of this :map:`MaskedImage` for which n pixels along
        the its mask boundary have been set to a particular value. This is
        useful in situations where there is absent data in the image which
        can cause, for example, erroneous computations of gradient or features.

        Parameters
        ----------
        value : float or (n_channels, 1) ndarray
        n_pixels : int, optional
            The number of pixels along the mask boundary that will be set to 0.

        Returns
        -------
         : :map:`MaskedImage`
            The copy of the image for which the n pixels along its mask
            boundary have been set to a particular value.
        """
        global binary_erosion
        if binary_erosion is None:
            from scipy.ndimage import binary_erosion  # expensive
        # Erode the edge of the mask in by one pixel
        eroded_mask = binary_erosion(self.mask.mask, iterations=n_pixels)

        # replace the eroded mask with the diff between the two
        # masks. This is only true in the region we want to nullify.
        np.logical_and(~eroded_mask, self.mask.mask, out=eroded_mask)
        # set all the boundary pixels to a particular value
        self.pixels[..., eroded_mask] = value

    def erode(self, n_pixels=1):
        r"""
        Returns a copy of this :map:`MaskedImage` in which the mask has been
        shrunk by n pixels along its boundary.

        Parameters
        ----------
        n_pixels : int, optional
            The number of pixels by which we want to shrink the mask along
            its own boundary.

        Returns
        -------
         : :map:`MaskedImage`
            The copy of the masked image in which the mask has been shrunk
            by n pixels along its boundary.
        """
        global binary_erosion
        if binary_erosion is None:
            from scipy.ndimage import binary_erosion  # expensive
        # Erode the edge of the mask in by one pixel
        eroded_mask = binary_erosion(self.mask.mask, iterations=n_pixels)

        image = self.copy()
        image.mask = BooleanImage(eroded_mask)
        return image

    def dilate(self, n_pixels=1):
        r"""
        Returns a copy of this :map:`MaskedImage` in which its mask has
        been expanded by n pixels along its boundary.

        Parameters
        ----------
        n_pixels : int, optional
            The number of pixels by which we want to expand the mask along
            its own boundary.

        Returns
        -------
         : :map:`MaskedImage`
            The copy of the masked image in which the mask has been expanded
            by n pixels along its boundary.
        """
        global binary_dilation
        if binary_dilation is None:
            from scipy.ndimage import binary_dilation  # expensive
        # Erode the edge of the mask in by one pixel
        dilated_mask = binary_dilation(self.mask.mask, iterations=n_pixels)

        image = self.copy()
        image.mask = BooleanImage(dilated_mask)
        return image
