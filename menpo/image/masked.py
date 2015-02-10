from __future__ import division
from warnings import warn
import numpy as np
binary_erosion = None  # expensive, from scipy.ndimage

from menpo.visualize.base import ImageViewer
gradient = None  # avoid circular reference, from menpo.feature

from .base import Image
from .boolean import BooleanImage


class MaskedImage(Image):
    r"""
    Represents an `n`-dimensional `k`-channel image, which has a mask.
    Images can be masked in order to identify a region of interest. All
    images implicitly have a mask that is defined as the the entire image.
    The mask is an instance of :map:`BooleanImage`.

    Parameters
    ----------
    image_data :  ``(M, N ..., Q, C)`` `ndarray`
        The pixel data for the image, where the last axis represents the
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
            self.mask = BooleanImage.blank(self.shape, fill=True)

    def as_unmasked(self, copy=True):
        r"""
        Return a copy of this image without the masking behavior.

        By default the mask is simply discarded. In the future more options
        may be possible.

        Parameters
        ----------
        copy : `bool`, optional
            If ``False``, the produced :map:`Image` will share pixels with
            ``self``. Only suggested to be used for performance.

        Returns
        -------
        image : :map:`Image`
            An image with the same pixels and landmarks as this one, but with
            no mask.
        """
        img = Image(self.pixels, copy=copy)
        img.landmarks = self.landmarks
        return img

    @classmethod
    def blank(cls, shape, n_channels=1, fill=0, dtype=np.float, mask=None):
        r"""
        Returns a blank image

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

            super(SubClass, cls).blank(shape,**kwargs)

        in order to appropriately propagate the subclass type to ``cls``.

        Returns
        -------
        blank_image : :class:`MaskedImage`
            A new masked image of the requested size.
        """
        # Ensure that the '+' operator means concatenate tuples
        shape = tuple(np.ceil(shape).astype(np.int))
        if fill == 0:
            pixels = np.zeros(shape + (n_channels,), dtype=dtype)
        else:
            pixels = np.ones(shape + (n_channels,), dtype=dtype) * fill
        return cls(pixels, copy=False, mask=mask)

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

        :type: ``(mask.n_true, n_channels)`` `ndarray`
        """
        if self.mask.all_true():
            return self.pixels
        return self.pixels[self.mask.mask]

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
            pixels = pixels.reshape(self.shape + (self.n_channels,))
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
            self.pixels[self.mask.mask] = pixels
            # oh dear, couldn't avoid a copy. Did the user try to?
            if not copy:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'copy can only be avoided if MaskedImage has an all_true '
                     'mask.')

    def __str__(self):
        return ('{} {}D MaskedImage with {} channels. '
                'Attached mask {:.1%} true'.format(
            self._str_shape, self.n_dims, self.n_channels,
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
            return self.masked_pixels().reshape([-1, self.n_channels])
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
        # Creates zeros of size (M x N x ... x n_channels)
        if self.mask.all_true():
            # we can just reshape the array!
            image_data = vector.reshape((self.shape + (n_channels,)))
        else:
            image_data = np.zeros(self.shape + (n_channels,))
            pixels_per_channel = vector.reshape((-1, n_channels))
            image_data[self.mask.mask] = pixels_per_channel
        new_image = MaskedImage(image_data, mask=self.mask)
        new_image.landmarks = self.landmarks
        return new_image

    def from_vector_inplace(self, vector, copy=True):
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
        self.set_masked_pixels(vector.reshape((-1, self.n_channels)),
                               copy=copy)

    def _view_2d(self, figure_id=None, new_figure=False, channels=None,
                 masked=True, interpolation="bilinear", alpha=1.,
                 render_axes=False, axes_font_name='sans-serif',
                 axes_font_size=10, axes_font_style='normal',
                 axes_font_weight='normal', axes_x_limits=None,
                 axes_y_limits=None, figure_size=(10, 8)):
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

        axes_x_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the x axis.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            If Image is not 2D
        """
        mask = self.mask.mask if masked else None
        pixels_to_view = self.pixels
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channels=channels,
                           mask=mask).render(render_axes=render_axes,
                                             axes_font_name=axes_font_name,
                                             axes_font_size=axes_font_size,
                                             axes_font_style=axes_font_style,
                                             axes_font_weight=axes_font_weight,
                                             axes_x_limits=axes_x_limits,
                                             axes_y_limits=axes_y_limits,
                                             figure_size=figure_size,
                                             interpolation=interpolation,
                                             alpha=alpha)

    def _view_landmarks_2d(self, channels=None, masked=True, group=None,
                           with_labels=None, without_labels=None,
                           figure_id=None, new_figure=False,
                           interpolation='bilinear', alpha=1.,
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

        axes_x_limits : (`float`, `float`) `tuple` or ``None`` optional
            The limits of the x axis.
        axes_y_limits : (`float`, `float`) `tuple` or ``None`` optional
            The limits of the y axis.
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
            figure_id, new_figure, interpolation, alpha, render_lines,
            line_colour, line_style, line_width, render_markers, marker_style,
            marker_size, marker_face_colour, marker_edge_colour,
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
            axes_y_limits, figure_size)

    def crop_inplace(self, min_indices, max_indices,
                     constrain_to_boundary=True):
        r"""
        Crops this image using the given minimum and maximum indices.
        Landmarks are correctly adjusted so they maintain their position
        relative to the newly cropped image.

        Parameters
        ----------
        min_indices: ``(n_dims, )`` `ndarray`
            The minimum index over each dimension.
        max_indices: ``(n_dims, )`` `ndarray`
            The maximum index over each dimension.
        constrain_to_boundary : `bool`, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map:`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image.

        Returns
        -------
        cropped_image : `type(self)`
            This image, but cropped.

        Raises
        ------
        ValueError
            ``min_indices`` and ``max_indices`` both have to be of length
            ``n_dims``. All ``max_indices`` must be greater than
            ``min_indices``.
        :map`ImageBoundaryError`
            Raised if ``constrain_to_boundary=False``, and an attempt is made
            to crop the image in a way that violates the image bounds.
        """
        # crop our image
        super(MaskedImage, self).crop_inplace(
            min_indices, max_indices,
            constrain_to_boundary=constrain_to_boundary)
        # crop our mask
        self.mask.crop_inplace(min_indices, max_indices,
                               constrain_to_boundary=constrain_to_boundary)
        return self

    def crop_to_true_mask(self, boundary=0, constrain_to_boundary=True):
        r"""
        Crop this image to be bounded just the `True` values of it's mask.

        Parameters
        ----------

        boundary: `int`, optional
            An extra padding to be added all around the true mask region.
        constrain_to_boundary : `bool`, optional
            If ``True`` the crop will be snapped to not go beyond this images
            boundary. If ``False``, an :map:`ImageBoundaryError` will be raised
            if an attempt is made to go beyond the edge of the image. Note that
            is only possible if ``boundary != 0``.

        Raises
        ------
        ImageBoundaryError
            Raised if 11constrain_to_boundary=False`1, and an attempt is
            made to crop the image in a way that violates the image bounds.
        """
        min_indices, max_indices = self.mask.bounds_true(
            boundary=boundary, constrain_to_bounds=False)
        # no point doing the bounds check twice - let the crop do it only.
        self.crop_inplace(min_indices, max_indices,
                          constrain_to_boundary=constrain_to_boundary)

    def warp_to_mask(self, template_mask, transform, warp_landmarks=False,
                     order=1, mode='constant', cval=0.):
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

        Returns
        -------
        warped_image : ``type(self)``
            A copy of this image, warped.
        """
        # call the super variant and get ourselves a MaskedImage back
        # with a blank mask
        warped_image = Image.warp_to_mask(self, template_mask, transform,
                                          warp_landmarks=warp_landmarks,
                                          order=order, mode=mode, cval=cval)
        warped_mask = self.mask.warp_to_mask(template_mask, transform,
                                             warp_landmarks=warp_landmarks,
                                             mode=mode, cval=cval)
        warped_image.mask = warped_mask
        return warped_image

    def warp_to_shape(self, template_shape, transform, warp_landmarks=False,
                      order=1, mode='constant', cval=0.):
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

        Returns
        -------
        warped_image : :map:`MaskedImage`
            A copy of this image, warped.
        """
        # call the super variant and get ourselves an Image back
        warped_image = Image.warp_to_shape(self, template_shape, transform,
                                           warp_landmarks=warp_landmarks,
                                           order=order, mode=mode, cval=cval)
        # warp the mask separately and reattach.
        mask = self.mask.warp_to_shape(template_shape, transform,
                                       warp_landmarks=warp_landmarks,
                                       mode=mode, cval=cval)
        # efficiently turn the Image into a MaskedImage, attaching the
        # landmarks
        masked_warped_image = MaskedImage(warped_image.pixels, mask=mask,
                                          copy=False)
        masked_warped_image.landmarks = warped_image.landmarks
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
        self._normalize_inplace(np.std, mode=mode,
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

        def scale_func(pixels, axis=None):
            return np.linalg.norm(pixels, axis=axis, **kwargs)

        self._normalize_inplace(scale_func, mode=mode,
                                limit_to_mask=limit_to_mask)

    def _normalize_inplace(self, scale_func, mode='all', limit_to_mask=True):
        if limit_to_mask:
            pixels = self.as_vector(keep_channels=True)
        else:
            pixels = Image.as_vector(self, keep_channels=True)
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
            normalized_pixels = centered_pixels / scale_factor

        if limit_to_mask:
            self.from_vector_inplace(normalized_pixels.flatten())
        else:
            Image.from_vector_inplace(self,
                                      normalized_pixels.flatten())

    def gradient(self, nullify_values_at_mask_boundaries=False):
        r"""
        Returns a :map:`MaskedImage` which is the gradient of this one. In the
        case of multiple channels, it returns the gradient over each axis over
        each channel as a flat list.

        Parameters
        ----------
        nullify_values_at_mask_boundaries : `bool`, optional
            If ``True`` a one pixel boundary is set to 0 around the edge of
            the ``True`` mask region. This is useful in situations where
            there is absent data in the image which will cause erroneous
            gradient settings.

        Returns
        -------
        gradient : :map:`MaskedImage`
            The gradient over each axis over each channel. Therefore, the
            gradient of a 2D, single channel image, will have length `2`.
            The length of a 2D, 3-channel image, will have length `6`.
        """
        global binary_erosion, gradient
        if gradient is None:
            from menpo.feature import gradient  # avoid circular reference
        # use the feature to take the gradient as normal
        grad_image = gradient(self)
        if nullify_values_at_mask_boundaries:
            if binary_erosion is None:
                from scipy.ndimage import binary_erosion  # expensive
            # Erode the edge of the mask in by one pixel
            eroded_mask = binary_erosion(self.mask.mask, iterations=1)

            # replace the eroded mask with the diff between the two
            # masks. This is only true in the region we want to nullify.
            np.logical_and(~eroded_mask, self.mask.mask, out=eroded_mask)
            # nullify all the boundary values in the grad image
            grad_image.pixels[eroded_mask] = 0.0
        return grad_image

    def constrain_mask_to_landmarks(self, group=None, label=None,
                                    trilist=None):
        r"""
        Restricts this image's mask to be equal to the convex hull around the
        landmarks chosen. This is not a per-pixel convex hull, but is instead
        estimated by a triangulation of the points that contain the convex
        hull.

        Parameters
        ----------
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `str`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        trilist: ``(t, 3)`` `ndarray`, optional
            Triangle list to be used on the landmarked points in selecting
            the mask region. If None defaults to performing Delaunay
            triangulation on the points.
        """
        self.mask.constrain_to_pointcloud(self.landmarks[group][label],
                                          trilist=trilist)

    def build_mask_around_landmarks(self, patch_size, group=None, label=None):
        r"""
        Restricts this images mask to be patches around each landmark in
        the chosen landmark group. This is useful for visualizing patch
        based methods.

        Parameters
        ----------
        patch_shape : `tuple`
            The size of the patch. Any floating point values are rounded up
            to the nearest integer.
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `str`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        """
        pc = self.landmarks[group][label]
        patch_size = np.ceil(patch_size)
        patch_half_size = patch_size / 2
        mask = np.zeros(self.shape)
        max_x = self.shape[0] - 1
        max_y = self.shape[1] - 1

        for i, point in enumerate(pc.points):
            start = np.floor(point - patch_half_size).astype(int)
            finish = np.floor(point + patch_half_size).astype(int)
            x, y = np.mgrid[start[0]:finish[0], start[1]:finish[1]]
            # deal with boundary cases
            x[x > max_x] = max_x
            y[y > max_y] = max_y
            x[x < 0] = 0
            y[y < 0] = 0
            mask[x.flatten(), y.flatten()] = True

        self.mask = BooleanImage(mask)
