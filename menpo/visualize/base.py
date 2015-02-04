from collections import Iterable

import numpy as np


Menpo3dErrorMessage = ("In order to keep menpo's dependencies simple, menpo "
                       "does not contain 3D importing and visualization code. "
                       "Please install menpo3d to view 3D meshes.")


class Renderer(object):
    r"""
    Abstract class for rendering visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.

    It is assumed that the renderers follow some form of stateful pattern for
    rendering to Figures. Therefore, the major interface for rendering involves
    providing a `figure_id` or a `bool` about whether a new figure should be
    used. If neither are provided then the default state of the rendering engine
    is assumed to be maintained.

    Providing both a ``figure_id`` and ``new_figure == True`` is not a valid
    state.

    Parameters
    ----------
    figure_id : `object`
        A figure id. Could be any valid object that identifies a figure in a
        given framework (`str`, `int`, `float`, etc.).
    new_figure : `bool`
        Whether the rendering engine should create a new figure.

    Raises
    ------
    ValueError
        It is not valid to provide a figure id AND request a new figure to
        be rendered on.
    """

    def __init__(self, figure_id, new_figure):
        if figure_id is not None and new_figure:
            raise ValueError("Conflicting arguments. figure_id cannot be "
                             "specified if the new_figure flag is True")

        self.figure_id = figure_id
        self.new_figure = new_figure
        self.figure = self.get_figure()

    def render(self, **kwargs):
        r"""
        Abstract method to be overridden by the renderer. This will implement
        the actual rendering code for a given object class.

        Parameters
        ----------
        kwargs : `dict`
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :map:`Renderer`
            Pointer to `self`.
        """
        pass

    def get_figure(self):
        r"""
        Abstract method for getting the correct figure to render on. Should
        also set the correct `figure_id` for the figure.

        Returns
        -------
        figure : `object`
            The figure object that the renderer will render on.
        """
        pass

    def save_figure(self, **kwargs):
        r"""
        Abstract method for saving the figure of the current `figure_id` to
        file. It will implement the actual saving code for a given object class.

        Parameters
        ----------
        kwargs : `dict`
            Options to be set when saving the figure to file.
        """
        pass


class viewwrapper(object):
    r"""
    This class abuses the Python descriptor protocol in order to dynamically
    change the view method at runtime. Although this is more obviously achieved
    through inheritance, the view methods practically amount to syntactic sugar
    and so we want to maintain a single view method per class. We do not want
    to add the mental overhead of implementing different 2D and 3D PointCloud
    classes for example, since, outside of viewing, their implementations would
    be identical.

    Also note that we could have separated out viewing entirely and made the
    check there, but the view method is an important paradigm in menpo that
    we want to maintain.

    Therefore, this function cleverly (and obscurely) returns the correct
    view method for the dimensionality of the given object.
    """

    def __init__(self, wrapped_func):
        fname = wrapped_func.__name__
        self._2d_fname = '_{}_2d'.format(fname)
        self._3d_fname = '_{}_3d'.format(fname)

    def __get__(self, instance, instancetype):
        if instance.n_dims == 2:
            return getattr(instance, self._2d_fname)
        elif instance.n_dims == 3:
            return getattr(instance, self._3d_fname)
        else:
            def raise_not_supported(self):
                r"""
                Viewing of objects with greater than 3 dimensions is not
                currently possible.
                """
                raise ValueError('Viewing of objects with greater than 3 '
                                 'dimensions is not currently possible.')
            return raise_not_supported


class Viewable(object):
    r"""
    Abstract interface for objects that can visualize themselves. This assumes
    that the class has dimensionality as the view method checks the ``n_dims``
    property to wire up the correct view method.
    """

    @viewwrapper
    def view(self):
        r"""
        Abstract method for viewing. See the :map:`viewwrapper` documentation
        for an explanation of how the `view` method works.
        """
        pass

    def _view_2d(self, **kwargs):
        raise NotImplementedError('2D Viewing is not supported.')

    def _view_3d(self, **kwargs):
        raise NotImplementedError('3D Viewing is not supported.')


class LandmarkableViewable(object):
    r"""
    Mixin for :map:`Landmarkable` and :map:`Viewable` objects. Provides a
    single helper method for viewing Landmarks and `self` on the same figure.
    """

    @viewwrapper
    def view_landmarks(self, **kwargs):
        pass

    def _view_landmarks_2d(self, **kwargs):
        raise NotImplementedError('2D Landmark Viewing is not supported.')

    def _view_landmarks_3d(self, **kwargs):
        raise NotImplementedError('3D Landmark Viewing is not supported.')


from menpo.visualize.viewmatplotlib import (
    MatplotlibImageViewer2d, MatplotlibImageSubplotsViewer2d,
    MatplotlibLandmarkViewer2d, MatplotlibAlignmentViewer2d,
    MatplotlibGraphPlotter, MatplotlibMultiImageViewer2d,
    MatplotlibMultiImageSubplotsViewer2d, MatplotlibPointGraphViewer2d)

# Default importer types
PointGraphViewer2d = MatplotlibPointGraphViewer2d
LandmarkViewer2d = MatplotlibLandmarkViewer2d
ImageViewer2d = MatplotlibImageViewer2d
ImageSubplotsViewer2d = MatplotlibImageSubplotsViewer2d

AlignmentViewer2d = MatplotlibAlignmentViewer2d
GraphPlotter = MatplotlibGraphPlotter
MultiImageViewer2d = MatplotlibMultiImageViewer2d
MultiImageSubplotsViewer2d = MatplotlibMultiImageSubplotsViewer2d


class ImageViewer(object):
    r"""
    Base :map:`Image` viewer that abstracts away dimensionality. It can
    visualize multiple channels of an image in subplots.

    Parameters
    ----------
    figure_id : `object`
        A figure id. Could be any valid object that identifies a figure in a
        given framework (`str`, `int`, `float`, etc.).
    new_figure : `bool`
        Whether the rendering engine should create a new figure.
    dimensions : {``2``, ``3``} `int`
        The number of dimensions in the image.
    pixels : ``(N, D)`` `ndarray`
        The pixels to render.
    channels: `int` or `list` or ``'all'`` or `None`
        A specific selection of channels to render. The user can choose either
        a single or multiple channels. If ``'all'``, render all channels in
        subplot mode. If `None` and image is not greyscale or RGB, render all
        channels in subplots. If `None` and image is greyscale or RGB, then do
        not plot channels in different subplots.
    mask: ``(N, D)`` `ndarray`
        A `bool` mask to be applied to the image. All points outside the
        mask are set to ``0``.
    """

    def __init__(self, figure_id, new_figure, dimensions, pixels,
                 channels=None, mask=None):
        pixels = pixels.copy()
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.dimensions = dimensions
        pixels, self.use_subplots = \
            self._parse_channels(channels, pixels)
        self.pixels = self._masked_pixels(pixels, mask)

    def _parse_channels(self, channels, pixels):
        r"""
        Parse `channels` parameter. If `channels` is `int` or `list`, keep it as
        is. If `channels` is ``'all'``, return a `list` of all the image's
        channels. If `channels` is `None`, return the minimum between an
        `upper_limit` and the image's number of channels. If image is greyscale
        or RGB and `channels` is `None`, then do not plot channels in different
        subplots.

        Parameters
        ----------
        channels : `int` or `list` or ``'all'`` or `None`
            A specific selection of channels to render.
        pixels : ``(N, D)`` `ndarray`
            The image's pixels to render.

        Returns
        -------
        pixels : ``(N, D)`` `ndarray`
            The pixels to be visualized.
        use_subplots : `bool`
            Whether to visualize using subplots.
        """
        # Flag to trigger ImageSubplotsViewer2d or ImageViewer2d
        use_subplots = True
        n_channels = pixels.shape[2]
        if channels is None:
            if n_channels == 1:
                pixels = pixels[..., 0]
                use_subplots = False
            elif n_channels == 3:
                use_subplots = False
        elif channels != 'all':
            if isinstance(channels, Iterable):
                if len(channels) == 1:
                    pixels = pixels[..., channels[0]]
                    use_subplots = False
                else:
                    pixels = pixels[..., channels]
            else:
                pixels = pixels[..., channels]
                use_subplots = False

        return pixels, use_subplots

    def _masked_pixels(self, pixels, mask):
        r"""
        Return the masked pixels using a given `bool` mask. In order to make
        sure that the non-masked pixels are visualized in white, their value
        is set to the maximum of pixels.

        Parameters
        ----------
        pixels : ``(N, D)`` `ndarray`
            The image's pixels to render.
        mask: ``(N, D)`` `ndarray`
            A `bool` mask to be applied to the image. All points outside the
            mask are set to the image max. If mask is `None`, then the initial
            pixels are returned.

        Returns
        -------
        masked_pixels : ``(N, D)`` `ndarray`
            The masked pixels.
        """
        if mask is not None:
            nanmax = np.nanmax(pixels)
            pixels[~mask] = nanmax + (0.01 * nanmax)
        return pixels

    def render(self, **kwargs):
        r"""
        Select the correct type of image viewer for the given image
        dimensionality.

        Parameters
        ----------
        kwargs : `dict`
            Passed through to image viewer.

        Returns
        -------
        viewer : :map:`Renderer`
            The rendering object.

        Raises
        ------
        ValueError
            Only 2D images are supported.
        """
        if self.dimensions == 2:
            if self.use_subplots:
                return ImageSubplotsViewer2d(self.figure_id, self.new_figure,
                                             self.pixels).render(**kwargs)
            else:
                return ImageViewer2d(self.figure_id, self.new_figure,
                                     self.pixels).render(**kwargs)
        else:
            raise ValueError("Only 2D images are currently supported")


def view_image_landmarks(image, channels, masked, group,
                         with_labels, without_labels, figure_id, new_figure,
                         interpolation, alpha, render_lines, line_colour,
                         line_style, line_width, render_markers, marker_style,
                         marker_size, marker_face_colour, marker_edge_colour,
                         marker_edge_width, render_numbering,
                         numbers_horizontal_align, numbers_vertical_align,
                         numbers_font_name, numbers_font_size,
                         numbers_font_style, numbers_font_weight,
                         numbers_font_colour, render_legend, legend_title,
                         legend_font_name, legend_font_style, legend_font_size,
                         legend_font_weight, legend_marker_scale,
                         legend_location, legend_bbox_to_anchor,
                         legend_border_axes_pad, legend_n_columns,
                         legend_horizontal_spacing, legend_vertical_spacing,
                         legend_border, legend_border_padding, legend_shadow,
                         legend_rounded_corners, render_axes, axes_font_name,
                         axes_font_size, axes_font_style, axes_font_weight,
                         axes_x_limits, axes_y_limits, figure_size):
    r"""
    This is a helper method that abstracts away the fact that viewing
    images and masked images is identical apart from the mask. Therefore,
    we do the class check in this method and then proceed identically whether
    the image is masked or not.

    See the documentation for _view_2d on Image or _view_2d on MaskedImage
    for information about the parameters.
    """
    import matplotlib.pyplot as plt

    if not image.has_landmarks:
        raise ValueError('Image does not have landmarks attached, unable '
                         'to view landmarks.')

    # Render self
    from menpo.image import MaskedImage
    if isinstance(image, MaskedImage):
        self_view = image.view(figure_id=figure_id, new_figure=new_figure,
                               channels=channels, masked=masked,
                               interpolation=interpolation, alpha=alpha)
    else:
        self_view = image.view(figure_id=figure_id, new_figure=new_figure,
                               channels=channels,
                               interpolation=interpolation, alpha=alpha)

    # Make sure axes are constrained to the image size
    if axes_x_limits is None:
        axes_x_limits = [0, image.width - 1]
    if axes_y_limits is None:
        axes_y_limits = [0, image.height - 1]

    # Render landmarks
    landmark_view = None  # initialize viewer object
    # useful in order to visualize the legend only for the last axis object
    render_legend_tmp = False
    for i, ax in enumerate(self_view.axes_list):
        # set current axis
        plt.sca(ax)
        # show legend only for the last axis object
        if i == len(self_view.axes_list) - 1:
            render_legend_tmp = render_legend

        # viewer
        landmark_view = image.landmarks[group].view(
            with_labels=with_labels, without_labels=without_labels,
            figure_id=self_view.figure_id, new_figure=False,
            image_view=True, render_lines=render_lines,
            line_colour=line_colour, line_style=line_style,
            line_width=line_width, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend_tmp, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
            legend_font_weight=legend_font_weight,
            legend_marker_scale=legend_marker_scale,
            legend_location=legend_location,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_border_axes_pad=legend_border_axes_pad,
            legend_n_columns=legend_n_columns,
            legend_horizontal_spacing=legend_horizontal_spacing,
            legend_vertical_spacing=legend_vertical_spacing,
            legend_border=legend_border,
            legend_border_padding=legend_border_padding,
            legend_shadow=legend_shadow,
            legend_rounded_corners=legend_rounded_corners,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, figure_size=figure_size)

    return landmark_view


class MultipleImageViewer(ImageViewer):

    def __init__(self, figure_id, new_figure, dimensions, pixels_list,
                 channels=None, mask=None):
        super(MultipleImageViewer, self).__init__(
            figure_id, new_figure, dimensions, pixels_list[0],
            channels=channels, mask=mask)
        pixels_list = [self._parse_channels(channels, p)[0]
                       for p in pixels_list]
        self.pixels_list = [self._masked_pixels(p, mask)
                            for p in pixels_list]

    def render(self, **kwargs):
        if self.dimensions == 2:
            if self.use_subplots:
                MultiImageSubplotsViewer2d(self.figure_id, self.new_figure,
                                           self.pixels_list).render(**kwargs)
            else:
                return MultiImageViewer2d(self.figure_id, self.new_figure,
                                          self.pixels_list).render(**kwargs)
        else:
            raise ValueError("Only 2D images are currently supported")
