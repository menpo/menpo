from collections import Iterable

import numpy as np
from menpo.base import MenpoMissingDependencyError


class Menpo3dMissingError(MenpoMissingDependencyError):
    r"""
    Exception that is thrown when an attempt is made to import a 3D
    visualisation method, but 'menpo3d' is not installed.
    """
    def __init__(self):
        super(Menpo3dMissingError, self).__init__('menpo3d')


class MenpowidgetsMissingError(MenpoMissingDependencyError):
    r"""
    Exception that is thrown when an attempt is made to import a widget, but
    'menpowidgets' is not installed.
    """
    def __init__(self):
        super(MenpowidgetsMissingError, self).__init__('menpowidgets')


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
            def raise_not_supported(*args, **kwargs):
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

        self._flip_image_channels()

    def _flip_image_channels(self):
        if self.pixels.ndim == 3:
            from menpo.image.base import channels_to_back
            self.pixels = channels_to_back(self.pixels)

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
        n_channels = pixels.shape[0]
        if channels is None:
            if n_channels == 1:
                pixels = pixels[0, ...]
                use_subplots = False
            elif n_channels == 3:
                use_subplots = False
        elif channels != 'all':
            if isinstance(channels, Iterable):
                if len(channels) == 1:
                    pixels = pixels[channels[0], ...]
                    use_subplots = False
                else:
                    pixels = pixels[channels, ...]
            else:
                pixels = pixels[channels, ...]
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
            pixels[..., ~mask] = nanmax + (0.01 * nanmax)
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
                         interpolation, cmap_name, alpha, render_lines,
                         line_colour, line_style, line_width,
                         render_markers, marker_style, marker_size,
                         marker_face_colour, marker_edge_colour,
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
                         axes_x_limits, axes_y_limits, axes_x_ticks,
                         axes_y_ticks, figure_size):
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

    # Parse axes limits
    image_axes_x_limits = None
    landmarks_axes_x_limits = axes_x_limits
    if axes_x_limits is None:
        image_axes_x_limits = landmarks_axes_x_limits = [0, image.width - 1]
    image_axes_y_limits = None
    landmarks_axes_y_limits = axes_y_limits
    if axes_y_limits is None:
        image_axes_y_limits = landmarks_axes_y_limits = [0, image.height - 1]

    # Render image
    from menpo.image import MaskedImage
    if isinstance(image, MaskedImage):
        self_view = image.view(figure_id=figure_id, new_figure=new_figure,
                               channels=channels, masked=masked,
                               interpolation=interpolation, cmap_name=cmap_name,
                               alpha=alpha, render_axes=render_axes,
                               axes_x_limits=image_axes_x_limits,
                               axes_y_limits=image_axes_y_limits)
    else:
        self_view = image.view(figure_id=figure_id, new_figure=new_figure,
                               channels=channels, interpolation=interpolation,
                               cmap_name=cmap_name, alpha=alpha,
                               render_axes=render_axes,
                               axes_x_limits=image_axes_x_limits,
                               axes_y_limits=image_axes_y_limits)

    # Render landmarks
    # correct group label in legend
    if group is None and image.landmarks.n_groups == 1:
        group = image.landmarks.group_labels[0]
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
            group=group, figure_id=self_view.figure_id, new_figure=False,
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
            axes_font_weight=axes_font_weight,
            axes_x_limits=landmarks_axes_x_limits,
            axes_y_limits=landmarks_axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

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


def plot_curve(x_axis, y_axis, figure_id=None, new_figure=True,
               legend_entries=None, title='', x_label='', y_label='',
               axes_x_limits=0., axes_y_limits=None, axes_x_ticks=None,
               axes_y_ticks=None, render_lines=True, line_colour=None,
               line_style='-', line_width=1, render_markers=True,
               marker_style='o', marker_size=6, marker_face_colour=None,
               marker_edge_colour='k', marker_edge_width=1., render_legend=True,
               legend_title='', legend_font_name='sans-serif',
               legend_font_style='normal', legend_font_size=10,
               legend_font_weight='normal', legend_marker_scale=None,
               legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
               legend_border_axes_pad=None, legend_n_columns=1,
               legend_horizontal_spacing=None, legend_vertical_spacing=None,
               legend_border=True, legend_border_padding=None,
               legend_shadow=False, legend_rounded_corners=False,
               render_axes=True, axes_font_name='sans-serif', axes_font_size=10,
               axes_font_style='normal', axes_font_weight='normal',
               figure_size=(10, 8), render_grid=True, grid_line_style='--',
               grid_line_width=1):
    r"""
    Plot a single or multiple curves on the same figure.

    Parameters
    ----------
    x_axis : `list` or `array`
        The values of the horizontal axis. They are common for all curves.
    y_axis : `list` of `lists` or `arrays`
        A `list` with `lists` or `arrays` with the values of the vertical axis
        for each curve.
    figure_id : `object`, optional
        The id of the figure to be used.
    new_figure : `bool`, optional
        If ``True``, a new figure is created.
    legend_entries : `list of `str` or ``None``, optional
        If `list` of `str`, it must have the same length as `errors` `list` and
        each `str` will be used to name each curve. If ``None``, the CED curves
        will be named as `'Curve %d'`.
    title : `str`, optional
        The figure's title.
    x_label : `str`, optional
        The label of the horizontal axis.
    y_label : `str`, optional
        The label of the vertical axis.
    axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
        The limits of the x axis. If `float`, then it sets padding on the
        right and left of the graph as a percentage of the curves' width. If
        `tuple` or `list`, then it defines the axis limits. If ``None``, then the
        limits are set automatically.
    axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
        The limits of the y axis. If `float`, then it sets padding on the
        top and bottom of the graph as a percentage of the curves' height. If
        `tuple` or `list`, then it defines the axis limits. If ``None``, then the
        limits are set automatically.
    axes_x_ticks : `list` or `tuple` or ``None``, optional
        The ticks of the x axis.
    axes_y_ticks : `list` or `tuple` or ``None``, optional
        The ticks of the y axis.
    render_lines : `bool` or `list` of `bool`, optional
        If ``True``, the line will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `y_axis`.
    line_colour : `colour` or `list` of `colour` or ``None``, optional
        The colour of the lines. If not a `list`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `y_axis`. If ``None``, the
        colours will be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    line_style : ``{'-', '--', '-.', ':'}`` or `list` of those, optional
        The style of the lines. If not a `list`, this value will be used for all
        curves. If `list`, a value must be specified for each curve, thus it must
        have the same length as `y_axis`.
    line_width : `float` or `list` of `float`, optional
        The width of the lines. If `float`, this value will be used for all
        curves. If `list`, a value must be specified for each curve, thus it must
        have the same length as `y_axis`.
    render_markers : `bool` or `list` of `bool`, optional
        If ``True``, the markers will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `y_axis`.
    marker_style : `marker` or `list` of `markers`, optional
        The style of the markers. If not a `list`, this value will be used for
        all curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `y_axis`.
        Example `marker` options ::

                {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                 'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

    marker_size : `int` or `list` of `int`, optional
        The size of the markers in points^2. If `int`, this value will be used
        for all curves. If `list`, a value must be specified for each curve, thus
        it must have the same length as `y_axis`.
    marker_face_colour : `colour` or `list` of `colour` or ``None``, optional
        The face (filling) colour of the markers. If not a `list`, this value
        will be used for all curves. If `list`, a value must be specified for
        each curve, thus it must have the same length as `y_axis`. If ``None``,
        the colours will be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    marker_edge_colour : `colour` or `list` of `colour` or ``None``, optional
        The edge colour of the markers. If not a `list`, this value will be used
        for all curves. If `list`, a value must be specified for each curve, thus
        it must have the same length as `y_axis`. If ``None``, the colours will
        be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    marker_edge_width : `float` or `list` of `float`, optional
        The width of the markers' edge. If `float`, this value will be used for
        all curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `y_axis`.
    render_legend : `bool`, optional
        If ``True``, the legend will be rendered.
    legend_title : `str`, optional
        The title of the legend.
    legend_font_name : See below, optional
        The font of the legend.
        Example options ::

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}
            
    legend_font_style : ``{'normal', 'italic', 'oblique'}``, optional
        The font style of the legend.
    legend_font_size : `int`, optional
        The font size of the legend.
    legend_font_weight : See below, optional
        The font weight of the legend.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

    legend_marker_scale : `float`, optional
        The relative size of the legend markers with respect to the original
    legend_location : `int`, optional
        The location of the legend. The predefined values are:

        =============== ===
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
        =============== ===

    legend_bbox_to_anchor : (`float`, `float`), optional
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
    axes_font_name : See below, optional
        The font of the axes.
        Example options ::

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

    axes_font_size : `int`, optional
        The font size of the axes.
    axes_font_style : ``{'normal', 'italic', 'oblique'}``, optional
        The font style of the axes.
    axes_font_weight : See below, optional
        The font weight of the axes.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

    figure_size : (`float`, `float`) or ``None``, optional
        The size of the figure in inches.
    render_grid : `bool`, optional
        If ``True``, the grid will be rendered.
    grid_line_style : ``{'-', '--', '-.', ':'}``, optional
        The style of the grid lines.
    grid_line_width : `float`, optional
        The width of the grid lines.

    Raises
    ------
    ValueError
        legend_entries list has different length than y_axis list

    Returns
    -------
    viewer : :map:`GraphPlotter`
        The viewer object.
    """
    from menpo.visualize import GraphPlotter

    # check y_axis
    if not isinstance(y_axis, list):
        y_axis = [y_axis]

    # check legend_entries
    if legend_entries is not None and len(legend_entries) != len(y_axis):
        raise ValueError('legend_entries list has different length than y_axis '
                         'list')

    # render
    return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                        x_axis=x_axis, y_axis=y_axis, title=title,
                        legend_entries=legend_entries, x_label=x_label,
                        y_label=y_label, x_axis_limits=axes_x_limits,
                        y_axis_limits=axes_y_limits, x_axis_ticks=axes_x_ticks,
                        y_axis_ticks=axes_y_ticks).render(
        render_lines=render_lines, line_colour=line_colour,
        line_style=line_style, line_width=line_width,
        render_markers=render_markers, marker_style=marker_style,
        marker_size=marker_size, marker_face_colour=marker_face_colour,
        marker_edge_colour=marker_edge_colour,
        marker_edge_width=marker_edge_width, render_legend=render_legend,
        legend_title=legend_title, legend_font_name=legend_font_name,
        legend_font_style=legend_font_style, legend_font_size=legend_font_size,
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
        legend_rounded_corners=legend_rounded_corners, render_axes=render_axes,
        axes_font_name=axes_font_name, axes_font_size=axes_font_size,
        axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
        figure_size=figure_size, render_grid=render_grid,
        grid_line_style=grid_line_style, grid_line_width=grid_line_width)


def render_rectangles_around_patches(centers, patch_shape, axes=None,
                                     image_view=True, line_colour='r',
                                     line_style='-', line_width=1,
                                     interpolation='none'):
    r"""
    Method that renders rectangles of the specified `patch_shape` centered
    around all the points of the provided `centers`.

    Parameters
    ----------
    centers : :map:`PointCloud`
        The centers around which to draw the rectangles.
    patch_shape : `tuple` or `ndarray`, optional
        The size of the rectangle to render.
    axes : `matplotlib.pyplot.axes` object or ``None``, optional
        The axes object on which to render.
    image_view : `bool`, optional
        If ``True`` the rectangles will be viewed as if they are in the image
        coordinate system.
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
    interpolation : See Below, optional
        In case a patch-based image is already rendered on the specified axes,
        this argument controls how tight the rectangles would be to the patches.
        It needs to have the same value as the one used when rendering the
        patches image, otherwise there is the danger that the rectangles won't
        be exactly on the border of the patches. Example options ::

            {none, nearest, bilinear, bicubic, spline16, spline36,
            hanning, hamming, hermite, kaiser, quadric, catrom, gaussian,
            bessel, mitchell, sinc, lanczos}

    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Dictionary with the line styles
    line_style_dict = {'-': 'solid', '--': 'dashed', '-.': 'dashdot',
                       ':': 'dotted'}

    # Get axes object
    if axes is None:
        axes = plt.gca()

    # Need those in order to compute the lower left corner of the rectangle
    half_patch_shape = [patch_shape[0] / 2,
                        patch_shape[1] / 2]

    # Set the view mode
    if image_view:
        xi = 1
        yi = 0
    else:
        xi = 0
        yi = 1

    # Set correct offsets so that the rectangle is tight to the patch
    if interpolation == 'none':
        off_start = 0.5
        off_end = 0.
    else:
        off_start = 1.
        off_end = 0.5

    # Render rectangles
    for p in range(centers.shape[0]):
        xc = np.intp(centers[p, xi] - half_patch_shape[xi]) - off_start
        yc = np.intp(centers[p, yi] - half_patch_shape[yi]) - off_start
        axes.add_patch(Rectangle((xc, yc),
                                 patch_shape[xi] + off_end,
                                 patch_shape[yi] + off_end,
                                 fill=False, edgecolor=line_colour,
                                 linewidth=line_width,
                                 linestyle=line_style_dict[line_style]))


def view_patches(patches, patch_centers, patches_indices=None,
                 offset_index=None, figure_id=None, new_figure=False,
                 background='white', render_patches=True, channels=None,
                 interpolation='none', cmap_name=None, alpha=1.,
                 render_patches_bboxes=True, bboxes_line_colour='r',
                 bboxes_line_style='-', bboxes_line_width=1,
                 render_centers=True, render_lines=True, line_colour=None,
                 line_style='-', line_width=1, render_markers=True,
                 marker_style='o', marker_size=20, marker_face_colour=None,
                 marker_edge_colour=None, marker_edge_width=1.,
                 render_numbering=False, numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_axes=False,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(10, 8)):
    r"""
    Method that renders the provided `patches` on a black canvas. The user can
    choose whether to render the patch centers (`render_centers`) as well as
    rectangle boundaries around the patches (`render_patches_bboxes`).

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
        The centers around which to visualize the patches.
    patches_indices : `int` or `list` of `int` or ``None``, optional
        Defines the patches that will be visualized. If ``None``, then all the
        patches are selected.
    offset_index : `int` or ``None``, optional
        The offset index within the provided `patches` argument, thus the index
        of the second dimension from which to sample. If ``None``, then ``0`` is
        used.
    figure_id : `object`, optional
        The id of the figure to be used.
    new_figure : `bool`, optional
        If ``True``, a new figure is created.
    background : ``{'black', 'white'}``, optional
        If ``'black'``, then the background is set equal to the minimum value
        of `patches`. If ``'white'``, then the background is set equal to the
        maximum value of `patches`.
    render_patches : `bool`, optional
        Flag that determines whether to render the patch values.
    channels : `int` or `list` of `int` or ``all`` or ``None``, optional
        If `int` or `list` of `int`, the specified channel(s) will be
        rendered. If ``all``, all the channels will be rendered in subplots.
        If ``None`` and the image is RGB, it will be rendered in RGB mode.
        If ``None`` and the image is not RGB, it is equivalent to ``all``.
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
    render_patches_bboxes : `bool`, optional
        Flag that determines whether to render the bounding box lines around the
        patches.
    bboxes_line_colour : See Below, optional
        The colour of the lines.
        Example options::

            {r, g, b, c, m, k, w}
            or
            (3, ) ndarray
    bboxes_line_style : ``{-, --, -., :}``, optional
        The style of the lines.
    bboxes_line_width : `float`, optional
        The width of the lines.
    render_centers : `bool`, optional
        Flag that determines whether to render the patch centers.
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
        right and left of the shape as a percentage of the shape's width. If
        `tuple` or `list`, then it defines the axis limits. If ``None``, then the
        limits are set automatically.
    axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
        The limits of the y axis. If `float`, then it sets padding on the
        top and bottom of the shape as a percentage of the shape's height. If
        `tuple` or `list`, then it defines the axis limits. If ``None``, then the
        limits are set automatically.
    axes_x_ticks : `list` or `tuple` or ``None``, optional
        The ticks of the x axis.
    axes_y_ticks : `list` or `tuple` or ``None``, optional
        The ticks of the y axis.
    figure_size : (`float`, `float`) `tuple` or ``None`` optional
        The size of the figure in inches.

    Returns
    -------
    viewer : `ImageViewer`
        The image viewing object.
    """
    from menpo.image.base import (_convert_patches_list_to_single_array,
                                  _create_patches_image)

    # If patches is a list, convert it to an array
    if isinstance(patches, list):
        patches = _convert_patches_list_to_single_array(patches,
                                                        patch_centers.n_points)

    # Create patches image
    if render_patches:
        patches_image = _create_patches_image(
            patches, patch_centers, patches_indices=patches_indices,
            offset_index=offset_index, background=background)
    else:
        if background == 'black':
            tmp_patches = np.zeros((patches.shape[0], patches.shape[1], 3,
                                    patches.shape[3], patches.shape[4]))
        elif background == 'white':
            tmp_patches = np.ones((patches.shape[0], patches.shape[1], 3,
                                   patches.shape[3], patches.shape[4]))
        patches_image = _create_patches_image(
            tmp_patches, patch_centers, patches_indices=patches_indices,
            offset_index=offset_index, background=background)
        channels = None

    # Render patches image
    if render_centers:
        patch_view = patches_image.view_landmarks(
            channels=channels, group='all_patch_centers', figure_id=figure_id,
            new_figure=new_figure, interpolation=interpolation,
            cmap_name=cmap_name, alpha=alpha, render_lines=render_lines,
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
            render_legend=False, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            figure_size=figure_size)
    else:
        patch_view = patches_image.view(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    # Render rectangles around patches
    if render_patches_bboxes:
        patch_shape = [patches.shape[3], patches.shape[4]]
        render_rectangles_around_patches(
            patches_image.landmarks['selected_patch_centers'].lms.points,
            patch_shape, image_view=True, line_colour=bboxes_line_colour,
            line_style=bboxes_line_style, line_width=bboxes_line_width,
            interpolation=interpolation)

    return patch_view


def plot_gaussian_ellipses(covariances, means, n_std=2, render_colour_bar=True,
                           colour_bar_label='Normalized Standard Deviation',
                           colour_map='jet', figure_id=None, new_figure=False,
                           image_view=True, line_colour='r', line_style='-',
                           line_width=1., render_markers=True,
                           marker_edge_colour='k', marker_face_colour='k',
                           marker_edge_width=1., marker_size=20,
                           marker_style='o', render_axes=False,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           crop_proportion=0.1, figure_size=(10, 8)):
    r"""
    Method that renders the Gaussian ellipses that correspond to a set of
    covariance matrices and mean vectors. Naturally, this only works for
    2-dimensional random variables.

    Parameters
    ----------
    covariances : `list` of ``(2, 2)`` `ndarray`
        The covariance matrices that correspond to each ellipse.
    means : `list` of ``(2, )`` `ndarray`
        The mean vectors that correspond to each ellipse.
    n_std : `float`, optional
        This defines the size of the ellipses in terms of number of standard
        deviations.
    render_colour_bar : `bool`, optional
        If ``True``, then the ellipses will be coloured based on their
        normalized standard deviations and a colour bar will also appear on
        the side. If ``False``, then all the ellipses will have the same colour.
    colour_bar_label : `str`, optional
        The title of the colour bar. It only applies if `render_colour_bar`
        is ``True``.
    colour_map : `str`, optional
        A valid Matplotlib colour map. For more info, please refer to
        `matplotlib.cm`.
    figure_id : `object`, optional
        The id of the figure to be used.
    new_figure : `bool`, optional
        If ``True``, a new figure is created.
    image_view : `bool`, optional
        If ``True`` the ellipses will be rendered in the image coordinates
        system.
    line_colour : See Below, optional
        The colour of the lines of the ellipses.
        Example options::

            {r, g, b, c, m, k, w}
            or
            (3, ) ndarray

    line_style : ``{-, --, -., :}``, optional
        The style of the lines of the ellipses.
    line_width : `float`, optional
        The width of the lines of the ellipses.
    render_markers : `bool`, optional
        If ``True``, the centers of the ellipses will be rendered.
    marker_style : See Below, optional
        The style of the centers of the ellipses. Example options ::

            {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

    marker_size : `int`, optional
        The size of the centers of the ellipses in points^2.
    marker_face_colour : See Below, optional
        The face (filling) colour of the centers of the ellipses.
        Example options ::

            {r, g, b, c, m, k, w}
            or
            (3, ) ndarray

    marker_edge_colour : See Below, optional
        The edge colour of the centers of the ellipses.
        Example options ::

            {r, g, b, c, m, k, w}
            or
            (3, ) ndarray

    marker_edge_width : `float`, optional
        The edge width of the centers of the ellipses.
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

    crop_proportion : `float`, optional
        The proportion to be left around the centers' pointcloud.
    figure_size : (`float`, `float`) `tuple` or ``None`` optional
        The size of the figure in inches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.font_manager import FontProperties
    from menpo.shape import PointCloud

    def eigh_sorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    # get correct line style
    if line_style == '-':
        line_style = 'solid'
    elif line_style == '--':
        line_style = 'dashed'
    elif line_style == '-.':
        line_style = 'dashdot'
    elif line_style == ':':
        line_style = 'dotted'
    else:
        raise ValueError("line_style must be selected from "
                         "['-', '--', '-.', ':'].")

    # create pointcloud
    pc = PointCloud(np.array(means))

    # compute axes limits
    bounds = pc.bounds()
    r = pc.range()
    x_rr = r[0] * crop_proportion
    y_rr = r[1] * crop_proportion
    axes_x_limits=[bounds[0][1] - x_rr, bounds[1][1] + x_rr]
    axes_y_limits=[bounds[0][0] - y_rr, bounds[1][0] + y_rr]
    normalizer = np.sum(r) / 2.

    # compute height, width, theta and std
    stds = []
    heights = []
    widths = []
    thetas = []
    for cov in covariances:
        vals, vecs = eigh_sorted(cov)
        width, height = np.sqrt(vals)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        stds.append(np.mean([height, width]) / normalizer)
        heights.append(height)
        widths.append(width)
        thetas.append(theta)

    if render_colour_bar:
        # set colormap values
        cmap = plt.get_cmap(colour_map)
        cNorm = colors.Normalize(vmin=np.min(stds), vmax=np.max(stds))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # visualize pointcloud
    if render_colour_bar:
        renderer = pc.view(
            figure_id=figure_id, new_figure=new_figure, image_view=image_view,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, figure_size=figure_size,
            render_markers=False)
    else:
        renderer = pc.view(
            figure_id=figure_id, new_figure=new_figure, image_view=image_view,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            marker_edge_width=marker_edge_width, marker_size=marker_size,
            marker_style=marker_style, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            figure_size=figure_size, render_markers=render_markers)

    # plot ellipses
    ax = plt.gca()
    for i in range(len(covariances)):
        # Width and height are "full" widths, not radius
        width = 2 * n_std * widths[i]
        height = 2 * n_std * heights[i]

        if image_view:
            colour = line_colour
            if render_colour_bar:
                colour = scalarMap.to_rgba(stds[i])
                if render_markers:
                    plt.scatter(means[i][1], means[i][0], facecolor=colour,
                                edgecolor=colour)
            ellip = Ellipse(xy=means[i][-1::-1], width=height, height=width,
                            angle=thetas[i], linestyle=line_style,
                            linewidth=line_width, edgecolor=colour,
                            facecolor='none')
        else:
            colour = line_colour
            if render_colour_bar:
                colour = scalarMap.to_rgba(stds[i])
                if render_markers:
                    plt.scatter(means[i][0], means[i][1], facecolor=colour,
                                edgecolor=colour)
            ellip = Ellipse(xy=means[i], width=width, height=height,
                            angle=thetas[i], linestyle=line_style,
                            linewidth=line_width, edgecolor=colour,
                            facecolor='none')
        ax.add_artist(ellip)

    # show colour bar
    if render_colour_bar:
        scalarMap.set_array(stds)
        cb = plt.colorbar(scalarMap, label=colour_bar_label)

        # change colour bar's font properties
        ax = cb.ax
        text = ax.yaxis.label
        font = FontProperties(size=axes_font_size, weight=axes_font_weight,
                              style=axes_font_style, family=axes_font_name)
        text.set_font_properties(font)

    return renderer
