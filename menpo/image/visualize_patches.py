from collections import Iterable

import numpy as np

from menpo.shape import PointCloud
from .base import Image, convert_patches_list_to_single_array


def create_patches_image(patches, patch_centers, patches_indices=None,
                         offset_index=None):
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

        1. ``(n_center, n_offset, self.n_channels, patch_size)`` `ndarray`
        2. `list` of ``n_center * n_offset`` :map:`Image` objects

    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches. It can have any of the two formats that are
        returned from the `extract_patches()` and
        `extract_patches_around_landmarks()` methods. Specifically, it can
        either be an ``(n_center, n_offset, self.n_channels, patch_size)``
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

    Returns
    -------
    patches_image : :map:`Image`
        The output patches image object.
    """
    # If patches is a list, convert it to array
    if isinstance(patches, list):
        patches = convert_patches_list_to_single_array(patches,
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

    # Create black new image and attach the corrected patch centers
    patches_image = Image.init_blank((height, width), n_channels)
    patches_image.landmarks['all_patch_centers'] = new_patch_centers
    patches_image.landmarks['selected_patch_centers'] = tmp_centers

    # Set the patches
    patches_image.set_patches_around_landmarks(patches[patches_indices],
                                               group='selected_patch_centers',
                                               offset_index=offset_index)

    return patches_image


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
                 channels=None, interpolation='none', cmap_name=None, alpha=1.,
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
                 axes_x_limits=None, axes_y_limits=None, figure_size=(10, 8)):
    r"""
    Method that renders the provided `patches` on a black canvas. The user can
    choose whether to render the patch centers (`render_centers`) as well as
    rectangle boundaries around the patches (`render_patches_bboxes`).

    The patches argument can have any of the two formats that are returned
    from the `extract_patches()` and `extract_patches_around_landmarks()`
    methods of the :map:`Image` class. Specifically it can be:

        1. ``(n_center, n_offset, self.n_channels, patch_size)`` `ndarray`
        2. `list` of ``n_center * n_offset`` :map:`Image` objects

    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches. It can have any of the two formats that are
        returned from the `extract_patches()` and
        `extract_patches_around_landmarks()` methods. Specifically, it can
        either be an ``(n_center, n_offset, self.n_channels, patch_size)``
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

    axes_x_limits : (`float`, `float`) `tuple` or ``None`` optional
        The limits of the x axis.
    axes_y_limits : (`float`, `float`) `tuple` or ``None`` optional
        The limits of the y axis.
    figure_size : (`float`, `float`) `tuple` or ``None`` optional
        The size of the figure in inches.

    Returns
    -------
    viewer : `ImageViewer`
        The image viewing object.
    """
    # If patches is a list, convert it to an array
    if isinstance(patches, list):
        patches = convert_patches_list_to_single_array(patches,
                                                       patch_centers.n_points)

    # Create patches image
    patches_image = create_patches_image(patches, patch_centers,
                                         patches_indices=patches_indices,
                                         offset_index=offset_index)

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
            figure_size=figure_size)
    else:
        patch_view = patches_image.view(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, figure_size=figure_size)

    # Render rectangles around patches
    if render_patches_bboxes:
        patch_shape = [patches.shape[3], patches.shape[4]]
        render_rectangles_around_patches(
            patches_image.landmarks['selected_patch_centers'].lms.points,
            patch_shape, image_view=True, line_colour=bboxes_line_colour,
            line_style=bboxes_line_style, line_width=bboxes_line_width,
            interpolation=interpolation)

    return patch_view
