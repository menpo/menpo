import collections
import numpy as np

from menpo.shape import bounding_box, TriMesh
from menpo.image import Image
from menpo.compatibility import basestring


def check_param(n, types, param_name, param):
    r"""
    Utility method for validating a given parameter. Since parameters may be
    a single option applied to all items in the list, or a list of different
    options per item, we validate this. If a single argument is passed, it
    is extended into a list.

    Parameters
    ----------
    n : `int`
        Number of items expected.
    types : `type`
        Python `type` or list of `types` that the parameters are expected to be.
    param_name : `str`
        The name of parameter for printing error messages.
    param : `object`
        An object or list of objects that represent a parameter for a
        function.

    Returns
    -------
    param_list : `list` of type `object`
        This is a validated list of length ``n``, one parameter for each
        element in the implicit list of other parameters.
    """
    error_msg = "{0} must be a {1} or a list/tuple of " \
                "{1} with length {2}".format(param_name, types, n)

    # Could be a single value - or we have an error
    if isinstance(param, types):
        return [param] * n
    elif not isinstance(param, collections.Iterable):
        raise ValueError(error_msg)

    # Must be an iterable object
    len_param = len(param)
    isinstance_all_in_param = all(isinstance(p, types) for p in param)

    if len_param == 1 and isinstance_all_in_param:
        return list(param) * n
    elif len_param == n and isinstance_all_in_param:
        return list(param)
    else:
        raise ValueError(error_msg)


def _rasterize_matplotlib(image, pclouds, render_lines=True, line_style='-',
                          line_colour='b', line_width=1, render_markers=True,
                          marker_style='o', marker_size=1,
                          marker_face_colour='b', marker_edge_colour='b',
                          marker_edge_width=1):
    import matplotlib.pyplot as plt

    # Convert image shape into 100 DPI inches
    # This makes sure we maintain the original image size
    image_shape = np.array(image.shape)[::-1] / 100.0
    f = plt.figure(figsize=image_shape, frameon=False, dpi=100)

    image.view(figure_id=f.number, figure_size=image_shape)
    for k, p in enumerate(pclouds):
        p.view(figure_id=f.number, render_axes=False, figure_size=image_shape,
               render_lines=render_lines[k], line_style=line_style[k],
               line_colour=line_colour[k], line_width=line_width[k],
               render_markers=render_markers[k], marker_style=marker_style[k],
               marker_size=marker_size[k],
               marker_face_colour=marker_face_colour[k],
               marker_edge_colour=marker_edge_colour[k],
               marker_edge_width=marker_edge_width[k])

    # Make sure the layout is tight so that the image is of the original size
    f.tight_layout(pad=0)
    # Get the pixels directly from the canvas buffer which is fast
    c_buffer, shape = f.canvas.print_to_buffer()
    # Turn buffer into numpy array and reshape to image
    pixels_buffer = np.array(c_buffer).reshape(shape[::-1] + (-1,))
    # Prevent matplotlib from rendering
    plt.close(f)
    # Ignore the Alpha channel
    return Image.init_from_rolled_channels(pixels_buffer[..., :3])


def _parse_colour(x):
    if isinstance(x, basestring):
        from matplotlib.colors import ColorConverter
        c = ColorConverter()
        x = c.to_rgb(x)

    if isinstance(x, (tuple, list)):
        # Assume we have a floating point rgb
        if any(a <= 1.0 for a in x):
            x = [int(a * 255) for a in x]
    return tuple(x)


def _rasterize_pillow(image, pclouds, render_lines=True, line_style='-',
                      line_colour='b', line_width=1, render_markers=True,
                      marker_style='o', marker_size=1, marker_face_colour='b',
                      marker_edge_colour='b', marker_edge_width=1):
    from PIL import ImageDraw

    if any(x != '-'for x in line_style):
        raise ValueError("The Pillow rasterizer only supports the '-' "
                         "line style.")
    if any(x not in {'o', 's'} for x in marker_style):
        raise ValueError("The Pillow rasterizer only supports the 'o' and 's' "
                         "marker styles.")
    if any(x > 1 for x in marker_edge_width):
        raise ValueError('The Pillow rasterizer only supports '
                         'marker_edge_width of 1 or 0.')

    pil_im = image.as_PILImage()
    draw = ImageDraw.Draw(pil_im)

    line_colour = [_parse_colour(x) for x in line_colour]
    marker_edge_colour = [_parse_colour(x) for x in marker_edge_colour]
    marker_face_colour = [_parse_colour(x) for x in marker_face_colour]

    for k in range(len(pclouds)):
        p = pclouds[k]
        if isinstance(p, TriMesh):
            pclouds[k] = p.as_pointgraph()

        points = p.points
        if (render_lines[k] and line_width[k] > 0 and
            hasattr(p, 'edges') and p.edges.size > 0):
            edges = p.edges
            lines = zip(points[edges[:, 0], :],
                        points[edges[:, 1], :])

            for l1, l2 in lines:
                draw.line([tuple(l1[::-1]), tuple(l2[::-1])],
                          fill=line_colour[k], width=line_width[k])

        if render_markers[k] and marker_size[k] > 0:
            draw_func = (draw.ellipse if marker_style[k] == 'o'
                         else draw.rectangle)
            outline = (marker_edge_colour[k] if marker_edge_width[k] == 1
                       else None)
            for p in points:
                y, x = p
                draw_func((x - marker_size[k], y - marker_size[k],
                           x + marker_size[k], y + marker_size[k]),
                          fill=marker_face_colour[k], outline=outline)

    del draw

    pixels = np.asarray(pil_im)
    if image.n_channels == 3:
        return Image.init_from_rolled_channels(pixels)
    else:
        return Image(pixels)


_RASTERIZE_BACKENDS = {'matplotlib': _rasterize_matplotlib,
                       'pillow': _rasterize_pillow}


def rasterize_landmarks_2d(image, group=None, render_lines=True, line_style='-',
                           line_colour='b', line_width=1, render_markers=True,
                           marker_style='o', marker_size=1,
                           marker_face_colour='b', marker_edge_colour='b',
                           marker_edge_width=1, backend='matplotlib'):
    r"""
    This method provides the ability to rasterize 2D landmarks onto an image.
    The returned image has the specified landmark groups rasterized onto
    the image - which is useful for things like creating result examples
    or rendering videos with annotations.

    Since multiple landmark groups can be specified, all arguments can
    take lists of parameters that map to the provided groups list. Therefore,
    the parameters must be lists of the correct length or a single parameter
    to apply to every landmark group.

    Multiple backends are provided, all with different strengths. The 'pillow'
    backend is very fast, but not very flexible. The `matplotlib` backend
    should be feature compatible with other Menpo rendering methods, but
    is much slower due to the overhead of creating a figure to render
    into.

    Parameters
    ----------
    image : :map:`Image` or subclass
        The image to render onto.
    group : `str` or `list` of `str`, optional
        The landmark group key, or a list of keys.
    render_lines : `bool`, optional
        If ``True``, and the provided landmark group is a :map:`PointGraph`,
        the edges are rendered.
    line_style : `str`, optional
        The style of the edge line. Not all backends support this argument.
    line_colour : `str` or `tuple`, optional
        A Matplotlib style colour or a backend dependant colour.
    line_width : `int`, optional
        The width of the line to rasterize.
    render_markers : `bool`, optional
        If ``True``, render markers at the coordinates of each landmark.
    marker_style : `str`, optional
        A Matplotlib marker style. Not all backends support all marker styles.
    marker_size : `int`, optional
        The size of the marker - different backends use different scale
        spaces so consistent output may by difficult.
    marker_face_colour : `str`, optional
        A Matplotlib style colour or a backend dependant colour.
    marker_edge_colour : `str`, optional
        A Matplotlib style colour or a backend dependant colour.
    marker_edge_width : `int`, optional
        The width of the marker edge. Not all backends support this.
    backend : {'matplotlib', 'pillow'}, optional
        The backend to use.

    Returns
    -------
    rasterized_image : :map:`Image`
        The image with the landmarks rasterized directly into the pixels.

    Raises
    ------
    ValueError
        Only 2D images are supported.
    ValueError
        Only RGB (3-channel) or Greyscale (1-channel) images are supported.
    """
    if image.n_channels != 1 and image.n_channels != 3:
        raise ValueError('Only RGB or Greyscale images can be rasterized')
    if image.n_dims != 2:
        raise ValueError('Only 2D images can be rasterized.')

    if backend in _RASTERIZE_BACKENDS:
        if isinstance(group, list):
            landmarks = [image.landmarks[g].lms for g in group]
        else:
            landmarks = [image.landmarks[group].lms]

        # Validate all the parameters for multiple landmark groups being
        # passed in
        n_pclouds = len(landmarks)
        render_lines = check_param(n_pclouds, bool, 'render_lines',
                                   render_lines)
        line_style = check_param(n_pclouds, basestring, 'line_style',
                                 line_style)
        line_colour = check_param(n_pclouds, (basestring, tuple), 'line_colour',
                                  line_colour)
        line_width = check_param(n_pclouds, int, 'line_width', line_width)
        render_markers = check_param(n_pclouds, bool, 'render_markers',
                                     render_markers)
        marker_style = check_param(n_pclouds, basestring, 'marker_style',
                                   marker_style)
        marker_size = check_param(n_pclouds, int, 'marker_size', marker_size)
        marker_face_colour = check_param(n_pclouds, (basestring, tuple),
                                         'marker_face_colour',
                                         marker_face_colour)
        marker_edge_colour = check_param(n_pclouds, (basestring, tuple),
                                         'marker_edge_colour',
                                         marker_edge_colour)
        marker_edge_width = check_param(n_pclouds, int, 'marker_edge_width',
                                        marker_edge_width)

        return _RASTERIZE_BACKENDS[backend](
            image, landmarks, render_lines=render_lines, line_style=line_style,
            line_colour=line_colour, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width)
    else:
        raise ValueError('Unsupported backend: {}'.format(backend))
