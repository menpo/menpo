import numpy as np
import scipy.ndimage as ndimage


def map_coordinates_interpolator(image, coords, shape, **kwargs):
    order = kwargs.get('order', 1)
    mode = kwargs.get('mode', 'constant')
    return ndimage.map_coordinates(np.squeeze(image), coords,
                                   order=order, mode=mode)


def matlab_interpolator(image, coords, shape, **kwargs):
    from pybug.matlab.wrapper import mlab
    params_list = []
    method = kwargs.get('method', 'linear')
    for i in xrange(coords.shape[0]):
        params_list.append(coords[i, ...].reshape(shape))
    # Append interpolation method
    params_list.append(method)
    return mlab.interpn(np.squeeze(image), *params_list).flatten()


def warp(image, template_shape, transform,
         interpolator=map_coordinates_interpolator,
         type='constant', order=1):
    # Swap x and y for images
    dims = list(template_shape)
    dims[:2] = dims[1::-1]
    ranges = [np.arange(dim) for dim in dims]

    # Generate regular grid and flatten for every dimension
    grids = np.meshgrid(*ranges)
    grids = [g.reshape(-1, 1) for g in grids]
    grids = np.hstack(grids)


    # Warp grids
    uv = transform.apply(grids).T
    uv = [np.reshape(x, template_shape) for x in uv]

    # Interpolate according to transformed grid
    warped_image = interpolator(image, uv, order, type)
    warped_image = np.nan_to_num(warped_image)

    return warped_image


def warp_image_onto_template_image(image, template_image, transform,
                                   interpolator=map_coordinates_interpolator,
                                   **kwargs):
    """
    Samples an image at all the masked pixel locations in template_image,
    returning a version of template image where all pixels have been sampled.

    :param image: The image that is to be sampled from
    :param template_image: The template image. Defines what pixels will be
        sampled.
    :param transform: The transform that dictates where the pixel locations
        of the template image are mapped to on image. Pixel values will be
        sampled from the mapped location.
    :param interpolator: The interpolator that will be used in resolving
        what value is chosen from the image at the mapped coordinates.
    :param mode:
    :param order:
    :return:
    """
    template_points = template_image.masked_pixel_indices
    points_to_sample = transform.apply(template_points).T

    sampled_pixel_values = np.zeros([template_image.pixels.size,
                                     image.n_channels])
    pixels = image.pixels
    for i in xrange(image.n_channels):
        sampled_pixel_values[..., i] = interpolator(pixels[..., i],
                                                    points_to_sample,
                                                    template_image.image_shape,
                                                    **kwargs)
    sampled_pixel_values = np.nan_to_num(sampled_pixel_values)

    return template_image.from_vector(sampled_pixel_values,
                                      n_channels=image.n_channels)