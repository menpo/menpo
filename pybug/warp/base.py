import numpy as np
from scipy.ndimage import map_coordinates


def scipy_warp(image, template_image, transform, order=1, mode='constant'):
    """
    Samples an image at all the masked pixel locations in template_image,
    returning a version of template image where all pixels have been sampled.

    :param image: The image that is to be sampled from
    :param template_image: The template image. Defines what pixels will be
        sampled.
    :param transform: The transform that dictates where the pixel locations
        of the template image are mapped to on image. Pixel values will be
        sampled from the mapped location.
    :return:
    """
    template_points = template_image.masked_pixel_indices
    points_to_sample = transform.apply(template_points).T
    # we want to sample each channel in turn, returning a vector of sampled
    # pixels. Store those in a (n_pixels, n_channels) array.
    sampled_pixel_values = np.zeros([template_image.n_masked_pixels,
                                     image.n_channels])
    pixels = image.pixels
    for i in xrange(image.n_channels):
        sampled_pixel_values[..., i] = map_coordinates(
            np.squeeze(pixels[..., i]), points_to_sample,
            order=order, mode=mode)

    # Set all NaN pixels to 0
    sampled_pixel_values = np.nan_to_num(sampled_pixel_values)

    # note that as Image.as_vector() returns a vector with stride
    # [R1 G1 B1, R2 ....] we can flatten our sampled_pixel_values to get the
    # normal Image vector form.
    return template_image.from_vector(sampled_pixel_values.flatten(),
                                      n_channels=image.n_channels)