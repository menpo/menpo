import numpy as np
from scipy.ndimage import map_coordinates
from pybug.image import Image


def scipy_warp(image, template_image, transform, order=1,
               warp_mask=False, mode='constant'):
    """
    Samples an image at all the masked pixel locations in template_image,
    returning a version of template image where all pixels have been sampled.

    :param image: The image that is to be sampled from
    :param template_image: The template image. Defines what pixels will be
        sampled.
    :param transform: The transform that dictates where the pixel locations
        of the template image are mapped to on image. Pixel values will be
        sampled from the mapped location.
    :param warp_mask: Boolean. If True, sample the image.mask at all
    template_image points, setting the returned image mask to the sampled
    value *within the masked region of template_image*. Not that this is
    most commonly set True in combination with a template_image with an all
    true mask, as this then is a warp of the image and it's full mask. (If
    template_image has a mask, only the masked region's mask will be updated).
    :return:
    """
    template_points = template_image.mask.true_indices
    points_to_sample = transform.apply(template_points).T
    # we want to sample each channel in turn, returning a vector of sampled
    # pixels. Store those in a (n_pixels, n_channels) array.
    sampled_pixel_values = np.zeros([template_image.mask.n_true,
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
    warped_image = template_image.from_vector(sampled_pixel_values.flatten(),
                                      n_channels=image.n_channels)
    if warp_mask:
        new_mask_values = map_coordinates(image.mask.pixels, points_to_sample,
                                          order=order, mode=mode)
        # rebuild the mask just like we do with images.
        new_mask = template_image.mask.from_vector(new_mask_values)
        # update the template to use the new mask
        warped_image.mask = new_mask
    return warped_image

