import numpy as np
from scipy.ndimage import map_coordinates
from pybug.warp.cinterp import interp2


def __cinterp2(pixels, points_to_sample, **kwargs):
    mode = kwargs.get('mode', 'bilinear')

    return interp2(pixels, points_to_sample[0, :], points_to_sample[1, :],
                   mode=mode)


def __scipy(pixels, points_to_sample, **kwargs):
    sampled_pixel_values = []
    mode = kwargs.get('mode', 'constant')
    order = kwargs.get('order', 1)

    # Loop over every channel in image
    for i in xrange(pixels.shape[2]):
        sampled_pixel_values.append(map_coordinates(np.squeeze(pixels[..., i]),
                                                    points_to_sample,
                                                    mode=mode,
                                                    order=order))

    sampled_pixel_values = [v.reshape([-1, 1]) for v in sampled_pixel_values]
    return np.concatenate(sampled_pixel_values, axis=1)


def cinterp2_warp(image, template_image, transform, warp_mask=False,
                  mode='bilinear'):
    """
    Samples an image at all the masked pixel locations in ``template_image``,
    returning a version of template image where all pixels have been sampled.

    This uses a C-based interpolator that was designed to be identical when
    used in both Python and Matlab.

    Parameters
    ----------
    image : :class:`pybug.image.base.Image`
        The image that is to be sampled from.
    template_image : :class:`pybug.image.base.Image`
        The template image. Defines what pixels will be sampled.
    transform : :class:`pybug.transform.base.Transform`
        The transform that dictates where the pixel locations
        of the template image are mapped to on image. Pixel values will be
        sampled from the mapped location.
    warp_mask : bool, optional
        If ``True``, sample the ``image.mask`` at all ``template_image``
        points, setting the returned image mask to the sampled value
        **within the masked region of ``template_image``**.

        Default: ``False``

        .. note::

            This is most commonly set ``True`` in combination with a
            ``template_image`` with an all ``True`` mask, as this then is a
            warp of the image and it's full mask. (If ``template_image`` has a
            mask, only the masked region's mask will be updated).
    mode : {'bilinear', 'bicubic', 'nearest'}, optional
        The type of interpolation to be carried out.

        Default: bilinear

    Returns
    -------
    warped_image : :class:`pybug.image.base.Image`
        The warped image. This is a copy of the original images with the pixel
        information changed to represent the warping.

    """
    return _base_warp(image, template_image, transform,  __cinterp2,
                      warp_mask=warp_mask, mode=mode)


def scipy_warp(image, template_image, transform, warp_mask=False,
               mode='constant', order=1):
    """
    Samples an image at all the masked pixel locations in ``template_image``,
    returning a version of template image where all pixels have been sampled.

    This uses ``scipy.ndimage.interpolation.map_coordinates`` internally.

    Parameters
    ----------
    image : :class:`pybug.image.base.Image`
        The image that is to be sampled from.
    template_image : :class:`pybug.image.base.Image`
        The template image. Defines what pixels will be sampled.
    transform : :class:`pybug.transform.base.Transform`
        The transform that dictates where the pixel locations
        of the template image are mapped to on image. Pixel values will be
        sampled from the mapped location.
    warp_mask : bool, optional
        If ``True``, sample the ``image.mask`` at all ``template_image``
        points, setting the returned image mask to the sampled value
        **within the masked region of ``template_image``**.

        Default: ``False``

        .. note::

            This is most commonly set ``True`` in combination with a
            ``template_image`` with an all ``True`` mask, as this then is a
            warp of the image and it's full mask. (If ``template_image`` has a
            mask, only the masked region's mask will be updated).
    mode : {'constant', 'nearest', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according to the
        given mode.

        Default: 'constant' (0)
    order : int, optional
        The order of the spline interpolation. The order has to be in the
        range 0-5.

        Default: 1

    Returns
    -------
    warped_image : :class:`pybug.image.base.Image`
        The warped image. This is a copy of the original images with the pixel
        information changed to represent the warping.

    """
    return _base_warp(image, template_image, transform,  __scipy,
                      warp_mask=warp_mask, mode=mode, order=order)


def _base_warp(image, template_image, transform, interpolator, warp_mask=False,
               **kwargs):
    """
    Performs the warping using the given interpolator. This may include
    warping the mask.

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

    Parameters
    ----------
    image : :class:`pybug.image.base.Image`
        The image that is to be sampled from.
    template_image : :class:`pybug.image.base.Image`
        The template image. Defines what pixels will be sampled.
    transform : :class:`pybug.transform.base.Transform`
        The transform that dictates where the pixel locations
        of the template image are mapped to on image. Pixel values will be
        sampled from the mapped location.
    interpolator : func
        A function that takes the image pixels and points to sample and returns
        the interpolated sub-pixel values.
    warp_mask : bool, optional
        If ``True``, sample the ``image.mask`` at all ``template_image``
        points, setting the returned image mask to the sampled value
        **within the masked region of ``template_image``**.

        Default: ``False``

        .. note::

            This is most commonly set ``True`` in combination with a
            ``template_image`` with an all ``True`` mask, as this then is a
            warp of the image and it's full mask. (If ``template_image`` has a
            mask, only the masked region's mask will be updated).
    kwargs : dict
        Passed through to the interpolator.

    Returns
    -------
    warped_image : :class:`pybug.image.base.Image`
        The warped image. This is a copy of the original images with the pixel
        information changed to represent the warping.
    """
    generic_mask = template_image.mask
    template_points = generic_mask.true_indices
    points_to_sample = transform.apply(template_points).T
    # we want to sample each channel in turn, returning a vector of sampled
    # pixels. Store those in a (n_pixels, n_channels) array.
    sampled_pixel_values = interpolator(image.pixels, points_to_sample,
                                        **kwargs)

    # Set all NaN pixels to 0
    sampled_pixel_values = np.nan_to_num(sampled_pixel_values)
    # note that as Image.as_vector() returns a vector with stride
    # [R1 G1 B1, R2 ....] we can flatten our sampled_pixel_values to get the
    # normal Image vector form.
    warped_image = type(image).blank(generic_mask.shape, mask=generic_mask,
                                     n_channels=image.n_channels)
    warped_image = warped_image.update_from_vector(
        sampled_pixel_values.flatten())

    if warp_mask:
        # note that we need to set the order to 0 for mapping binary data
        kwargs.setdefault('order', 0)
        new_mask_values = interpolator(image.mask.pixels, points_to_sample,
                                       **kwargs)
        # rebuild the mask just like we do with images.
        new_mask = template_image.mask.from_vector(new_mask_values)
        # update the template to use the new mask
        warped_image.mask = new_mask
    return warped_image

