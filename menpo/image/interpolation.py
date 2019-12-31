import numpy as np

map_coordinates = None  # expensive, from scipy.ndimage
from menpo.transform import Homogeneous

# Store out a transform that simply switches the x and y axis
xy_yx = Homogeneous(np.array([[0., 1., 0.],
                              [1., 0., 0.],
                              [0., 0., 1.]]))


def scipy_interpolation(pixels, points_to_sample, mode='constant', order=1,
                        cval=0.):
    r"""
    Interpolation utilizing scipy map_coordinates function.

    Parameters
    ----------
    pixels : ``(n_channels, M, N, ...)`` `ndarray`
        The image to be sampled from, the first axis containing channel
        information
    points_to_sample : ``(n_points, n_dims)`` `ndarray`
        The points which should be sampled from pixels
    mode : ``{constant, nearest, reflect, wrap}``, optional
        Points outside the boundaries of the input are filled according to the
        given mode
    order : `int,` optional
        The order of the spline interpolation. The order has to be in the
        range [0, 5].
    cval : `float`, optional
        The value that should be used for points that are sampled from
        outside the image bounds if mode is ``constant``.

    Returns
    -------
    sampled_image : `ndarray`
        The pixel information sampled at each of the points.
    """
    global map_coordinates
    if map_coordinates is None:
        from scipy.ndimage import map_coordinates  # expensive
    sampled_pixel_values = np.empty((pixels.shape[0], points_to_sample.shape[0]),
                                    dtype=pixels.dtype)

    # Loop over every channel in image - we know first axis is always channels
    # Note that map_coordinates uses the opposite (dims, points) convention
    # to us so we transpose
    points_to_sample_t = points_to_sample.T
    for i in range(pixels.shape[0]):
        map_coordinates(pixels[i],
                        points_to_sample_t,
                        mode=mode,
                        order=order,
                        cval=cval,
                        output=sampled_pixel_values[i])
    return sampled_pixel_values


try:
    import cv2


    def _mode_to_opencv(mode):
        if mode == 'nearest':
            return cv2.BORDER_REPLICATE
        elif mode == 'constant':
            return cv2.BORDER_CONSTANT
        else:
            raise ValueError('Unknown mode "{}", must be one of '
                             '(nearest, constant)'.format(mode))


    def _order_to_opencv(order):
        if order == 0:
            return cv2.INTER_NEAREST
        elif order == 1:
            return cv2.INTER_LINEAR
        else:
            raise ValueError('Unsupported order "{}", '
                             'must be one of (0, 1)'.format(order))


    def cv2_perspective_interpolation(pixels, template_shape, h_transform,
                                      mode='constant', order=1, cval=0.):
        r"""
        Interpolation utilizing OpenCV fast perspective warping. This method
        assumes that the warp takes the form of a homogeneous transform, and
        thus is much faster for operations such as scaling.

        Note less modes and orders are supported than the more generic
        interpolation methods.

        Parameters
        ----------
        pixels : ``(n_channels, M, N, ...)`` `ndarray`
            The image to be sampled from, the first axis containing channel
            information.
        template_shape : `tuple`
            The shape of the new image that will be sampled
        mode : ``{constant, nearest}``, optional
            Points outside the boundaries of the input are filled according to the
            given mode.
        order : int, optional
            The order of the spline interpolation. The order has to be in the
            range [0, 1].
        cval : `float`, optional
            The value that should be used for points that are sampled from
            outside the image bounds if mode is 'constant'

        Returns
        -------
        sampled_image : `ndarray`
            The pixel information sampled at each of the points.
        """
        matrix = xy_yx.compose_before(h_transform).compose_before(xy_yx).h_matrix
        # Ensure template shape is a tuple (as required by OpenCV)
        template_shape = tuple(template_shape)
        cv_template_shape = template_shape[::-1]  # Flip to (W, H)

        # Unfortunately, OpenCV does not seem to support the boolean numpy type
        if pixels.dtype == np.bool:
            in_pixels = pixels.astype(np.uint8)
        else:
            in_pixels = pixels

        warped_image = np.empty((pixels.shape[0],) + template_shape,
                                dtype=in_pixels.dtype)
        for i in range(pixels.shape[0]):
            cv2.warpPerspective(in_pixels[i], matrix, cv_template_shape,
                                dst=warped_image[i],
                                flags=_order_to_opencv(order) + cv2.WARP_INVERSE_MAP,
                                borderMode=_mode_to_opencv(mode),
                                borderValue=cval)

        # As above, we may need to convert the uint8 back to bool
        if pixels.dtype == np.bool:
            warped_image = warped_image.astype(np.bool)
        return warped_image
except ImportError:
    pass
