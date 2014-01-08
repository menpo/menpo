import numpy as np
from numpy.testing import assert_approx_equal
from pybug.io import auto_import
from pybug import data_path_to
from pybug.transform import AffineTransform
from pybug.lucaskanade.image import ImageInverseCompositional, \
    ImageForwardAdditive
from pybug.lucaskanade.residual import *


# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
takeo = auto_import(takeo_path)[0]
image = takeo.as_greyscale()
template_image = image.cropped_copy([70, 30], [169, 129])
template_mask = template_image.mask

# Setup global conditions
initial_params = np.array([0, 0, 0, 0, 70.5, 30.5])
target_params = np.array([0, 0.2, 0.1, 0, 70, 30])
target_shape = (99, 99)


def compute_rms_point_error(test_pts, template_affine, M):
    iteration_pts = M.apply(template_affine.T)
    diff_pts = test_pts - iteration_pts
    return np.sqrt(np.mean(diff_pts ** 2))


def setup_conditions(interpolator):
    target_transform = AffineTransform.from_vector(target_params)
    image_warped = image.warp_to(template_mask, target_transform,
                                 interpolator=interpolator)
    return image, image_warped, initial_params


def setup_error():
    target_transform = AffineTransform.from_vector(target_params)
    original_box = np.array([[0,               0],
                             [target_shape[0], 0],
                             [target_shape[0], target_shape[1]],
                             [0,               target_shape[1]]]).T
    target_pts = target_transform.apply(original_box.T)

    return target_pts, original_box


def compute_fixed_error(transform):
    target_pts, original_box = setup_error()
    return compute_rms_point_error(target_pts, original_box, transform)


def residual_wrapper(residual, algorithm, interpolator, expected_error):
    image, template, initial_params = setup_conditions(interpolator)
    align_algorithm = algorithm(
        template, residual, AffineTransform.from_vector(initial_params))
    transform = align_algorithm.align(image, initial_params)
    rms_error = compute_fixed_error(transform)
    assert_approx_equal(rms_error, expected_error)

###############################################################################


def test_2d_ls_ic_map_coords():
    residual_wrapper(LSIntensity(), ImageInverseCompositional,
                     'scipy',
                     0.33450925088722977)


def test_2d_ls_fa_map_coords():
    residual_wrapper(LSIntensity(), ImageForwardAdditive,
                     'scipy',
                     2.2014299004800235)


def test_2d_ecc_ic_map_coords():
    residual_wrapper(ECC(), ImageInverseCompositional,
                     'scipy',
                     0.00012855026074710274)


def test_2d_ecc_fa_map_coords():
    residual_wrapper(ECC(), ImageForwardAdditive,
                     'scipy',
                     2.2293502746889122)


def test_2d_gabor_ic_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageInverseCompositional,
                     'scipy',
                     8.089827782035554)


def test_2d_gabor_fa_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageForwardAdditive,
                     'scipy',
                     8.917760741895027)


def test_2d_gradientimages_ic_map_coords():
    residual_wrapper(GradientImages(), ImageInverseCompositional,
                     'scipy',
                     11.092589991835462)


def test_2d_gradientimages_fa_map_coords():
    residual_wrapper(GradientImages(), ImageForwardAdditive,
                     'scipy',
                     10.685518152615705)


def test_2d_gradientcorrelation_ic_map_coords():
    residual_wrapper(GradientCorrelation(), ImageInverseCompositional,
                     'scipy',
                     8.63126181477196)


def test_2d_gradientcorrelation_fa_map_coords():
    residual_wrapper(GradientCorrelation(), ImageForwardAdditive,
                     'scipy',
                     2.6699528027972064)
