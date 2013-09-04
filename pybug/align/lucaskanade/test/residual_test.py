import numpy as np
from numpy.testing import assert_approx_equal
from pybug.warp import scipy_warp
from pybug.io import auto_import
from pybug import data_path_to
from pybug.transform import AffineTransform
from pybug.align.lucaskanade.image import ImageInverseCompositional, \
    ImageForwardAdditive
from pybug.align.lucaskanade.residual import *


# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
takeo = auto_import(takeo_path)[0]
image = takeo.as_greyscale()
template, translation = image.crop(slice(70, 169), slice(30, 129))

# Setup global conditions
initial_params = np.array([0, 0, 0, 0, 70.5, 30.5])
target_params = np.array([0, 0.2, 0.1, 0, 70, 30])
target_shape = (99, 99)


def compute_rms_point_error(test_pts, template_affine, M):
    iteration_pts = M.apply(template_affine.T)
    diff_pts = test_pts - iteration_pts
    return np.sqrt(np.mean(diff_pts ** 2))


def setup_conditions(warp):
    target_transform = AffineTransform.from_vector(target_params)
    tmplt = warp(image, template, target_transform)

    return image, tmplt, initial_params


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


def residual_wrapper(residual, algo, warp, expected_error):
    image, template, initial_params = setup_conditions(warp)
    align_algo = algo(template, residual,
                      AffineTransform.from_vector(initial_params), warp=warp)
    transform = align_algo.align(image, initial_params)
    rms_error = compute_fixed_error(transform)
    assert_approx_equal(rms_error, expected_error)

###############################################################################


def test_2d_ls_ic_map_coords():
    residual_wrapper(LSIntensity(), ImageInverseCompositional,
                     scipy_warp,
                     0.5470207993885552)


def test_2d_ls_fa_map_coords():
    residual_wrapper(LSIntensity(), ImageForwardAdditive,
                     scipy_warp,
                     2.2014299004800235)


def test_2d_ecc_ic_map_coords():
    residual_wrapper(ECC(), ImageInverseCompositional,
                     scipy_warp,
                     0.0002234343558694913)


def test_2d_ecc_fa_map_coords():
    residual_wrapper(ECC(), ImageForwardAdditive,
                     scipy_warp,
                     2.2293502746889122)


def test_2d_gabor_ic_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageInverseCompositional,
                     scipy_warp,
                     8.514546094166729)


def test_2d_gabor_fa_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageForwardAdditive,
                     scipy_warp,
                     8.917760741895027)


def test_2d_gradientimages_ic_map_coords():
    residual_wrapper(GradientImages(), ImageInverseCompositional,
                     scipy_warp,
                     10.990483894459013)


def test_2d_gradientimages_fa_map_coords():
    residual_wrapper(GradientImages(), ImageForwardAdditive,
                     scipy_warp,
                     10.753320121441979)


def test_2d_gradientcorrelation_ic_map_coords():
    residual_wrapper(GradientCorrelation(), ImageInverseCompositional,
                     scipy_warp,
                     8.755365214854102)


def test_2d_gradientcorrelation_fa_map_coords():
    residual_wrapper(GradientCorrelation(), ImageForwardAdditive,
                     scipy_warp,
                     10.494169974979323)
