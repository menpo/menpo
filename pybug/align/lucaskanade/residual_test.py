from unittest import SkipTest
import numpy as np
from numpy.testing import assert_approx_equal
from pybug.warp import warp
from pybug.io import auto_import
from pybug import data_path_to
from pybug.transform import AffineTransform
from pybug.align.lucaskanade import ImageInverseCompositional, \
    ImageForwardAdditive
from pybug.align.lucaskanade.residual import *

try:
    from pybug.warp.base import matlab_interpolator
except Exception:
    pass

# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
takeo = auto_import(takeo_path)[0]
image = takeo.pixels[..., 0]

# Setup global conditions
initial_params = np.array([0, 0, 0, 0, 30.5, 70.5])
target_params = np.array([0, 0.1, 0.2, 0, 30, 70])
target_shape = (99, 99)


def compute_rms_point_error(test_pts, template_affine, M):
    iteration_pts = M.apply(template_affine.T)
    diff_pts = test_pts - iteration_pts
    return np.sqrt(np.mean(diff_pts ** 2))


def setup_conditions(interpolator):
    global image
    global initial_params
    global target_params
    global target_shape

    template = warp(image, target_shape,
                    AffineTransform.from_vector(target_params),
                    interpolator=interpolator)

    return image, template, initial_params


def setup_error():
    global target_params
    global target_shape

    target_transform = AffineTransform.from_vector(target_params)
    original_box = np.array([[0, 0],
                             [target_shape[1], 0],
                             [target_shape[1], target_shape[0]],
                             [0, target_shape[0]]]).T
    target_pts = target_transform.apply(original_box.T)

    return target_pts, original_box


def compute_fixed_error(transform):
    target_pts, original_box = setup_error()
    return compute_rms_point_error(target_pts, original_box, transform)


def residual_wrapper(residual, algo, interpolator, expected_error):
    image, template, initial_params = setup_conditions(interpolator)
    align_algo = algo(image, template, residual,
                      AffineTransform.from_vector(initial_params),
                      interpolator=interpolator)
    transform = align_algo.align()
    rms_error = compute_fixed_error(transform)
    assert_approx_equal(rms_error, expected_error)

###############################################################################


def test_2d_ls_ic_map_coords():
    residual_wrapper(LSIntensity(), ImageInverseCompositional,
                     map_coordinates_interpolator,
                     0.5470207993885552)


def test_2d_ls_fa_map_coords():
    residual_wrapper(LSIntensity(), ImageForwardAdditive,
                     map_coordinates_interpolator,
                     2.2014299004800235)


def test_2d_ecc_ic_map_coords():
    residual_wrapper(ECC(), ImageInverseCompositional,
                     map_coordinates_interpolator,
                     0.0002234343558694913)


def test_2d_ecc_fa_map_coords():
    residual_wrapper(ECC(), ImageForwardAdditive, map_coordinates_interpolator,
                     2.2293502746889122)


def test_2d_gabor_ic_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageInverseCompositional,
                     map_coordinates_interpolator,
                     8.514546094166729)


def test_2d_gabor_fa_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageForwardAdditive,
                     map_coordinates_interpolator,
                     8.917760741895027)


def test_2d_gradientimages_ic_map_coords():
    residual_wrapper(GradientImages(), ImageInverseCompositional,
                     map_coordinates_interpolator,
                     10.990483894459013)


def test_2d_gradientimages_fa_map_coords():
    residual_wrapper(GradientImages(), ImageForwardAdditive,
                     map_coordinates_interpolator,
                     10.753320121441979)


def test_2d_gradientcorrelation_ic_map_coords():
    residual_wrapper(GradientCorrelation(), ImageInverseCompositional,
                     map_coordinates_interpolator,
                     8.75514246052598)


def test_2d_gradientcorrelation_fa_map_coords():
    residual_wrapper(GradientCorrelation(), ImageForwardAdditive,
                     map_coordinates_interpolator,
                     2.9755961254911782)

###############################################################################


def test_2d_ls_ic_matlab():
    try:
        residual_wrapper(LSIntensity(), ImageInverseCompositional,
                         matlab_interpolator,
                         0.040637276229920555)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_ls_fa_matlab():
    try:
        residual_wrapper(LSIntensity(), ImageForwardAdditive,
                         matlab_interpolator,
                         0.9461180208577028)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_ecc_ic_matlab():
    try:
        residual_wrapper(ECC(), ImageInverseCompositional, matlab_interpolator,
                         5.7493047578790984e-05)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_ecc_fa_matlab():
    try:
        residual_wrapper(ECC(), ImageForwardAdditive, matlab_interpolator,
                         1.934284157262843)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_gabor_ic_matlab():
    try:
        global target_shape
        residual_wrapper(GaborFourier(target_shape), ImageInverseCompositional,
                         matlab_interpolator,
                         8.337234230634943)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_gabor_fa_matlab():
    try:
        global target_shape
        residual_wrapper(GaborFourier(target_shape), ImageForwardAdditive,
                         matlab_interpolator,
                         8.717929122721493)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_gradientimages_ic_matlab():
    try:
        residual_wrapper(GradientImages(), ImageInverseCompositional,
                         matlab_interpolator,
                         11.110591911780372)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_gradientimages_fa_matlab():
    try:
        residual_wrapper(GradientImages(), ImageForwardAdditive,
                         matlab_interpolator,
                         10.669544303282278)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_gradientcorrelation_ic_matlab():
    try:
        residual_wrapper(GradientCorrelation(),
                         ImageInverseCompositional, matlab_interpolator,
                         7.217953890493592)
    except NameError:
        raise SkipTest("Matlab not found")


def test_2d_gradientcorrelation_fa():
    try:
        residual_wrapper(GradientCorrelation(),
                         ImageForwardAdditive, matlab_interpolator,
                         2.0480934329718363)
    except NameError:
        raise SkipTest("Matlab not found")