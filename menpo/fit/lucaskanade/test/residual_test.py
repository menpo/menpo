from numpy.testing import assert_approx_equal

from menpo.image import BooleanImage
import menpo.io as pio
from menpo.transform import Affine

from menpo.fit.lucaskanade.image import (ImageInverseCompositional,
                                         ImageForwardAdditive)
from menpo.fit.lucaskanade.residual import *


target_shape = (90, 90)

# Setup the static assets (the takeo image)
takeo = pio.import_builtin_asset('takeo.ppm')
image = takeo.as_greyscale()
template_mask = BooleanImage.blank(target_shape)

# Setup global conditions
initial_params = np.array([0, 0, 0, 0, 70.5, 30.5])
target_params = np.array([0, 0.2, 0.1, 0, 70, 30])


def compute_rms_point_error(test_pts, template_affine, M):
    iteration_pts = M.apply(template_affine.T)
    diff_pts = test_pts - iteration_pts
    return np.sqrt(np.mean(diff_pts ** 2))


def setup_conditions(interpolator):
    target_transform = Affine.identity(2).from_vector(target_params)
    image_warped = image.warp_to(template_mask, target_transform,
                                 interpolator=interpolator)
    return image, image_warped, initial_params


def setup_error():
    target_transform = Affine.identity(2).from_vector(target_params)
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
        template, residual, Affine.identity(2).from_vector(
            initial_params))
    fitting = align_algorithm.fit(image, initial_params)
    transform = fitting.final_transform
    rms_error = compute_fixed_error(transform)
    assert_approx_equal(rms_error, expected_error)

###############################################################################


def test_2d_ls_ic_map_coords():
    residual_wrapper(LSIntensity(), ImageInverseCompositional,
                     'scipy',
                     1.551662325323618e-05)


def test_2d_ls_fa_map_coords():
    residual_wrapper(LSIntensity(), ImageForwardAdditive,
                     'scipy',
                     1.4743724618438824e-05)


def test_2d_ecc_ic_map_coords():
    residual_wrapper(ECC(), ImageInverseCompositional,
                     'scipy',
                     2.0802365833550096e-06)


def test_2d_ecc_fa_map_coords():
    residual_wrapper(ECC(), ImageForwardAdditive,
                     'scipy',
                     1.9477094730117106)


def test_2d_gabor_ic_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageInverseCompositional,
                     'scipy',
                     8.754889609451457)


def test_2d_gabor_fa_map_coords():
    global target_shape
    residual_wrapper(GaborFourier(target_shape), ImageForwardAdditive,
                     'scipy',
                     0.03729593561127312)


def test_2d_gradientimages_ic_map_coords():
    residual_wrapper(GradientImages(), ImageInverseCompositional,
                     'scipy',
                     10.002935866056646)


def test_2d_gradientimages_fa_map_coords():
    residual_wrapper(GradientImages(), ImageForwardAdditive,
                     'scipy',
                     9.952157644001336)


def test_2d_gradientcorrelation_ic_map_coords():
    residual_wrapper(GradientCorrelation(), ImageInverseCompositional,
                     'scipy',
                     6.93587178891484)


def test_2d_gradientcorrelation_fa_map_coords():
    residual_wrapper(GradientCorrelation(), ImageForwardAdditive,
                     'scipy',
                     5.388114566437586)
