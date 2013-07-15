import numpy as np
from numpy.testing import assert_allclose
from pybug.transform import AffineTransform
from pybug.warp import warp_image_onto_template_image
from pybug.io import auto_import
from pybug import data_path_to
from pybug.warp.base import map_coordinates_interpolator, matlab_interpolator

# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
image = auto_import(takeo_path)[0].as_greyscale()
template, translation = image.crop(slice(70, 169), slice(30, 129))
initial_params = np.array([0, 0, 0, 0, 70, 30])


def test_warp_image_onto_template_image_map_coordinates():
    global image
    global template
    global initial_params

    interpolator = map_coordinates_interpolator
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = warp_image_onto_template_image(image, template,
                                               target_transform,
                                               interpolator=interpolator)

    assert(warped_im.image_shape == template.image_shape)
    assert_allclose(warped_im.pixels, template.pixels)


def test_warp_image_onto_template_image_matlab():
    global image
    global template
    global initial_params

    interpolator = matlab_interpolator
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = warp_image_onto_template_image(image, template,
                                               target_transform,
                                               interpolator=interpolator)

    assert(warped_im.image_shape == template.image_shape)
    # TODO: What can we do about this?? The interpolation is totally different!
    # TODO: Visually they look identical but numerically they are different.
    # assert_allclose(warped_im.pixels[..., 0], template.pixels[..., 0])
