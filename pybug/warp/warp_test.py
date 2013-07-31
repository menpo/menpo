import numpy as np
from numpy.testing import assert_allclose
from pybug.transform import AffineTransform
from pybug.warp import scipy_warp
from pybug.io import auto_import
from pybug import data_path_to


# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
image = auto_import(takeo_path)[0].as_greyscale()
template, translation = image.crop(slice(70, 169), slice(30, 129))
initial_params = np.array([0, 0, 0, 0, 70, 30])


def test_scipy_warp():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = scipy_warp(image, template, target_transform)

    assert(warped_im.image_shape == template.image_shape)
    assert_allclose(warped_im.pixels, template.pixels)
