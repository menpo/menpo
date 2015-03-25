import numpy as np
from numpy.testing import assert_allclose
from menpo.shape import PointCloud
from menpo.image import MaskedImage, BooleanImage


def test_constrain_mask_to_landmarks():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0.0, 0.0], [5.0, 0.0],
                                                [5.0, 5.0], [0.0, 5.0]]))
    img.constrain_mask_to_landmarks(group='box')

    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(img.mask.n_true() == 36)
    assert_allclose(img.mask.pixels, example_mask.pixels)
