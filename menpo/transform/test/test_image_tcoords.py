import numpy as np
from numpy.testing import assert_equal

from menpo.transform import image_coords_to_tcoords, tcoords_to_image_coords

IMG_SHAPE = (121, 251)
TCOORDS = np.array([[0, 0],
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    [0.5, 0.5]])

IMG_COORDS = np.array([[120, 0],
                       [0, 0],
                       [0, 250],
                       [120, 250],
                       [60, 125]])


def test_tcoords_to_image_coords():
    assert_equal(tcoords_to_image_coords(IMG_SHAPE).apply(TCOORDS), IMG_COORDS)


def test_image_coords_to_tcoords():
    assert_equal(image_coords_to_tcoords(IMG_SHAPE).apply(IMG_COORDS), TCOORDS)
