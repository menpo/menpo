import numpy as np
from numpy.testing import assert_allclose
from menpo.image import Image
from menpo.shape import bounding_box


def test_rescale_landmarks_to_diagonal_range():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks['test'] = bounding_box([0, 0], [10, 10])

    img_rescaled = img.rescale_landmarks_to_diagonal_range(20 * np.sqrt(2))
    assert_allclose(img_rescaled.shape, (200, 200))


def test_rescale_to_diagonal():
    img = Image.init_blank((100, 100), n_channels=1)

    img_rescaled = img.rescale_to_diagonal(200 * np.sqrt(2))
    assert_allclose(img_rescaled.shape, (200, 200))


def test_rescale_to_pointcloud():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks['test'] = bounding_box([0, 0], [10, 10])
    pcloud = bounding_box([0, 0], [25, 25])

    img_rescaled = img.rescale_to_pointcloud(pcloud)
    assert_allclose(img_rescaled.shape, (250, 250))


def test_crop_to_pointcloud():
    img = Image.init_blank((100, 100), n_channels=1)
    pcloud = bounding_box([0, 0], [50, 50])

    img_cropped = img.crop_to_pointcloud(pcloud)
    assert_allclose(img_cropped.shape, (50, 50))


def test_crop_to_landmarks():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks['test'] = bounding_box([0, 0], [10, 10])

    img_cropped = img.crop_to_landmarks()
    assert_allclose(img_cropped.shape, (10, 10))
    assert 1


def test_crop_to_pointcloud_proportion():
    img = Image.init_blank((100, 100), n_channels=1)
    pcloud = bounding_box([0, 0], [50, 50])

    img_cropped = img.crop_to_pointcloud_proportion(pcloud, 0.1)
    assert_allclose(img_cropped.shape, (55, 55))


def test_crop_to_landmarks_proportion():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks['test'] = bounding_box([0, 0], [10, 10])

    img_cropped = img.crop_to_landmarks_proportion(0.1)
    assert_allclose(img_cropped.shape, (11, 11))
