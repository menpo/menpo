import numpy as np
from pytest import raises
from numpy.testing import assert_allclose

from menpo.shape import PointCloud
from menpo.image import MaskedImage, BooleanImage


def test_init_from_pointcloud_constrain_mask():
    pc = PointCloud(np.array([[5, 5], [5, 20], [20, 20]]))
    im = MaskedImage.init_from_pointcloud(pc, constrain_mask=True)
    assert im.shape == (15, 15)
    assert im.mask.n_true() == 120


def test_init_from_pointcloud_no_constrain_mask():
    pc = PointCloud(np.array([[5, 5], [5, 20], [20, 20]]))
    im = MaskedImage.init_from_pointcloud(pc, constrain_mask=False)
    assert im.shape == (15, 15)
    assert im.mask.n_true() == 225


def test_constrain_mask_to_landmarks_pwa():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0.0, 0.0], [5.0, 0.0],
                                                [5.0, 5.0], [0.0, 5.0]]))
    new_img = img.constrain_mask_to_landmarks(group='box')

    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(new_img.mask.n_true() == 36)
    assert_allclose(new_img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_pwa_batched():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0.0, 0.0], [5.0, 0.0],
                                                [5.0, 5.0], [0.0, 5.0]]))
    new_img = img.constrain_mask_to_landmarks(group='box', batch_size=2)

    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(new_img.mask.n_true() == 36)
    assert_allclose(new_img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_convex_hull():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0.], [5., 0.],
                                                [5., 5.], [0., 5.]]))
    new_img = img.constrain_mask_to_landmarks(group='box',
                                              point_in_pointcloud='convex_hull')
    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, 1:6] = True
    assert(new_img.mask.n_true() == 30)
    assert_allclose(new_img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_callable():
    def bounding_box(_, indices):
        return np.ones(indices.shape[0], dtype=np.bool)

    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0.], [5., 0.],
                                                [5., 5.], [0., 5.]]))
    new_img = img.constrain_mask_to_landmarks(group='box',
                                              point_in_pointcloud=bounding_box)
    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(new_img.mask.n_true() == 36)
    assert_allclose(new_img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_non_2d():
    img = MaskedImage.init_blank((10, 10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0., 0.]]))
    with raises(ValueError):
        img.constrain_mask_to_landmarks()


def test_constrain_mask_to_landmarks_unknown_key():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0., 0.]]))
    with raises(ValueError):
        img.constrain_mask_to_landmarks(point_in_pointcloud='unknown')


def test_erode():
    img = MaskedImage.init_blank((10, 10))
    img2 = img.erode()
    assert(img2.mask.n_true() == 64)
    img3 = img.erode(n_pixels=3)
    assert(img3.mask.n_true() == 16)


def test_constrain_mask_to_patches_around_landmarks_even():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0.], [5., 0.],
                                                [5., 5.], [0., 5.]]))
    new_img = img.constrain_mask_to_patches_around_landmarks((2,2), group='box')
    assert(new_img.mask.n_true() == 9)
    assert_allclose(new_img.mask.pixels[:, 0, 0], True)
    assert_allclose(new_img.mask.pixels[:, 4:6, 0], True)
    assert_allclose(new_img.mask.pixels[:, 0, 4:6], True)
    assert_allclose(new_img.mask.pixels[:, 4:6, 4:6], True)


def test_constrain_mask_to_patches_around_landmarks_odd():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0.], [5., 0.],
                                                [5., 5.], [0., 5.]]))
    new_img = img.constrain_mask_to_patches_around_landmarks((3,3), group='box')
    assert(new_img.mask.n_true() == 25)
    assert_allclose(new_img.mask.pixels[:, :2, :2], True)
    assert_allclose(new_img.mask.pixels[:, 4:7, :2], True)
    assert_allclose(new_img.mask.pixels[:, :2, 4:7], True)
    assert_allclose(new_img.mask.pixels[:, 4:7, 4:7], True)


def test_set_boundary_pixels():
    mask = np.ones((10, 10), dtype=np.bool)
    img = MaskedImage.init_blank((10, 10), mask=mask, fill=0., n_channels=1)
    new_img = img.set_boundary_pixels(value=2.)
    assert(new_img.mask.n_true() == 100)
    assert(~np.allclose(img.pixels, new_img.pixels))
    assert_allclose(new_img.pixels[0, 1:-1, 1:-1], 0.)
    assert_allclose(new_img.pixels[0, :, 0],       2.)
    assert_allclose(new_img.pixels[0, 0, :],       2.)
    assert_allclose(new_img.pixels[0, :, -1],      2.)
    assert_allclose(new_img.pixels[0, -1, :],      2.)


def test_dilate():
    img = MaskedImage.init_blank((10, 10))
    img = img.erode(n_pixels=3)
    img2 = img.dilate()
    assert(img2.mask.n_true() == 32)
    img3 = img.dilate(n_pixels=3)
    assert(img3.mask.n_true() == 76)


def test_init_from_rolled_channels():
    p = np.empty([50, 60, 3])
    im = MaskedImage.init_from_channels_at_back(p)
    assert im.n_channels == 3
    assert im.height == 50
    assert im.width == 60


def test_init_from_rolled_channels_masked():
    p = np.empty([50, 60, 3])
    example_mask = BooleanImage.init_blank((50, 60), fill=False)
    example_mask.pixels[0, :6, :6] = True

    im = MaskedImage.init_from_channels_at_back(p, mask=example_mask)
    assert im.n_channels == 3
    assert im.height == 50
    assert im.width == 60
    assert im.mask.n_true() == 36
