from unittest.mock import PropertyMock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from pytest import raises

import menpo
import menpo.io as mio
from menpo.image import BooleanImage, Image, MaskedImage, OutOfMaskSampleError
from menpo.image.interpolation import cv2_perspective_interpolation
from menpo.shape import PointCloud, bounding_box
from menpo.transform import Affine, Rotation, UniformScale

CROP_COORDS = (np.array([70, 30]), np.array([169, 129]))


@pytest.fixture()
def interpolation_method(mocker, method):
    # This uses the pytest-mock package in order to provide a fixture that will
    # mock the global variable in the image module that determines if OpenCV
    # should be used or not. Useful for running the unit tests with and without
    # opencv
    interp = PropertyMock()
    interp.return_value = method
    mocker.patch("menpo.image.base.cv2_perspective_interpolation", new_callable=interp)


def opencv_and_scipy_interpolation(func):
    # This decorator combines the "interpolation_method" fixture from above
    # with a parameterize decorator to run unit tests with and without OpenCV
    parameterize = pytest.mark.parametrize(
        "method",
        [None, cv2_perspective_interpolation],
        ids=lambda x: "opencv" if x is not None else "scipy",
    )
    use_fixtures = pytest.mark.usefixtures("interpolation_method")
    return use_fixtures(parameterize(func))


@pytest.fixture()
def rgb_image():
    return mio.import_builtin_asset("takeo.ppm")


@pytest.fixture()
def gray_image():
    return mio.import_builtin_asset("takeo.ppm").as_greyscale()


@pytest.fixture()
def gray_template(gray_image):
    gray_template = gray_image.crop(*CROP_COORDS)
    mask = BooleanImage.init_blank(gray_template.shape)
    return gray_template, mask


@pytest.fixture()
def target_transform():
    initial_params = np.array([0, 0, 0, 0, 70, 30])
    return Affine.init_identity(2).from_vector(initial_params)


def test_warp_gray(gray_image, gray_template, target_transform):
    gray_template, mask = gray_template
    warped_im = gray_image.warp_to_mask(mask, target_transform)

    assert warped_im.shape == gray_template.shape
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_warp_gray_batch(gray_image, gray_template, target_transform):
    gray_template, mask = gray_template
    warped_im = gray_image.warp_to_mask(mask, target_transform, batch_size=100)

    assert warped_im.shape == gray_template.shape
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_warp_multi(rgb_image, target_transform):
    rgb_template = rgb_image.crop(*CROP_COORDS)
    mask = BooleanImage.init_blank(rgb_template.shape)
    warped_im = rgb_image.warp_to_mask(mask, target_transform)

    assert warped_im.shape == rgb_template.shape
    assert_allclose(warped_im.pixels, rgb_template.pixels)


def test_warp_to_mask_boolean():
    b = BooleanImage.init_blank((10, 10))
    b.pixels[:, :5] = False
    template_mask = BooleanImage.init_blank((10, 10))
    template_mask.pixels[:5, :] = False
    t = Affine.init_identity(2)
    warped_mask = b.warp_to_mask(template_mask, t)
    assert type(warped_mask) == BooleanImage
    result = template_mask.pixels.copy()
    result[:, :5] = False
    assert np.all(result == warped_mask.pixels)


def test_warp_to_mask_image():
    img = Image.init_blank((10, 10), n_channels=2)
    img.pixels[:, :, :5] = 0.5
    template_mask = BooleanImage.init_blank((10, 10))
    template_mask.pixels[:, 5:, :] = False
    t = Affine.init_identity(2)
    warped_img = img.warp_to_mask(template_mask, t)
    assert type(warped_img) == MaskedImage
    result = Image.init_blank((10, 10), n_channels=2).pixels
    result[:, :5, :5] = 0.5
    assert np.all(result == warped_img.pixels)


def test_warp_to_mask_masked_image():
    mask = BooleanImage.init_blank((15, 15))
    # make a truncated mask on the original image
    mask.pixels[0, -1, -1] = False
    img = MaskedImage.init_blank((15, 15), n_channels=2, mask=mask, fill=2.5)
    template_mask = BooleanImage.init_blank((10, 10), fill=False)
    template_mask.pixels[:, :5, :5] = True
    t = Affine.init_identity(2)
    warped_img = img.warp_to_mask(template_mask, t)
    assert type(warped_img) == MaskedImage

    result = Image.init_blank((10, 10), n_channels=2).pixels
    result[:, :5, :5] = 2.5
    result_mask = BooleanImage.init_blank((10, 10), fill=False).pixels
    result_mask[:, :5, :5] = True
    assert warped_img.n_true_pixels() == 25
    assert_allclose(result, warped_img.pixels)
    assert_allclose(result_mask, warped_img.mask.pixels)


def test_warp_to_mask_masked_image_all_true():
    img = MaskedImage.init_blank((10, 10), fill=2.5)

    template_mask = BooleanImage.init_blank((10, 10), fill=False)
    template_mask.pixels[:, :5, :5] = True
    t = Affine.init_identity(2)
    warped_img = img.warp_to_mask(template_mask, t)
    assert type(warped_img) == MaskedImage


def test_warp_to_shape_equal_warp_to_mask():
    r = menpo.transform.UniformScale(2.0, n_dims=2)
    b = mio.import_builtin_asset("breakingbad.jpg")
    m_shape = b.warp_to_shape((540, 960), r)
    m_mask = b.warp_to_mask(menpo.image.BooleanImage.init_blank((540, 960)), r)
    assert_allclose(m_shape.pixels, m_mask.pixels)


def test_warp_to_shape_batch():
    r = menpo.transform.Affine.init_identity(2)
    b = mio.import_builtin_asset("takeo.ppm")
    m_shape = b.warp_to_shape(b.shape, r, batch_size=100)
    assert_allclose(m_shape.pixels, b.pixels)


def test_rescale_boolean():
    mask = BooleanImage.init_blank((100, 100))
    mask.resize((10, 10))


def test_rescale_return_transform():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks["test"] = bounding_box([40, 40], [80, 80])
    cropped_img, transform = img.rescale(1.5, return_transform=True)
    img_back = cropped_img.warp_to_shape(img.shape, transform.pseudoinverse())
    assert_allclose(img_back.shape, img.shape)
    assert_allclose(img_back.pixels, img.pixels)
    assert_allclose(img_back.landmarks["test"].points, img.landmarks["test"].points)


def test_sample_image():
    im = Image.init_blank((100, 100), fill=2)
    p = PointCloud(np.array([[0, 0], [1, 0]]))

    arr = im.sample(p)
    assert_allclose(arr, [[2.0, 2.0]])


def test_sample_maskedimage():
    im = MaskedImage.init_blank((100, 100), fill=2)
    p = PointCloud(np.array([[0, 0], [1, 0]]))

    arr = im.sample(p)
    assert_allclose(arr, [[2.0, 2.0]])


def test_sample_maskedimage_error():
    m = np.zeros([100, 100], dtype=bool)
    im = MaskedImage.init_blank((100, 100), mask=m, fill=2)
    p = PointCloud(np.array([[0, 0], [1, 0]]))
    with raises(OutOfMaskSampleError):
        im.sample(p, verify_mask=True)


def test_sample_maskedimage_error_values():
    m = np.zeros([100, 100], dtype=bool)
    m[1, 0] = True
    im = MaskedImage.init_blank((100, 100), mask=m, fill=2)
    p = PointCloud(np.array([[0, 0], [1, 0]]))
    try:
        im.sample(p, verify_mask=True)
        # Expect exception!
        assert 0
    except OutOfMaskSampleError as e:
        sampled_mask = e.sampled_mask
        sampled_values = e.sampled_values
        assert_allclose(sampled_values, [[2.0, 2.0]])
        assert_allclose(sampled_mask, [[False, True]])


def test_sample_booleanimage():
    im = BooleanImage.init_blank((100, 100))
    im.pixels[0, 1, 0] = False
    p = PointCloud(np.array([[0, 0], [1, 0]]))

    arr = im.sample(p)
    assert_allclose(arr, [[True, False]])


@opencv_and_scipy_interpolation
def test_transform_about_centre(method):
    pixels_16 = np.arange(16, dtype=float)
    image = Image(pixels_16.reshape(4, 4))
    transform = Rotation.init_from_2d_ccw_angle(180).compose_before(
        UniformScale(2, n_dims=2)
    )
    # rotate 180 + scale degrees
    transformed_img = image.transform_about_centre(transform, mode="nearest", order=1)
    expected_pixels = np.concatenate(
        [np.linspace(15 - 2 * i, 12 - 2 * i, num=7)[None] for i in range(7)]
    )

    assert transformed_img.shape == (7, 7)
    assert_allclose(transformed_img.pixels[0], expected_pixels, rtol=1e-8, atol=1e-8)


@opencv_and_scipy_interpolation
def test_zoom_image():
    im = Image.init_blank((100, 100), fill=0)
    # White square in the centre of size 10x10
    im.pixels[0, 45:55, 45:55] = 1.0

    # Zoom in 50% makes the white square 5 pixel bigger in theory (16x16)
    zim = im.zoom(1.5)
    assert np.count_nonzero(zim.pixels) == 256


def test_zoom_booleanimage():
    im = BooleanImage.init_blank((100, 100))
    im.pixels[0, 0, :] = False
    im.pixels[0, -1, :] = False
    im.pixels[0, :, 0] = False
    im.pixels[0, :, -1] = False

    zim = im.zoom(1.2)
    assert np.all(zim.pixels)


@opencv_and_scipy_interpolation
def test_mirror_horizontal_image():
    image = Image(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    )
    image.landmarks["temp"] = PointCloud(
        np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    )
    mirrored_img = image.mirror(axis=0)
    assert_allclose(
        mirrored_img.pixels,
        np.array(
            [[[9.0, 10.0, 11.0, 12.0], [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]]]
        ),
    )
    assert_allclose(
        mirrored_img.landmarks["temp"].points,
        np.array([[1.0, 1.0], [1.0, 2.0], [0.0, 1.0], [0.0, 2.0]]),
    )


@opencv_and_scipy_interpolation
def test_mirror_vertical_image():
    image = Image(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    )
    image.landmarks["temp"] = PointCloud(
        np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
    )
    mirrored_img = image.mirror()
    assert_allclose(
        mirrored_img.pixels,
        np.array(
            [[[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0], [12.0, 11.0, 10.0, 9.0]]]
        ),
    )
    assert_allclose(
        mirrored_img.landmarks["temp"].points,
        np.array([[1.0, 3.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]]),
    )


def test_mirror_image_axis_error():
    with raises(ValueError):
        Image(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])).mirror(axis=2)


def test_mirror_masked_image():
    image = MaskedImage(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    mirrored_img = image.mirror()
    assert type(mirrored_img) == MaskedImage


def test_mirror_return_transform():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks["test"] = bounding_box([40, 40], [80, 80])
    cropped_img, transform = img.mirror(return_transform=True)
    img_back = cropped_img.warp_to_shape(img.shape, transform.pseudoinverse())
    assert_allclose(img_back.shape, img.shape)
    assert_allclose(img_back.pixels, img.pixels)
    assert_allclose(img_back.landmarks["test"].points, img.landmarks["test"].points)


@opencv_and_scipy_interpolation
def test_rotate_image_90_180():
    image = Image(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    )
    image.landmarks["temp"] = PointCloud(
        np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    )
    # rotate 90 degrees
    rotated_img = image.rotate_ccw_about_centre(theta=90, order=1, mode="nearest")
    rotated_img.landmarks["temp"] = rotated_img.landmarks["temp"].constrain_to_bounds(
        rotated_img.bounds()
    )
    assert_allclose(
        rotated_img.pixels,
        np.array(
            [[[4.0, 8.0, 12.0], [3.0, 7.0, 11.0], [2.0, 6.0, 10.0], [1.0, 5.0, 9.0]]]
        ),
    )
    assert_almost_equal(
        rotated_img.landmarks["temp"].points,
        np.array([[2.0, 1.0], [1.0, 1.0], [2.0, 2.0], [1.0, 2.0]]),
    )

    # rotate 180 degrees
    rotated_img = image.rotate_ccw_about_centre(theta=180, order=1, mode="nearest")
    rotated_img.landmarks["temp"] = rotated_img.landmarks["temp"].constrain_to_bounds(
        rotated_img.bounds()
    )
    assert_allclose(
        rotated_img.pixels,
        np.array(
            [[[12.0, 11.0, 10.0, 9.0], [8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]]
        ),
    )
    assert_almost_equal(
        rotated_img.landmarks["temp"].points,
        np.array([[1.0, 2.0], [1.0, 1.0], [0.0, 2.0], [0.0, 1.0]]),
    )


def test_rotate_image_45():
    image = Image(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
    )
    image.landmarks["temp"] = PointCloud(
        np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    )
    rotated_img = image.rotate_ccw_about_centre(theta=45, order=0)
    assert_allclose(
        rotated_img.pixels,
        np.array(
            [
                [
                    [0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, 3.0, 7.0, 8.0, 0.0],
                    [1.0, 6.0, 7.0, 11.0, 16.0],
                    [0.0, 5.0, 10.0, 15.0, 15.0],
                    [0.0, 0.0, 13.0, 14.0, 0.0],
                ]
            ]
        ),
    )
    assert_almost_equal(
        rotated_img.landmarks["temp"].points,
        np.array([[2.121, 1.414], [1.414, 2.121], [2.828, 2.121], [2.121, 2.828]]),
        decimal=3,
    )


def test_rotate_return_transform():
    img = Image.init_blank((100, 100), n_channels=1)
    img.landmarks["test"] = bounding_box([40, 40], [80, 80])
    cropped_img, transform = img.rotate_ccw_about_centre(60, return_transform=True)
    img_back = cropped_img.warp_to_shape(img.shape, transform.pseudoinverse())
    assert_allclose(img_back.shape, img.shape)
    assert_allclose(img_back.pixels, img.pixels)
    assert_allclose(img_back.landmarks["test"].points, img.landmarks["test"].points)


@opencv_and_scipy_interpolation
def test_maskedimage_retain_shape():
    image = mio.import_builtin_asset("takeo.ppm")
    image = image.as_masked()
    rotated_img = image.rotate_ccw_about_centre(theta=77, retain_shape=True)
    assert image.shape == rotated_img.shape
    assert type(rotated_img) == MaskedImage
